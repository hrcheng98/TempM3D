import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, BasicEncoderQuarter
from .utils.utils import bilinear_sampler, coords_grid, upflow4
from .path_match import PathMatch

import argparse

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class DIP(nn.Module):
    def __init__(self, max_offset=192, mixed_precision=False, test_mode=False):
        super(DIP, self).__init__()

        self.max_offset = max_offset
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = 128
        self.context_dim = 128

        self.dropout = 0
        self.iters = 5

        # feature network, and update block
        self.fnet = BasicEncoderQuarter(output_dim=256, norm_fn='instance', dropout=self.dropout)

        self.update_block_s = SmallUpdateBlock(hidden_dim=self.hidden_dim)
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def random_init_flow(self, fmap, max_offset, test_mode = False):
        N, C, H, W = fmap.shape
        if test_mode:
            init_seed = 20
            torch.manual_seed(init_seed)
            torch.cuda.manual_seed(init_seed)
        flow = (torch.rand(N, 2, H, W) - 0.5) * 2
        flow = flow.to(fmap.device) * max_offset
        return flow

    def upsample_flow(self, flow, mask, rate=4):
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def build_pyramid(self, fmap1, fmap2, cnet, max_layers = 5,  min_width = 40):
        py_fmap1 = []
        py_fmap2 = []
        py_cnet = []
        py_fmap1.append(fmap1)
        py_fmap2.append(fmap2)
        py_cnet.append(cnet)

        curr_fmap1 = fmap1
        curr_fmap2 = fmap2
        curr_cnet = cnet
        for i in range(max_layers - 1):
            if (curr_fmap1.shape[2] < min_width) and (curr_fmap1.shape[3] < min_width):
                break
            down_scale = 2**(i + 1)
            curr_fmap1 = F.avg_pool2d(curr_fmap1, 2, stride=2)
            curr_fmap2 = F.avg_pool2d(curr_fmap2, 2, stride=2)
            curr_cnet = F.avg_pool2d(curr_cnet, 2, stride=2)
            py_fmap1.append(curr_fmap1)
            py_fmap2.append(curr_fmap2)
            py_cnet.append(curr_cnet)
        
        return py_fmap1, py_fmap2, py_cnet

    def upflow(self, flow, targetMap, mode='bilinear'):
        """ Upsample flow """
        new_size = (targetMap.shape[2], targetMap.shape[3])
        factor = 1.0 * targetMap.shape[2] / flow.shape[2]
        return  factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def inference(self, image1, image2, iters = 6, max_layers = 3, init_flow = None):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        max_layers = 3
        max_offset = 256
        auto_layer = 3
        if init_flow is not None:
            mag = torch.norm(init_flow, dim=1)
            max_offset = torch.max(mag)
            auto_layer = max_offset / 32
            auto_layer = int(auto_layer.ceil().cpu().numpy())
            if auto_layer < max_layers:
                max_layers = auto_layer
            print('mag:', max_offset, 'layers:', max_layers)

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
            new_size = (image1.shape[2], image1.shape[3])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        #build layers
        min_width = 40
        py_fmap1, py_fmap2, py_cnet = self.build_pyramid(fmap1, fmap2, fmap1, max_layers = max_layers,  min_width = min_width)
        n_levels = len(py_fmap1)

        #build layers
        s_fmap1 = py_fmap1[n_levels - 1]
        #init
        if init_flow is not None:
            new_size = (s_fmap1.shape[2], s_fmap1.shape[3])
            scale = s_fmap1.shape[2] / (init_flow.shape[2] * 1.0)
            s_flow =  scale *  F.interpolate(init_flow, size=new_size, mode='bilinear', align_corners=True)
            initail_flow_max = 2**(auto_layer + 1) * 1.0
            noise = self.random_init_flow(s_fmap1, max_offset= 16 , test_mode = self.test_mode)
            s_flow = s_flow + noise
        else:
            scale = 2**(n_levels - 1 + 2) * 1.0
            s_flow = self.random_init_flow(s_fmap1, max_offset=self.max_offset / scale, test_mode = self.test_mode)


        up_mask = None
        for i in range(n_levels):
            # print('i: ', i)
            curr_fmap1 = py_fmap1[n_levels - i - 1]
            curr_fmap2 = py_fmap2[n_levels - i - 1]
            curr_cnet = py_cnet[n_levels - i - 1]
            patch_fn = PathMatch(curr_fmap1, curr_fmap2)
            with autocast(enabled=self.mixed_precision):
                net, inp = torch.split(curr_cnet, [self.hidden_dim, self.context_dim], dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)
            if i > 0:
                s_flow = self.upflow(flow_up, curr_fmap1)
                noise = self.random_init_flow(curr_fmap1, max_offset= 4, test_mode = self.test_mode)
                s_flow = s_flow + noise

            for itr in range(iters):
                s_flow = s_flow.detach()
                out_corrs = patch_fn(s_flow, is_search=False)

                with autocast(enabled=self.mixed_precision):
                    net, up_mask, delta_flow = self.update_block_s(net, inp, out_corrs, s_flow)

                s_flow = s_flow + delta_flow

                s_flow = s_flow.detach()
                out_corrs = patch_fn(s_flow, is_search = True)
                with autocast(enabled=self.mixed_precision):
                    net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, s_flow)

                s_flow = s_flow + delta_flow
                flow_up = self.upsample_flow(s_flow, up_mask, rate=4)

        return flow_up


    def forward(self, image1, image2 = None, iters = 6, init_flow = None):
        """ Estimate optical flow between pair of frames """

        if self.test_mode and init_flow is not None:
            flow_up = self.inference(image1, image2, iters = iters, init_flow = init_flow)
            return flow_up

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # run the context network
        with autocast(enabled=self.mixed_precision):
            net, inp = torch.split(fmap1, [self.hidden_dim, self.context_dim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            # 1/4 -> 1/16
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            s_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)
            # context(left)
            s_net = F.avg_pool2d(net, 4, stride=4)
            s_inp = F.avg_pool2d(inp, 4, stride=4)

        # 1/16
        s_patch_fn = PathMatch(s_fmap1, s_fmap2)

        # init flow
        s_flow = None
        s_flow = self.random_init_flow(s_fmap1, max_offset=self.max_offset // 16, test_mode = self.test_mode)
       
        # small initial: 1/16
        flow = None
        flow_up = None
        flow_predictions = []
        for itr in range(iters):
            # --------------- update1 ---------------
            s_flow = s_flow.detach()
            out_corrs = s_patch_fn(s_flow, is_search=False)

            with autocast(enabled=self.mixed_precision):
                s_net, up_mask, delta_flow = self.update_block_s(s_net, s_inp, out_corrs, s_flow)

            s_flow = s_flow + delta_flow
            flow = self.upsample_flow(s_flow, up_mask, rate=4)
            flow_up = upflow4(flow)
            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            s_flow = s_flow.detach()
            out_corrs = s_patch_fn(s_flow, is_search=True)

            with autocast(enabled=self.mixed_precision):
                s_net, up_mask, delta_flow = self.update_block(s_net, s_inp, out_corrs, s_flow)

            s_flow = s_flow + delta_flow
            flow = self.upsample_flow(s_flow, up_mask, rate=4)
            flow_up =  upflow4(flow)
            flow_predictions.append(flow_up)

        patch_fn = PathMatch(fmap1, fmap2)
        # large refine: 1/4
        for itr in range(iters):
            # --------------- update1 ---------------
            flow = flow.detach()
            out_corrs = patch_fn(flow, is_search=False)
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block_s(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = self.upsample_flow(flow, up_mask, rate=4)
            flow_predictions.append(flow_up)

            # --------------- update2 ---------------
            flow = flow.detach()
            out_corrs = patch_fn(flow, is_search=True)
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = self.upsample_flow(flow, up_mask, rate=4)
            flow_predictions.append(flow_up)

        if self.test_mode:
            return flow_up

        return flow_predictions
