import torch
import torch.nn as nn
import numpy as np

from lib.backbones.resnet import resnet50
from lib.backbones.dla import dla34
from lib.backbones.dlaup import DLAUp
from lib.backbones.dlaup import DLAUpv2

import torchvision.ops.roi_align as roi_align
from lib.losses.loss_function import extract_input_from_tensor
from lib.helpers.decode_helper import _topk,_nms

from datetime import datetime
import torch.nn.functional as F
from lib.backbones.resnet import Bottleneck
import cv2 as cv

_delete = False
_use_frames = 3

def deprocess(t):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean, std = mean[[2, 0, 1]], std[[2, 0, 1]]
    t = np.transpose(t, [1, 2, 0])
    t = ((t * std) + mean) * 255.
    return t.astype(np.uint8)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def warpFeatFlowBK(feat, SSIM_mask, u_offset, v_offset):
    b, c, h, w = feat.shape
    coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, coords_v, coords_u = coords_b.to(feat.device), coords_v.to(feat.device), coords_u.to(feat.device)

    scale_factor_v = h / u_offset.shape[-2]
    scale_factor_u = w / u_offset.shape[-1]

    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    SSIM_mask = F.interpolate(SSIM_mask, (h, w), mode='nearest')

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = torch.clamp(coords_u + u_offset.squeeze(), 0, w - 2)
    coords_v_trans = torch.clamp(coords_v + v_offset.squeeze(), 0, h - 2)

    new_u_low = torch.floor(coords_u_trans)
    new_u_high = torch.floor(coords_u_trans) + 1
    new_v_low = torch.floor(coords_v_trans)
    new_v_high = torch.floor(coords_v_trans) + 1

    weight_u_low = 1 - (coords_u_trans - new_u_low)
    weight_u_high = (coords_u_trans - new_u_low)
    weight_v_low = 1 - (coords_v_trans - new_v_low)
    weight_v_high = (coords_v_trans - new_v_low)

    weight_v_low_u_low = weight_v_low * weight_u_low
    weight_v_low_u_high = weight_v_low * weight_u_high
    weight_v_high_u_low = weight_v_high * weight_u_low
    weight_v_high_u_high = weight_v_high * weight_u_high

    feat_warp = feat[coords_b.view(-1).type(torch.long), :, new_v_low.view(-1).type(torch.long),
                    new_u_low.view(-1).type(torch.long)].view(b, h * w, c).permute(0, 2, 1) * weight_v_low_u_low.view(b, 1, h * w) + \
                feat[coords_b.view(-1).type(torch.long), :, new_v_low.view(-1).type(torch.long),
                    new_u_high.view(-1).type(torch.long)].view(b, h * w, c).permute(0, 2, 1) * weight_v_low_u_high.view(b, 1, h * w) + \
                feat[coords_b.view(-1).type(torch.long), :, new_v_high.view(-1).type(torch.long),
                    new_u_low.view(-1).type(torch.long)].view(b, h * w, c).permute(0, 2, 1) * weight_v_high_u_low.view(b, 1, h * w) + \
                feat[coords_b.view(-1).type(torch.long), :, new_v_high.view(-1).type(torch.long),
                    new_u_high.view(-1).type(torch.long)].view(b, h * w, c).permute(0, 2, 1) * weight_v_high_u_high.view(b, 1, h * w)

    feat_warp = feat_warp.view(b, c, h, w) * SSIM_mask


    '''cat flow'''
    feat_warp = torch.cat([feat_warp, u_offset, v_offset], dim=1)

    return feat_warp


class GUPNet(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, mean_size=None):
        assert downsample in [4, 8, 16, 32]
        super().__init__()


        self.backbone = globals()[backbone](pretrained=True, return_levels=True)
        self.head_conv = 256  # default setting for head conv
        self.mean_size = nn.Parameter(torch.tensor(mean_size,dtype=torch.float32),requires_grad=False)
        self.cls_num = mean_size.shape[0]
        # channels = self.backbone.channels  # channels list for feature maps generated by backbone
        channels = [16, 32, 64+2, 128, 256, 512]
        if _delete:
            channels = [16, 32, 64, 128, 256, 512]
        # channels = [16, 32, 64, 128, 256, 512]
        # channels = [16, 32, 64*3+4, 128, 256, 512]
        # channels = [16, 32, 64, 128, 256, 512]


        self.first_level = int(np.log2(downsample))
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.feat_up = globals()[neck](channels[self.first_level:], scales_list=scales)

        # initialize the head of pipeline, according to heads setting.
        self.heatmap = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        self.offset_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_2d = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))


        self.pixel_depth = nn.Sequential(nn.Conv2d(channels[self.first_level], self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))


        self.depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     # nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
                                     nn.Conv2d(self.head_conv, 2*4+4, kernel_size=1, stride=1, padding=0, bias=True))

        ####################################################################
        # self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        # # self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        # #                              nn.BatchNorm2d(self.head_conv*2),
        # #                              nn.LeakyReLU(inplace=True),
        # #                              nn.Conv2d(self.head_conv*2, self.head_conv, kernel_size=3, padding=1, bias=True),
        # #                              nn.BatchNorm2d(self.head_conv),
        # #                              nn.LeakyReLU(inplace=True),
        # #                              nn.Conv2d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))

        '''LOCAL DENSE NOC depths'''
        self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        # self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv*2, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        self.NOC_depth_offset = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        # self.NOC_depth_offset = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv * 2, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        self.NOC_depth_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        # self.NOC_depth_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv * 2, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        self.NOC_depth_offset_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        # self.NOC_depth_offset_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv * 2, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
                                     nn.LeakyReLU(inplace=True),
                                     nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        ####################################################################


        self.offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.size_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 4, kernel_size=1, stride=1, padding=0, bias=True))
        # self.size_3d_v2 = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              # nn.BatchNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
        #                              nn.Conv2d(self.head_conv, 4, kernel_size=1, stride=1, padding=0, bias=True))
        self.heading = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))


        self.corners_offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 8*2, kernel_size=1, stride=1, padding=0, bias=True))

        # self.fuse_layers = nn.Sequential(
        #     nn.Conv2d(66, 64*4, kernel_size=1, stride=1,  bias=False),
        #     nn.BatchNorm2d(64*4),
        #     Bottleneck(64*4, 64),
        #     nn.Conv2d(64*4, 64, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        #     )


        # init layers
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)

        self.depth.apply(weights_init_xavier)
        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        # self.size_3d_v2.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)

        self.NOC_depth.apply(weights_init_xavier)
        self.NOC_depth_offset.apply(weights_init_xavier)
        self.NOC_depth_uncern.apply(weights_init_xavier)
        self.NOC_depth_offset_uncern.apply(weights_init_xavier)

        # self.fuse_layers.apply(weights_init_xavier)


    def forward(self, input, coord_ranges,calibs, targets=None, K=50, mode='train'):
        # device_id = input.device
        # BATCH_SIZE = input.size(0)
        #
        # feat = self.backbone(input)
        # feat = self.feat_up(feat[self.first_level:])
        #

        # img = input['rgb_imgs'][:, 0, ...]
        # pre_1 = input['rgb_imgs'][:, 1, ...]
        # pre_2 = input['rgb_imgs'][:, 2, ...]
        # pre_1_flow = input['flow_imgs'][:, 0, ...]
        # pre_2_flow = input['flow_imgs'][:, 1, ...]
        # pre_1_flow_bk = input['flow_imgs'][:, 2, ...]
        # pre_2_flow_bk = input['flow_imgs'][:, 3, ...]
        img = input['img']
        pre_1 = input['pre_1']
        pre_2 = input['pre_2']
        pre_1_flow = input['pre_1_flow']
        pre_2_flow = input['pre_2_flow']
        pre_1_flow_bk = input['pre_1_flow_bk']
        pre_2_flow_bk = input['pre_2_flow_bk']
        pre_3 = input['pre_3']
        pre_4 = input['pre_4']
        pre_3_flow = input['pre_3_flow']
        pre_4_flow = input['pre_4_flow']
        pre_3_flow_bk = input['pre_3_flow_bk']
        pre_4_flow_bk = input['pre_4_flow_bk']
        device_id = img.device

        # torch.cuda.synchronize()
        # a = datetime.now()
        recover_1 = warpFeatFlowBK(pre_1_flow, torch.ones_like(pre_1_flow[:, :1, ...]), pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...])
        recover_2 = warpFeatFlowBK(pre_2_flow, torch.ones_like(pre_2_flow[:, :1, ...]), pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...])
        recover_3 = warpFeatFlowBK(pre_3_flow, torch.ones_like(pre_3_flow[:, :1, ...]), pre_3_flow_bk[:, :1, ...], pre_3_flow_bk[:, 1:2, ...])
        # recover_4 = warpFeatFlowBK(pre_4_flow, torch.ones_like(pre_4_flow[:, :1, ...]), pre_4_flow_bk[:, :1, ...], pre_4_flow_bk[:, 1:2, ...])
        recover_1 = -recover_1[:, :2, ...]
        recover_2 = -recover_2[:, :2, ...]
        recover_3 = -recover_3[:, :2, ...]
        # recover_4 = -recover_4[:, :2, ...]
        pre1_mask = (((recover_1[:, 0:1, ...] - pre_1_flow_bk[:, 0:1, ...])**2 +
                      (recover_1[:, 1:2, ...] - pre_1_flow_bk[:, 1:2, ...])**2) < 9).type(torch.float32)
        pre2_mask = (((recover_2[:, 0:1, ...] - pre_2_flow_bk[:, 0:1, ...])**2 +
                      (recover_2[:, 1:2, ...] - pre_2_flow_bk[:, 1:2, ...])**2) < 9).type(torch.float32)
        pre3_mask = (((recover_3[:, 0:1, ...] - pre_3_flow_bk[:, 0:1, ...])**2 +
                      (recover_3[:, 1:2, ...] - pre_3_flow_bk[:, 1:2, ...])**2) < 9).type(torch.float32)
        

        # pre4_mask = (((recover_4[:, 0:1, ...] - pre_4_flow_bk[:, 0:1, ...])**2 +
        #               (recover_4[:, 1:2, ...] - pre_4_flow_bk[:, 1:2, ...])**2) < 9).type(torch.float32)
        # torch.cuda.synchronize()
        # b = datetime.now()
        # print('flow', b, b-a)

        # torch.cuda.synchronize()
        # a = datetime.now()
        feat = self.backbone(img)[2:]

        # torch.cuda.synchronize()
        # b = datetime.now()
        # print('backbone', b, b-a)

        if not _delete:
            self.backbone.eval()
            '''propagate'''
            # cur_factor, pre_factor = 1/2, 1/2
            # cur_factor, pre_factor = 1/3, 2/3
            cur_factor, pre_factor = 2/3, 1/3

            # print('***1', recover_1.shape)
            # print('***2', pre_1_flow.shape)
            # print('***3', pre_1_flow_bk.shape)
            # print('***4', feat[0].shape)
            

            if _use_frames >2:
                with torch.no_grad():
                    feat_pre3_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_3))))]
                feat_pre3_bk = [warpFeatFlowBK(i.detach(), pre3_mask, pre_3_flow_bk[:, :1, ...], pre_3_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre3_base)]
                
            if _use_frames >1:
                with torch.no_grad():
                    feat_pre2_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_2))))]
                if _use_frames>2:
                    feat_pre2_base = [cur_factor * feat_pre2_base[0] + pre_factor * feat_pre3_bk[0][:, :-2, ...]]
                feat_pre2_bk = [warpFeatFlowBK(i.detach(), pre2_mask, pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]
            
            if _use_frames >0:
                with torch.no_grad():
                    feat_pre1_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_1))))]
                if _use_frames>1:
                    feat_pre1_base = [cur_factor * feat_pre1_base[0] + pre_factor * feat_pre2_bk[0][:, :-2, ...]]
                feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]

            # feat_pre2_bk = [warpFeatFlowBK(i.detach(), torch.ones_like(pre2_mask), pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]
            # feat_pre2_bk = [warpFeatFlowBK(i.detach(), torch.ones_like(pre2_mask), torch.zeros_like(pre_2_flow_bk[:, :1, ...]), torch.zeros_like(pre_2_flow_bk[:, :1, ...])) for ind, i in enumerate(feat_pre2_base)]

            
            self.backbone.train()

            
            

            

            # feat_pre1_base = [cur_factor * feat_pre1_base[0] + pre_factor * cur_factor * feat_pre2_bk[0][:, :-2, ...]]
            
            # feat_pre1_bk = [warpFeatFlowBK(i.detach(), torch.ones_like(pre1_mask), pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
            # feat_pre1_bk = [warpFeatFlowBK(i.detach(), torch.ones_like(pre1_mask), torch.zeros_like(pre_2_flow_bk[:, :1, ...]), torch.zeros_like(pre_2_flow_bk[:, :1, ...])) for ind, i in enumerate(feat_pre1_base)]

            feat[0] = torch.cat([cur_factor*feat[0] + pre_factor*feat_pre1_bk[0][:, :-2, ...],
                                    pre_factor*feat_pre1_bk[0][:, -2:, ...] + \
                                    pre_factor*pre_factor*feat_pre2_bk[0][:, -2:, ...] + \
                                    pre_factor*pre_factor*pre_factor*feat_pre3_bk[0][:, -2:, ...]
                                ], dim=1)
        # feat[0] = cur_factor * feat[0] + pre_factor * feat_pre1_bk[0][:, :-2, ...]

        # '''1 frame'''
        # feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
        #
        # feat[0] = torch.cat([cur_factor * feat[0] + pre_factor * feat_pre1_bk[0][:, :-2, ...],
        #                      pre_factor * feat_pre1_bk[0][:, -2:, ...]], dim=1)
        #


        # feat[0] = torch.cat([cur_factor * feat[0] + pre_factor * feat[0],
        #                      0 * feat_pre1_bk[0][:, -2:, ...] + 0*feat_pre2_bk[0][:, -2:, ...]], dim=1)

        # feat_pre1_base = [torch.cat([feat_pre2_bk[0], feat_pre1_base[0]], dim=1)]
        # feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
        # feat[0] = torch.cat([feat_pre1_bk[0], feat[0]], dim=1)

        feat = self.feat_up(feat)


        # feat = [self.feat_up(self.backbone(img)[2:])]
        # self.backbone.eval()
        # self.feat_up.eval()
        # with torch.no_grad():
        #     feat_pre2_base = [self.feat_up(self.backbone(pre_2)[2:])]
        #     feat_pre1_base = [self.feat_up(self.backbone(pre_1)[2:])]
        # self.backbone.train()
        # self.feat_up.train()
        #
        # cur_factor, pre_factor = 1/2, 1/2
        # feat_pre2_bk = [warpFeatFlowBK(i.detach(), pre2_mask, pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]
        # feat_pre1_base = [cur_factor * feat_pre1_base[0] + pre_factor * feat_pre2_bk[0][:, :-2, ...]]
        # feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
        # feat[0] = torch.cat([cur_factor * feat[0] + pre_factor * feat_pre1_bk[0][:, :-2, ...],
        #                      pre_factor * feat_pre1_bk[0][:, -2:, ...] + pre_factor*pre_factor*feat_pre2_bk[0][:, -2:, ...]], dim=1)
        #
        # feat = self.fuse_layers(feat[0])


        ret = {}
        '''
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feat)
        '''
        ret['heatmap']=self.heatmap(feat)
        ret['offset_2d']=self.offset_2d(feat)
        ret['size_2d']=self.size_2d(feat)

        ret['pixel_depth']=self.pixel_depth(feat)

        # torch.cuda.synchronize()
        #two stage
        # assert(mode in ['train','val','test'])
        assert(mode in ['train','val','test'])
        if mode=='train':   #extract train structure in the train (only) and the val mode
            inds,cls_ids = targets['indices'],targets['cls_ids']
            masks = targets['mask_2d']
            # masks = targets['mask_2d'].type(torch.bool)
        else:    #extract test structure in the test (only) and the val mode
            inds,cls_ids = _topk(_nms(torch.clamp(ret['heatmap'].sigmoid(), min=1e-4, max=1 - 1e-4)), K=K)[1:3]
            # if torch.__version__ == '1.10.0+cu113':
            if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0', '1.4.0']:
                masks = torch.ones(inds.size()).type(torch.bool).to(device_id)
            else:
                masks = torch.ones(inds.size()).type(torch.uint8).to(device_id)

        ret.update(self.get_roi_feat(feat,inds,masks,ret,calibs,coord_ranges,cls_ids))
        return ret


    def get_roi_feat_by_mask(self,feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        res = {}
        RoI_align_size = 7

        if num_masked_bin!=0:
            #get box2d of each roi region
            # box2d_masked = extract_input_from_tensor(box2d_maps,inds,mask)
            scale_box2d_masked = extract_input_from_tensor(box2d_maps,inds,mask)
            #get roi feature
            # roi_feature_masked = roi_align(feat,box2d_masked,[7,7])
            # roi_feature_masked = roi_align(feat,scale_box2d_masked,[7,7])
            roi_feature_masked = roi_align(feat,scale_box2d_masked,[RoI_align_size,RoI_align_size])
            #get coord range of each roi
            # coord_ranges_mask2d = coord_ranges[box2d_masked[:,0].long()]
            coord_ranges_mask2d = coord_ranges[scale_box2d_masked[:,0].long()]

            #map box2d coordinate from feature map size domain to original image size domain
            # box2d_masked = torch.cat([box2d_masked[:,0:1],
            #            box2d_masked[:,1:2]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
            #            box2d_masked[:,2:3]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2],
            #            box2d_masked[:,3:4]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
            #            box2d_masked[:,4:5]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2]],1)
            box2d_masked = torch.cat([scale_box2d_masked[:,0:1],
                       scale_box2d_masked[:,1:2]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,2:3]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2],
                       scale_box2d_masked[:,3:4]/WIDE  *(coord_ranges_mask2d[:,1,0:1]-coord_ranges_mask2d[:,0,0:1])+coord_ranges_mask2d[:,0,0:1],
                       scale_box2d_masked[:,4:5]/HEIGHT*(coord_ranges_mask2d[:,1,1:2]-coord_ranges_mask2d[:,0,1:2])+coord_ranges_mask2d[:,0,1:2]],1)
            roi_calibs = calibs[box2d_masked[:,0].long()]
            #project the coordinate in the normal image to the camera coord by calibs
            coords_in_camera_coord = torch.cat([self.project2rect(roi_calibs,torch.cat([box2d_masked[:,1:3],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2],
                                          self.project2rect(roi_calibs,torch.cat([box2d_masked[:,3:5],torch.ones([num_masked_bin,1]).to(device_id)],-1))[:,:2]],-1)
            coords_in_camera_coord = torch.cat([box2d_masked[:,0:1],coords_in_camera_coord],-1)
            #generate coord maps
            # coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/6 for i in range(7)],-1).unsqueeze(1).repeat([1,7,1]).unsqueeze(1),
            #                     torch.cat([coords_in_camera_coord[:,2:3]+i*(coords_in_camera_coord[:,4:5]-coords_in_camera_coord[:,2:3])/6 for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1)],1)
            coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/(RoI_align_size-1) for i in range(RoI_align_size)],-1).unsqueeze(1).repeat([1,RoI_align_size,1]).unsqueeze(1),
                                torch.cat([coords_in_camera_coord[:,2:3]+i*(coords_in_camera_coord[:,4:5]-coords_in_camera_coord[:,2:3])/(RoI_align_size-1) for i in range(RoI_align_size)],-1).unsqueeze(2).repeat([1,1,RoI_align_size]).unsqueeze(1)],1)


            #concatenate coord maps with feature maps in the channel dim
            cls_hots = torch.zeros(num_masked_bin,self.cls_num).to(device_id)
            cls_hots[torch.arange(num_masked_bin).to(device_id),cls_ids[mask].long()] = 1.0
            
            # roi_feature_masked = torch.cat([roi_feature_masked,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)
            roi_feature_masked = torch.cat([roi_feature_masked,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,RoI_align_size,RoI_align_size])],1)

            #compute heights of projected objects
            box2d_height = torch.clamp(box2d_masked[:,4]-box2d_masked[:,2],min=1.0)
            scale_depth = torch.clamp((scale_box2d_masked[:,4]-scale_box2d_masked[:,2])*4, min=1.0) / \
                          torch.clamp(box2d_masked[:,4]-box2d_masked[:,2], min=1.0)

            #compute real 3d height
            size3d_offset = self.size_3d(roi_feature_masked)[:,:,0,0]
            # size3d_offset = self.size_3d_v2(roi_feature_masked)[:,:,0,0]
            h3d_log_std = size3d_offset[:,3:4]
            size3d_offset = size3d_offset[:,:3] 

            size_3d = (self.mean_size[cls_ids[mask].long()]+size3d_offset)
            depth_geo = size_3d[:,0]/box2d_height.squeeze()*roi_calibs[:,0,0]
            
            # depth_net_out = self.depth(roi_feature_masked)[:,:,0,0]
            # depth_geo_log_std = (h3d_log_std.squeeze()+2*(roi_calibs[:,0,0].log()-box2d_height.log())).unsqueeze(-1)
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,1:2],depth_geo_log_std],-1),-1,keepdim=True)
            #
            # depth_net_out = torch.cat([(1. / (depth_net_out[:,0:1].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)

            # if torch.sum(depth_net_out) != torch.sum(depth_net_out):
            #     c = 1

            '''direct instance depth'''
            depth_net_out = self.depth(roi_feature_masked)[:,:,0,0]
            depth_net_log_std = depth_net_out[:,1:2]
            depth_net_out = torch.cat([depth_net_out[:, 0:1] * scale_depth.unsqueeze(-1), depth_net_log_std], -1)
            # '''do not scale instance depth'''
            # depth_net_out = torch.cat([depth_net_out[:, 0:1], depth_net_log_std], -1)


            ####################################################################
            '''LOCAL DENSE NOC depths'''
            NOC_depth = self.NOC_depth(roi_feature_masked)
            NOC_depth_offset = self.NOC_depth_offset(roi_feature_masked)


            '''do not use geo depth'''
            noc_depth_out = NOC_depth[:, 0, :, :]
            noc_depth_out = (-noc_depth_out).exp()
            noc_depth_out = noc_depth_out * scale_depth.unsqueeze(-1).unsqueeze(-1)
            noc_depth_offset_out = NOC_depth_offset[:, 0, :, :]
            # noc_depth_offset_out = noc_depth_offset_out * scale_depth.unsqueeze(-1).unsqueeze(-1)


            noc_depth_out_uncern = self.NOC_depth_uncern(roi_feature_masked)[:, 0, :, :]
            noc_depth_offset_out_uncern = self.NOC_depth_offset_uncern(roi_feature_masked)[:, 0, :, :]

            noc_merge_depth_out = noc_depth_out + noc_depth_offset_out
            noc_merge_depth_out_uncern = torch.logsumexp(torch.stack([noc_depth_offset_out_uncern, noc_depth_out_uncern],
                                                                     -1), -1)

            K, _, _ = noc_merge_depth_out_uncern.shape
            merge_prob = (-(0.5 * noc_merge_depth_out_uncern).exp()).exp()
            merge_depth = (torch.sum((noc_merge_depth_out * merge_prob).view(K, -1), dim=-1) /
                           torch.sum(merge_prob.view(K, -1), dim=-1))
            # merge_prob = (torch.sum(merge_prob.view(K, -1)**2, dim=-1) / \
            #               torch.sum(merge_prob.view(K, -1), dim=-1))
            # merge_prob = torch.log(-torch.log(merge_prob)) * 2
            merge_prob = (noc_merge_depth_out_uncern.view(K, -1)).min(dim=1)[0]
            # print('merge_prob:', merge_prob)


            res['train_tag'] = torch.ones(num_masked_bin).type(torch.bool).to(device_id)
            res['heading'] = self.heading(roi_feature_masked)[:,:,0,0]
            # res['heading'] = self.heading_1(roi_feature_masked)[:,:,0,0]

            res['depth'] = depth_net_out

            ####################################################################
            '''LOCAL DENSE NOC depths'''
            res['noc_depth_out'] = noc_depth_out
            # res['caddn_noc_depth_out'] = caddn_noc_depth_out

            res['noc_depth_offset_out'] = noc_depth_offset_out
            res['noc_merge_depth_out'] = noc_merge_depth_out

            res['noc_depth_out_uncern'] = noc_depth_out_uncern
            res['noc_depth_offset_out_uncern'] = noc_depth_offset_out_uncern
            res['noc_merge_depth_out_uncern'] = noc_merge_depth_out_uncern

            res['merge_depth'] = merge_depth
            res['merge_prob'] = merge_prob

            res['scale_depth_factor'] = scale_depth.unsqueeze(-1).unsqueeze(-1)
            # print('scale_depth_factor:', scale_depth.view(-1))
            ####################################################################


            res['offset_3d'] = self.offset_3d(roi_feature_masked)[:,:,0,0]
            res['size_3d']= size3d_offset
            res['h3d_log_variance'] = h3d_log_std

            # print('*********', [(k, res[k].shape) for k in res.keys()])
        else:
            res['depth'] = torch.zeros([1,2]).to(device_id)
            res['offset_3d'] = torch.zeros([1,2]).to(device_id)
            res['size_3d'] = torch.zeros([1,3]).to(device_id)
            res['train_tag'] = torch.zeros(1).type(torch.bool).to(device_id)
            res['heading'] = torch.zeros([1,24]).to(device_id)
            res['h3d_log_variance'] = torch.zeros([1,1]).to(device_id)


            res['noc_depth_out'] = torch.zeros([1,7,7]).to(device_id)
            res['noc_depth_offset_out'] = torch.zeros([1,7,7]).to(device_id)
            res['noc_merge_depth_out'] = torch.zeros([1,7,7]).to(device_id)

            res['noc_depth_out_uncern'] = torch.zeros([1,7,7]).to(device_id)
            res['noc_depth_offset_out_uncern'] = torch.zeros([1,7,7]).to(device_id)
            res['noc_merge_depth_out_uncern'] = torch.zeros([1,7,7]).to(device_id)

            res['merge_depth'] = torch.zeros([1]).to(device_id)
            res['merge_prob'] = torch.zeros([1]).to(device_id)

            res['scale_depth_factor'] = torch.zeros([1,1,1]).to(device_id)



        return res


    def get_roi_feat(self,feat,inds,mask,ret,calibs,coord_ranges,cls_ids):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        coord_map = torch.cat([torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT,1]).unsqueeze(0),\
                        torch.arange(HEIGHT).unsqueeze(-1).repeat([1,WIDE]).unsqueeze(0)],0).unsqueeze(0).repeat([BATCH_SIZE,1,1,1]).type(torch.float).to(device_id)
        box2d_centre = coord_map + ret['offset_2d']
        box2d_maps = torch.cat([box2d_centre-ret['size_2d']/2,box2d_centre+ret['size_2d']/2],1)
        box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,1,HEIGHT,WIDE]).type(torch.float).to(device_id),box2d_maps],1)
        #box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids)
        return res

    def project2rect(self,calib,point_img):
        c_u = calib[:,0,2]
        c_v = calib[:,1,2]
        f_u = calib[:,0,0]
        f_v = calib[:,1,1]
        b_x = calib[:,0,3]/(-f_u) # relative
        b_y = calib[:,1,3]/(-f_v)
        x = (point_img[:,0]-c_u)*point_img[:,2]/f_u + b_x
        y = (point_img[:,1]-c_v)*point_img[:,2]/f_v + b_y
        z = point_img[:,2]
        centre_by_obj = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1),z.unsqueeze(-1)],-1)
        return centre_by_obj

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    import torch
    net = CenterNet3D()
    print(net)
    input = torch.randn(4, 3, 384, 1280)
    print(input.shape, input.dtype)
    output = net(input)
