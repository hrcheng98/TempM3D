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

_delete = True


################################################################
'''for an'''

import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

class HSigmoidv2(nn.Module):
    """ (add ref)
    """
    def __init__(self, inplace=True):
        super(HSigmoidv2, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., inplace=self.inplace) / 6.
        return out

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight,
                                     a=a,
                                     mode=mode,
                                     nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight,
                                    a=a,
                                    mode=mode,
                                    nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class AttnWeights(nn.Module):
    """ Attention weights for the mixture of affine transformations
        https://arxiv.org/abs/1908.01259
    """
    def __init__(self,
                 attn_mode,
                 num_features,
                 num_affine_trans,
                 num_groups=1,
                 use_rsd=True,
                 use_maxpool=False,
                 use_bn=True,
                 eps=1e-3):
        super(AttnWeights, self).__init__()

        if use_rsd:
            use_maxpool = False

        self.num_affine_trans = num_affine_trans
        self.use_rsd = use_rsd
        self.use_maxpool = use_maxpool
        self.eps = eps
        if not self.use_rsd:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        layers = []
        if attn_mode == 0:
            layers = [
                nn.Conv2d(num_features, num_affine_trans, 1, bias=not use_bn),
                nn.BatchNorm2d(num_affine_trans) if use_bn else nn.Identity(),
                HSigmoidv2()
            ]
        elif attn_mode == 1:
            if num_groups > 0:
                assert num_groups <= num_affine_trans
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.GroupNorm(num_channels=num_affine_trans,
                                 num_groups=num_groups),
                    HSigmoidv2()
                ]
            else:
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.BatchNorm2d(num_affine_trans)
                    if use_bn else nn.Identity(),
                    HSigmoidv2()
                ]
        else:
            raise NotImplementedError("Unknow attention weight type")

        self.attention = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)


    def forward(self, x):
        b, c, h, w = x.size()
        if self.use_rsd:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
            y = mean * (var + self.eps).rsqrt()

            # var = torch.var(x, dim=(2, 3), keepdim=True)
            # y *= (var + self.eps).rsqrt()
        else:
            y = self.avgpool(x)
            if self.use_maxpool:
                y += F.max_pool2d(x, (h, w), stride=(h, w)).view(b, c, 1, 1)
        return self.attention(y).view(b, self.num_affine_trans)



class AttnBatchNorm2d(nn.BatchNorm2d):
    """ Attentive Normalization with BatchNorm2d backbone
        https://arxiv.org/abs/1908.01259
    """

    _abbr_ = "AttnBN2d"

    def __init__(self,
                 num_features,
                 num_affine_trans=10,
                 attn_mode=0,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 use_rsd=True,
                 use_maxpool=False,
                 use_bn=True,
                 eps_var=1e-3):
        super(AttnBatchNorm2d,
              self).__init__(num_features,
                             affine=False,
                             eps=eps,
                             momentum=momentum,
                             track_running_stats=track_running_stats)

        self.num_affine_trans = num_affine_trans
        self.attn_mode = attn_mode
        self.use_rsd = use_rsd
        self.eps_var = eps_var

        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))

        self.attn_weights = AttnWeights(attn_mode,
                                        num_features,
                                        num_affine_trans,
                                        use_rsd=use_rsd,
                                        use_maxpool=use_maxpool,
                                        use_bn=use_bn,
                                        eps=eps_var)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        output = super(AttnBatchNorm2d, self).forward(x)
        size = output.size()
        y = self.attn_weights(x)  # bxk

        weight = y @ self.weight_  # bxc
        bias = y @ self.bias_  # bxc
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

################################################################



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


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        try:
            if m.bias:
                nn.init.constant_(m.bias, 0.0)
        except:
            nn.init.constant_(m.bias, 0.0)


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 1, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 1, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 1, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 1, 1)

        self.pose_conv = nn.Conv2d(256, 2, 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.net[i](out)
            out = self.relu(out)

        out = self.pose_conv(out)

        return out

# # def warpFeat(feat, u_offset, v_offset):
# def warpFeatFlowBK(feat, u_offset, v_offset):
#     b, c, h, w = feat.shape
#     coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
#     coords_b, coords_v, coords_u = coords_b.to(feat.device), coords_v.to(feat.device), coords_u.to(feat.device)
#
#     u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True)
#     v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True)
#
#     # print(coords_u.shape, u_offset.squeeze().shape)
#     coords_u_trans = torch.clamp(coords_u - u_offset.squeeze(), 1, w-2)
#     coords_v_trans = torch.clamp(coords_v - v_offset.squeeze(), 1, h-2)
#
#     new_u_low = torch.floor(coords_u_trans)
#     new_u_high = torch.floor(coords_u_trans) + 1
#     new_v_low = torch.floor(coords_v_trans)
#     new_v_high = torch.floor(coords_v_trans) + 1
#
#     weight_u_low = 1 - (coords_u_trans - new_u_low)
#     weight_u_high = (coords_u_trans - new_u_low)
#     weight_v_low = 1 - (coords_v_trans - new_v_low)
#     weight_v_high = (coords_v_trans - new_v_low)
#
#     weight_v_low_u_low = weight_v_low * weight_u_low
#     weight_v_low_u_high = weight_v_low * weight_u_high
#     weight_v_high_u_low = weight_v_high * weight_u_low
#     weight_v_high_u_high = weight_v_high * weight_u_high
#
#     feat_warp = feat[coords_b.view(-1).type(torch.long), :, new_v_low.view(-1).type(torch.long), new_u_low.view(-1).type(torch.long)].view(b, h*w, c).permute(0, 2, 1) * weight_v_low_u_low.view(b, 1, h*w) + \
#                 feat[coords_b.view(-1).type(torch.long), :, new_v_low.view(-1).type(torch.long), new_u_high.view(-1).type(torch.long)].view(b, h*w, c).permute(0, 2, 1) * weight_v_low_u_high.view(b, 1, h*w) + \
#                 feat[coords_b.view(-1).type(torch.long), :, new_v_high.view(-1).type(torch.long), new_u_low.view(-1).type(torch.long)].view(b, h*w, c).permute(0, 2, 1) * weight_v_high_u_low.view(b, 1, h*w) + \
#                 feat[coords_b.view(-1).type(torch.long), :, new_v_high.view(-1).type(torch.long), new_u_high.view(-1).type(torch.long)].view(b, h*w, c).permute(0, 2, 1) * weight_v_high_u_high.view(b, 1, h*w)
#
#     feat_warp = feat_warp.view(b, c, h, w)
#
#     return feat_warp


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



def warpFeatFlow(feat, u_offset, v_offset):
    b, c, h, w = feat.shape
    coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, coords_v, coords_u = coords_b.to(feat.device), coords_v.to(feat.device), coords_u.to(feat.device)

    scale_factor_u = h / u_offset.shape[-2]
    scale_factor_v = w / u_offset.shape[-1]
    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = torch.clamp(coords_u + u_offset.squeeze(), 0, w-1).type(torch.long)
    coords_v_trans = torch.clamp(coords_v + v_offset.squeeze(), 0, h-1).type(torch.long)

    feat_warp = torch.zeros_like(feat)
    # feat_warp[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)] = \
    #     feat[coords_b.view(-1).type(torch.long), :, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)]
    feat_warp[coords_b.view(-1).type(torch.long), :, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = \
        feat[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]

    '''cat flow'''
    flow_warp = torch.zeros_like(feat[:, :2, ...])
    flow_warp[coords_b.view(-1).type(torch.long), 0, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = -u_offset[coords_b.view(-1).type(torch.long), 0, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]
    flow_warp[coords_b.view(-1).type(torch.long), 1, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = -v_offset[coords_b.view(-1).type(torch.long), 0, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]


    # '''norm flow'''
    # flow_warp[:, 0, ...] /= flow_warp.shape[-1]
    # flow_warp[:, 1, ...] /= flow_warp.shape[-2]

    feat_warp = torch.cat([feat_warp, flow_warp], dim=1)
    # feat_warp = torch.cat([feat_warp, flow_warp, coords_u_trans.unsqueeze(1), coords_v_trans.unsqueeze(1),
    #                        coords_u.unsqueeze(1), coords_v.unsqueeze(1)], dim=1)

    return feat_warp



# def warpFeatFlowWarpDepth(feat, depth, t_cy, u_offset, v_offset):
def warpFeatFlowWarpDepth(feat, depth, instr, u_offset, v_offset):
    t_fx, t_fy, t_cx, t_cy = instr[:, 0], instr[:, 1], instr[:, 2], instr[:, 3]

    b, c, h, w = feat.shape
    coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, coords_v, coords_u = coords_b.to(feat.device), coords_v.to(feat.device), coords_u.to(feat.device)

    scale_factor_u = h / u_offset.shape[-2]
    scale_factor_v = w / u_offset.shape[-1]
    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = torch.clamp(coords_u + u_offset.squeeze(), 0, w-1).type(torch.long)
    coords_v_trans = torch.clamp(coords_v + v_offset.squeeze(), 0, h-1).type(torch.long)

    feat_warp = torch.zeros_like(feat)
    # feat_warp[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)] = \
    #     feat[coords_b.view(-1).type(torch.long), :, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)]
    feat_warp[coords_b.view(-1).type(torch.long), :, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = \
        feat[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]

    '''cat flow'''
    flow_warp = torch.zeros_like(feat[:, :2, ...])
    flow_warp[coords_b.view(-1).type(torch.long), 0, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = -u_offset[coords_b.view(-1).type(torch.long), 0, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]
    flow_warp[coords_b.view(-1).type(torch.long), 1, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = -v_offset[coords_b.view(-1).type(torch.long), 0, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]


    # '''norm flow'''
    # flow_warp[:, 0, ...] /= flow_warp.shape[-1]
    # flow_warp[:, 1, ...] /= flow_warp.shape[-2]


    '''cat depth'''
    wrap_depth = torch.zeros_like(feat[:, :1, ...])
    # wrap_depth[coords_b.view(-1).type(torch.long), 0, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = \
    #     depth[coords_b.view(-1).type(torch.long), 0, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)] * \
    #         ((coords_v.view(-1).type(torch.long) - t_cy) / (coords_v_trans.view(-1).type(torch.long) - t_cy))

    trans_factor_1 = coords_v.unsqueeze(1)/scale_factor_v - t_cy.view(-1, 1, 1, 1)
    trans_factor_2 = (coords_v.unsqueeze(1) + v_offset)/scale_factor_v - t_cy.view(-1, 1, 1, 1)
    trans_factor_ind = trans_factor_2 < (0.5 / scale_factor_v)
    trans_factor_2[trans_factor_ind] = trans_factor_1[trans_factor_ind]
    # scale_trans_factor = torch.clamp(trans_factor_1 / trans_factor_2, 0.1, 10)
    scale_trans_factor = torch.clamp(trans_factor_1 / trans_factor_2, 0.1, 3)
    # depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=True)
    depth = F.interpolate(depth, (h, w), mode='nearest')
    trans_depth = depth * scale_trans_factor
    wrap_depth[coords_b.view(-1).type(torch.long), 0, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = trans_depth[coords_b.view(-1).type(torch.long), 0, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]


    wrold_x = (coords_u.unsqueeze(1)/scale_factor_u - t_cx.view(-1, 1, 1, 1)) * wrap_depth / t_fx.view(-1, 1, 1, 1)
    wrold_y = (coords_v.unsqueeze(1)/scale_factor_v - t_cy.view(-1, 1, 1, 1)) * wrap_depth / t_fy.view(-1, 1, 1, 1)


    # # feat_warp = torch.cat([feat_warp, flow_warp, wrap_depth], dim=1)
    feat_warp = torch.cat([feat_warp, wrap_depth], dim=1)
    # feat_warp = torch.cat([feat_warp, wrold_x, wrold_y, wrap_depth], dim=1)

    return feat_warp



def warpFeatFlow_v2(feat, u_offset, v_offset):
    b, c, h, w = feat.shape
    coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, coords_v, coords_u = coords_b.to(feat.device), coords_v.to(feat.device), coords_u.to(feat.device)

    scale_factor_u = h / u_offset.shape[-2]
    scale_factor_v = w / u_offset.shape[-1]
    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = torch.clamp(coords_u + u_offset.squeeze(), 1, w-2)
    coords_v_trans = torch.clamp(coords_v + v_offset.squeeze(), 1, h-2)


    new_u_low = (torch.floor(coords_u_trans)).type(torch.long)
    new_u_high = (torch.floor(coords_u_trans) + 1).type(torch.long)
    new_v_low = (torch.floor(coords_v_trans)).type(torch.long)
    new_v_high = (torch.floor(coords_v_trans) + 1).type(torch.long)

    weight_u_low = 1 - (coords_u_trans - new_u_low)
    weight_u_high = (coords_u_trans - new_u_low)
    weight_v_low = 1 - (coords_v_trans - new_v_low)
    weight_v_high = (coords_v_trans - new_v_low)

    weight_v_low_u_low = weight_v_low * weight_u_low
    weight_v_low_u_high = weight_v_low * weight_u_high
    weight_v_high_u_low = weight_v_high * weight_u_low
    weight_v_high_u_high = weight_v_high * weight_u_high

    feat_warp_v_low_u_low = torch.zeros_like(feat)
    feat_warp_v_low_u_high = torch.zeros_like(feat)
    feat_warp_v_high_u_low = torch.zeros_like(feat)
    feat_warp_v_high_u_high = torch.zeros_like(feat)
    feat_warp_v_low_u_low[coords_b.view(-1).type(torch.long), :, new_v_low.view(-1).type(torch.long), new_u_low.view(-1).type(torch.long)] = \
        (weight_v_low_u_low.unsqueeze(1) * feat)[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]
    feat_warp_v_low_u_high[coords_b.view(-1).type(torch.long), :, new_v_low.view(-1).type(torch.long), new_u_high.view(-1).type(torch.long)] = \
        (weight_v_low_u_high.unsqueeze(1) * feat)[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]
    feat_warp_v_high_u_low[coords_b.view(-1).type(torch.long), :, new_v_high.view(-1).type(torch.long), new_u_low.view(-1).type(torch.long)] = \
        (weight_v_high_u_low.unsqueeze(1) * feat)[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]
    feat_warp_v_high_u_high[coords_b.view(-1).type(torch.long), :, new_v_high.view(-1).type(torch.long), new_u_high.view(-1).type(torch.long)] = \
        (weight_v_high_u_high.unsqueeze(1) * feat)[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]

    feat_warp = feat_warp_v_low_u_low + feat_warp_v_low_u_high + feat_warp_v_high_u_low + feat_warp_v_high_u_high

    return feat_warp


def warpFeatFlow_v3(feat, u_offset, v_offset):
    b, c, h, w = feat.shape
    coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, coords_v, coords_u = coords_b.to(feat.device), coords_v.to(feat.device), coords_u.to(feat.device)

    scale_factor_u = w / u_offset.shape[-1]
    scale_factor_v = h / u_offset.shape[-2]
    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = (torch.clamp(coords_u + u_offset.squeeze(), 0, w-2))
    coords_v_trans = (torch.clamp(coords_v + v_offset.squeeze(), 0, h-2))

    new_u_low = (torch.floor(coords_u_trans)).type(torch.long)
    new_u_high = (torch.floor(coords_u_trans) + 1).type(torch.long)
    new_v_low = (torch.floor(coords_v_trans)).type(torch.long)
    new_v_high = (torch.floor(coords_v_trans) + 1).type(torch.long)

    weight_u_low = 1. - (coords_u_trans - new_u_low)
    weight_u_high = (coords_u_trans - new_u_low)
    weight_v_low = 1. - (coords_v_trans - new_v_low)
    weight_v_high = (coords_v_trans - new_v_low)

    '''trans coord'''
    coords_b, coords_u, coords_v = coords_b.view(-1), coords_u.view(-1), coords_v.view(-1)
    new_u_low, new_u_high, new_v_low, new_v_high = new_u_low.view(-1), new_u_high.view(-1), new_v_low.view(-1), new_v_high.view(-1)


    weight_v_low_u_low = ((weight_v_low * weight_u_low).unsqueeze(1))[coords_b, :, coords_v, coords_u]
    weight_v_low_u_high = ((weight_v_low * weight_u_high).unsqueeze(1))[coords_b, :, coords_v, coords_u]
    weight_v_high_u_low = ((weight_v_high * weight_u_low).unsqueeze(1))[coords_b, :, coords_v, coords_u]
    weight_v_high_u_high = ((weight_v_high * weight_u_high).unsqueeze(1))[coords_b, :, coords_v, coords_u]


    '''weighted feat'''
    feat_warp_v_low_u_low, feat_warp_v_low_u_low_weight = torch.zeros_like(feat), torch.zeros_like(feat[:, :1, ...])
    feat_warp_v_low_u_high, feat_warp_v_low_u_high_weight = torch.zeros_like(feat), torch.zeros_like(feat[:, :1, ...])
    feat_warp_v_high_u_low, feat_warp_v_high_u_low_weight = torch.zeros_like(feat), torch.zeros_like(feat[:, :1, ...])
    feat_warp_v_high_u_high, feat_warp_v_high_u_high_weight = torch.zeros_like(feat), torch.zeros_like(feat[:, :1, ...])

    feat_selected = feat[coords_b, :, coords_v, coords_u]
    feat_warp_v_low_u_low[coords_b, :, new_v_low, new_u_low] = weight_v_low_u_low * feat_selected
    feat_warp_v_low_u_low_weight[coords_b, :, new_v_low, new_u_low] = weight_v_low_u_low

    feat_warp_v_low_u_high[coords_b, :, new_v_low, new_u_high] = weight_v_low_u_high * feat_selected
    feat_warp_v_low_u_high_weight[coords_b, :, new_v_low, new_u_high] = weight_v_low_u_high

    feat_warp_v_high_u_low[coords_b, :, new_v_high, new_u_low] = weight_v_high_u_low * feat_selected
    feat_warp_v_high_u_low_weight[coords_b, :, new_v_high, new_u_low] = weight_v_high_u_low

    feat_warp_v_high_u_high[coords_b, :, new_v_high, new_u_high] = weight_v_high_u_high * feat_selected
    feat_warp_v_high_u_high_weight[coords_b, :, new_v_high, new_u_high] = weight_v_high_u_high

    feat_warp = feat_warp_v_low_u_low + feat_warp_v_low_u_high + feat_warp_v_high_u_low + feat_warp_v_high_u_high
    feat_warp_weight = feat_warp_v_low_u_low_weight + feat_warp_v_low_u_high_weight + feat_warp_v_high_u_low_weight + feat_warp_v_high_u_high_weight
    feat_warp_weight[feat_warp_weight == 0] = 1
    re_weight_ind = 1. / feat_warp_weight
    feat_warp = re_weight_ind * feat_warp
    feat_warp[feat_warp > torch.max(feat_selected)] = 0.
    feat_warp[feat_warp < torch.min(feat_selected)] = 0.

    return feat_warp


def warpFeatFlow_v4(feat, u_offset, v_offset):
    b, c, h, w = feat.shape
    coords_b, org_coords_v, org_coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, org_coords_v, org_coords_u = coords_b.to(feat.device), org_coords_v.to(feat.device), org_coords_u.to(feat.device)

    scale_factor_u = w / u_offset.shape[-1]
    scale_factor_v = h / u_offset.shape[-2]
    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = (torch.clamp(org_coords_u + u_offset.squeeze(), 0, w-2))
    coords_v_trans = (torch.clamp(org_coords_v + v_offset.squeeze(), 0, h-2))

    new_u_low = (torch.floor(coords_u_trans)).type(torch.long)
    new_u_high = (torch.floor(coords_u_trans) + 1).type(torch.long)
    new_v_low = (torch.floor(coords_v_trans)).type(torch.long)
    new_v_high = (torch.floor(coords_v_trans) + 1).type(torch.long)


    '''trans coord'''
    coords_b, coords_u, coords_v = coords_b.view(-1), org_coords_u.view(-1), org_coords_v.view(-1)
    new_u_low, new_u_high, new_v_low, new_v_high = new_u_low.view(-1), new_u_high.view(-1), new_v_low.view(-1), new_v_high.view(-1)

    '''weighted feat'''
    feat_warp_v_low_u_low = torch.zeros_like(feat)
    feat_warp_v_low_u_high = torch.zeros_like(feat)
    feat_warp_v_high_u_low = torch.zeros_like(feat)
    feat_warp_v_high_u_high = torch.zeros_like(feat)

    feat_selected = feat[coords_b, :, coords_v, coords_u]
    feat_warp_v_low_u_low[coords_b, :, new_v_low, new_u_low] = feat_selected
    feat_warp_v_low_u_high[coords_b, :, new_v_low, new_u_high] = feat_selected
    feat_warp_v_high_u_low[coords_b, :, new_v_high, new_u_low] = feat_selected
    feat_warp_v_high_u_high[coords_b, :, new_v_high, new_u_high] = feat_selected

    feat_warp = 0.25 * (feat_warp_v_low_u_low + feat_warp_v_low_u_high + feat_warp_v_high_u_low + feat_warp_v_high_u_high)
    re_w = 4. / (
                (feat_warp_v_low_u_low != 0).type(torch.float32) + (feat_warp_v_low_u_high != 0).type(torch.float32) + (
                    feat_warp_v_high_u_low != 0).type(torch.float32) + (feat_warp_v_high_u_high != 0).type(
            torch.float32))
    re_w[re_w == np.inf] = 0
    feat_warp = re_w * feat_warp


    '''cat flow'''
    flow_warp = torch.zeros_like(feat[:, :2, ...])
    uv_offset = torch.cat([u_offset, v_offset], dim=1)
    uv_offset_selected = uv_offset[coords_b, :, coords_v, coords_u]
    flow_warp[coords_b, :, new_v_low, new_u_low] += uv_offset_selected
    flow_warp[coords_b, :, new_v_low, new_u_high] += uv_offset_selected
    flow_warp[coords_b, :, new_v_high, new_u_low] += uv_offset_selected
    flow_warp[coords_b, :, new_v_high, new_u_high] += uv_offset_selected
    flow_warp = 0.25 * flow_warp * re_w[:, :2, ...]

    '''norm flow'''
    flow_warp[:, 0, ...] /= flow_warp.shape[-1]
    flow_warp[:, 1, ...] /= flow_warp.shape[-2]

    feat_warp = torch.cat([feat_warp, flow_warp], dim=1)

    # feat_warp = torch.cat([feat_warp, u_offset, v_offset, coords_u_trans.unsqueeze(1), coords_v_trans.unsqueeze(1),
    #                        org_coords_u.unsqueeze(1), org_coords_v.unsqueeze(1)], dim=1)

    return feat_warp


def gather_ind_from_arg(ind, shape):
    b, c, h, w = shape
    ind_b = ind // (c * h * w)
    ind_c = (ind - ind_b * c * h * w) // (h * w)
    ind_h = (ind - ind_b * c * h * w - ind_c * h * w) // w
    ind_w = ind - ind_b * c * h * w - ind_c * h * w - ind_h * w
    return (ind_b, ind_c, ind_h, ind_w)


def deprocess(t):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = np.transpose(t, [1, 2, 0])
    t = ((t * std) + mean) * 255.
    return t.astype(np.uint8)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def TransFlow(u_offset, v_offset):
    b, c, h, w = u_offset.shape
    coords_b, coords_v, coords_u = torch.meshgrid(torch.arange(b), torch.arange(h), torch.arange(w))
    coords_b, coords_v, coords_u = coords_b.to(u_offset.device), coords_v.to(u_offset.device), coords_u.to(u_offset.device)

    scale_factor_u = h / u_offset.shape[-2]
    scale_factor_v = w / u_offset.shape[-1]
    u_offset = F.interpolate(u_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_u
    v_offset = F.interpolate(v_offset, (h, w), mode='bilinear', align_corners=True) * scale_factor_v

    # print(coords_u.shape, u_offset.squeeze().shape)
    coords_u_trans = torch.clamp(coords_u + u_offset.squeeze(), 0, w-1).type(torch.long)
    coords_v_trans = torch.clamp(coords_v + v_offset.squeeze(), 0, h-1).type(torch.long)

    trans_flow_u = torch.ones_like(u_offset) * -1000
    trans_flow_u[coords_b.view(-1).type(torch.long), :, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = \
        u_offset[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]

    trans_flow_v = torch.ones_like(v_offset) * -1000
    trans_flow_v[coords_b.view(-1).type(torch.long), :, coords_v_trans.view(-1).type(torch.long), coords_u_trans.view(-1).type(torch.long)] = \
        v_offset[coords_b.view(-1).type(torch.long), :, coords_v.view(-1).type(torch.long), coords_u.view(-1).type(torch.long)]

    return torch.cat([trans_flow_u, trans_flow_v], dim=1)



class GUPNet(nn.Module):
    def __init__(self, backbone='dla34', neck='DLAUp', downsample=4, mean_size=None):
        assert downsample in [4, 8, 16, 32]
        super().__init__()


        self.backbone = globals()[backbone](pretrained=True, return_levels=True)
        self.head_conv = 256  # default setting for head conv
        self.mean_size = nn.Parameter(torch.tensor(mean_size,dtype=torch.float32),requires_grad=False)
        self.cls_num = mean_size.shape[0]

        # channels = self.backbone.channels  # channels list for feature maps generated by backbone

        # channels = [16*3, 32*3, 64*3, 128*3, 256*3, 512*3]
        # channels = [16*3+8, 32*3+8, 64*3+8, 128*3+8, 256*3+8, 512*3+8]
        # channels = [16*3+12, 32*3+12, 64*3+12, 128*3+12, 256*3+12, 512*3+12]
        # channels = [16*3+4, 32*3+4, 64*3+4, 128*3+4, 256*3+4, 512*3+4]

        # channels = [16, 32, 64*3+4, 128, 256, 512]
        # channels = [16, 32, 64*3+4, 128*3+4, 256, 512]
        # channels = [16, 32, 64*3+4, 128*3+4, 256*3+4, 512]
        # channels = [16, 32, 64*3+4, 128*3+4, 256*3+4, 512*3+4]
        channels = [16, 32, 64+2, 128, 256, 512]
        if _delete:
            channels = [16, 32, 64, 128, 256, 512]
        
        # channels = [16, 32, 64+4, 128, 256, 512]

        # channels = [16*5+8, 32*5+8, 64*5+8, 128*5+8, 256*5+8, 512*5+8]
        # channels = [16*3+3, 32*3+3, 64*3+3, 128*3+3, 256*3+3, 512*3+3]
        # channels = [16+3, 32+3, 64+3, 128+3, 256+3, 512+3]
        # channels = [16*3+9, 32*3+9, 64*3+9, 128*3+9, 256*3+9, 512*3+9]
        # channels = [16*2, 32*2, 64*2, 128*2, 256*2, 512*2]

        self.first_level = int(np.log2(downsample))
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.feat_up = globals()[neck](channels[self.first_level:], scales_list=scales)

        # channels[self.first_level] = 3 * channels[self.first_level]

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
        '''LOCAL DENSE NOC depths'''
        # self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              # nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        #                              nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        #                              # nn.Conv2d(self.head_conv, 81+1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_offset = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              # nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        #                              nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))


        # self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              # nn.BatchNorm2d(self.head_conv),
        #                                 nn.InstanceNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_offset = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              # nn.BatchNorm2d(self.head_conv),
        #                                       nn.InstanceNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              # nn.BatchNorm2d(self.head_conv),
        #                                       nn.InstanceNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_offset_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              # nn.BatchNorm2d(self.head_conv),
        #                                              nn.InstanceNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))



        # self.NOC_depth = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              # AttnBatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                nn.Conv2d(self.head_conv*2, self.head_conv,
        #                                          kernel_size=3, padding=1, bias=True),
        #                                nn.BatchNorm2d(self.head_conv),
        #                                # AttnBatchNorm2d(self.head_conv),
        #                                nn.LeakyReLU(inplace=True),
        #                                nn.Conv2d(self.head_conv, self.head_conv,
        #                                          kernel_size=3, padding=1, bias=True),
        #                                nn.BatchNorm2d(self.head_conv),
        #                                # AttnBatchNorm2d(self.head_conv),
        #                                nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        #                              # nn.Conv2d(self.head_conv, 96, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_offset = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              # AttnBatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                   nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                             kernel_size=3, padding=1, bias=True),
        #                                   nn.BatchNorm2d(self.head_conv),
        #                                   # AttnBatchNorm2d(self.head_conv),
        #                                   nn.LeakyReLU(inplace=True),
        #                                       nn.Conv2d(self.head_conv, self.head_conv,
        #                                                 kernel_size=3, padding=1, bias=True),
        #                                       nn.BatchNorm2d(self.head_conv),
        #                                       # AttnBatchNorm2d(self.head_conv),
        #                                       nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              # AttnBatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                       nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                                 kernel_size=3, padding=1, bias=True),
        #                                       nn.BatchNorm2d(self.head_conv),
        #                                       # AttnBatchNorm2d(self.head_conv),
        #                                       nn.LeakyReLU(inplace=True),
        #                                       nn.Conv2d(self.head_conv, self.head_conv,
        #                                                 kernel_size=3, padding=1, bias=True),
        #                                       nn.BatchNorm2d(self.head_conv),
        #                                       # AttnBatchNorm2d(self.head_conv),
        #                                       nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.NOC_depth_offset_uncern = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              # AttnBatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                              nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                                        kernel_size=3, padding=1, bias=True),
        #                                              nn.BatchNorm2d(self.head_conv),
        #                                              # AttnBatchNorm2d(self.head_conv),
        #                                              nn.LeakyReLU(inplace=True),
        #                                              nn.Conv2d(self.head_conv, self.head_conv,
        #                                                        kernel_size=3, padding=1, bias=True),
        #                                              nn.BatchNorm2d(self.head_conv),
        #                                              # AttnBatchNorm2d(self.head_conv),
        #                                              nn.LeakyReLU(inplace=True),
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
                                     nn.BatchNorm2d(self.head_conv),
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
                                     nn.BatchNorm2d(self.head_conv),
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
                                     nn.BatchNorm2d(self.head_conv),
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
                                     nn.BatchNorm2d(self.head_conv),
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
        self.heading = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))
        # self.heading_1 = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              # AttnBatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                       nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                                 kernel_size=3, padding=1, bias=True),
        #                                       nn.BatchNorm2d(self.head_conv),
        #                                       # AttnBatchNorm2d(self.head_conv),
        #                                       nn.LeakyReLU(inplace=True),
        #                                       nn.Conv2d(self.head_conv, self.head_conv,
        #                                                 kernel_size=3, padding=1, bias=True),
        #                                       nn.BatchNorm2d(self.head_conv),
        #                                       # AttnBatchNorm2d(self.head_conv),
        #                                       nn.LeakyReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))


        self.corners_offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.BatchNorm2d(self.head_conv),
                                     nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(self.head_conv, 8*2, kernel_size=1, stride=1, padding=0, bias=True))
        # self.corners_offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                                        nn.BatchNorm2d(self.head_conv * 2),
        #                                        nn.LeakyReLU(inplace=True),
        #                                        nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                                  kernel_size=3, padding=1, bias=True),
        #                                        nn.BatchNorm2d(self.head_conv),
        #                                        nn.LeakyReLU(inplace=True),
        #                                        nn.AdaptiveAvgPool2d(1),
        #                              nn.Conv2d(self.head_conv, 8*2, kernel_size=1, stride=1, padding=0, bias=True))

        # self.offset_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                          kernel_size=3, padding=1, bias=True),
        #                                nn.BatchNorm2d(self.head_conv),
        #                                nn.LeakyReLU(inplace=True),
        #                                nn.AdaptiveAvgPool2d(1),
        #                              nn.Conv2d(self.head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        # self.size_3d = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                          kernel_size=3, padding=1, bias=True),
        #                                nn.BatchNorm2d(self.head_conv),
        #                                nn.LeakyReLU(inplace=True),
        #                              nn.AdaptiveAvgPool2d(1),
        #                              nn.Conv2d(self.head_conv, 4, kernel_size=1, stride=1, padding=0, bias=True))
        # self.heading = nn.Sequential(nn.Conv2d(channels[self.first_level]+2+self.cls_num, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                                nn.Conv2d(self.head_conv * 2, self.head_conv,
        #                                          kernel_size=3, padding=1, bias=True),
        #                                nn.BatchNorm2d(self.head_conv),
        #                                nn.LeakyReLU(inplace=True),
        #                              nn.AdaptiveAvgPool2d(1),
        #                              nn.Conv2d(self.head_conv, 24, kernel_size=1, stride=1, padding=0, bias=True))


        # self.coord_offset = PoseCNN(2)


        # self.att_flow = nn.Sequential(nn.Conv2d(6, self.head_conv, kernel_size=3, stride=2, padding=1, bias=True),
        #                              nn.BatchNorm2d(self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(self.head_conv, 2*self.head_conv, kernel_size=3, stride=2, padding=1, bias=True),
        #                              nn.BatchNorm2d(2*self.head_conv),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(2*self.head_conv, 3, kernel_size=1, stride=1, padding=0, bias=True))
        # self.att_flow.apply(weights_init_xavier)

        # self.att_flow = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
        #                              nn.BatchNorm2d(32),
        #                              nn.ReLU(inplace=True),
        #                              nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True))
        # self.att_flow.apply(weights_init_xavier)

        # self.att = nn.ModuleList([SELayer(64 + 2),
        #                           SELayer(128 + 2),
        #                           SELayer(256 + 2),
        #                           SELayer(512 + 2),
        #                           ])
        # self.att.apply(weights_init_xavier)


        # self.fuse_3d = nn.Sequential(nn.Conv3d(channels[self.first_level]+3, self.head_conv*2, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm3d(self.head_conv*2),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv3d(self.head_conv * 2, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm3d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              nn.Conv3d(self.head_conv, self.head_conv, kernel_size=3, padding=1, bias=True),
        #                              nn.BatchNorm3d(self.head_conv),
        #                              nn.LeakyReLU(inplace=True),
        #                              # nn.Conv3d(self.head_conv, channels[self.first_level], kernel_size=1, padding=0, bias=True),
        #                              nn.Conv3d(self.head_conv, channels[self.first_level], kernel_size=3, padding=1, bias=True),
        #                              # nn.AvgPool3d((3, 1, 1)),
        #                              nn.MaxPool3d((3, 1, 1)),
        #                              )

        # '''
        # pixel wise depth for vedio
        # [torch.Size([2, 64, 96, 320]),
        #  torch.Size([2, 128, 48, 160]),
        #  torch.Size([2, 256, 24, 80]),
        #  torch.Size([2, 512, 12, 40])]
        #
        # '''

        # self.pixel_depth_0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(256),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(128),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(128),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))
        #
        # self.pixel_depth_backbone = nn.ModuleList([self.pixel_depth_3, self.pixel_depth_2, self.pixel_depth_1, self.pixel_depth_0])


        # pixel_depth_channels = [64, 128, 256, 512]
        # self.pixel_depth_new = globals()[neck](pixel_depth_channels, scales_list=scales)
        # self.pixel_depth_head = nn.Sequential(nn.Conv2d(pixel_depth_channels[self.first_level], 32, kernel_size=3, padding=1, bias=True),
        #                                        nn.BatchNorm2d(32),
        #                                        nn.LeakyReLU(inplace=True),
        #                                        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
        #                                        nn.BatchNorm2d(32),
        #                                        nn.LeakyReLU(inplace=True),
        #                                        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_head.apply(weights_init_xavier)
        #
        #
        # self.pixel_depth_head_fuse = nn.Sequential(nn.Conv2d(64*3+9, 32, kernel_size=3, padding=1, bias=True),
        #                                        nn.BatchNorm2d(32),
        #                                        nn.LeakyReLU(inplace=True),
        #                                        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
        #                                        nn.BatchNorm2d(32),
        #                                        nn.LeakyReLU(inplace=True),
        #                                        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_head_fuse.apply(weights_init_xavier)


        # self.pixel_depth_0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(256),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(256),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
        #                                    nn.BatchNorm2d(64),
        #                                    nn.LeakyReLU(inplace=True),
        #                                    nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True))
        # self.pixel_depth_0.apply(weights_init_xavier)



        # init layers
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.fill_fc_weights(self.offset_2d)
        self.fill_fc_weights(self.size_2d)

        self.depth.apply(weights_init_xavier)
        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)

        self.NOC_depth.apply(weights_init_xavier)
        self.NOC_depth_offset.apply(weights_init_xavier)
        self.NOC_depth_uncern.apply(weights_init_xavier)
        self.NOC_depth_offset_uncern.apply(weights_init_xavier)


    def forward(self, input, coord_ranges,calibs, targets=None, K=50, mode='train'):
        # input = input['img']
        #
        # device_id = input.device
        # BATCH_SIZE = input.size(0)
        #
        #
        # feat = self.backbone(input)
        # feat = self.feat_up(feat[self.first_level:])
        # ret = {}


        # input = input['img']
        # device_id = input.device
        # BATCH_SIZE = input.size(0)
        # feat = self.backbone(input)
        # feat = self.feat_up(feat[self.first_level:])

        # img = input['img']
        # pre_1 = input['pre_1']
        # pre_2 = input['pre_2']
        # device_id = img.device
        #
        #
        # feat_pre2 = self.backbone(pre_2)
        # feat_pre2 = [i.detach() for i in feat_pre2]
        # feat_pre1 = self.backbon
        # feat_pre1 = [i.detach() for i in feat_pre1]
        # feat_img = self.backbone(img)
        # feat_fuse = [torch.cat([i*0.2, j*0.5, k], dim=1) for i, j, k in zip(feat_pre2, feat_pre1, feat_img)]
        # feat = self.feat_up(feat_fuse[self.first_level:])

        # torch.cuda.synchronize()
        # a = datetime.now()


        # # print(pre_2.device, self.coord_offset.net[0].weight.device)
        # uv_offset_2 = self.coord_offset(torch.cat([pre_2, img], dim=1))
        # uv_offset_1 = self.coord_offset(torch.cat([pre_1, img], dim=1))
        # # b = datetime.now()
        # # print(b - a, a, b)

        # feat_pre2 = self.backbone(pre_2)
        # # feat_pre2 = [warpFeat(i.detach(), uv_offset_2[:, :1, ...], uv_offset_2[:, 1:2, ...]) for i in feat_pre2]
        # # feat_pre2 = [warpFeatFlow(i.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)) for i in feat_pre2]
        # feat_pre2 = [warpFeatFlow(i, pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)) for i in feat_pre2]
        # feat_pre1 = self.backbone(pre_1)
        # # feat_pre1 = [warpFeat(i.detach(), uv_offset_1[:, :1, ...], uv_offset_1[:, 1:2, ...]) for i in feat_pre1]
        # # feat_pre1 = [warpFeatFlow(i.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1)) for i in feat_pre1]
        # feat_pre1 = [warpFeatFlow(i, pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1)) for i in feat_pre1]
        # feat_img = self.backbone(img)
        # feat_fuse = [torch.cat([i, j, k], dim=1) for i, j, k in zip(feat_pre2, feat_pre1, feat_img)]
        # # feat_fuse = [1/3*(i+j+k) for i, j, k in zip(feat_pre2, feat_pre1, feat_img)]
        # feat = self.feat_up(feat_fuse[self.first_level:])
        '''3 frame'''
        # import cv2 as cv
        # tmp2 = warpFeatFlow(pre_2, pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # tmp3 = warpFeatFlow_v4(pre_2, pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # cv.imwrite('tmp.png', deprocess(img.cpu().numpy()[0]))
        # cv.imwrite('tmp1.png', deprocess(tmp2.cpu().numpy()[0]))
        # cv.imwrite('tmp2.png', deprocess(tmp3.cpu().numpy()[0]))
        # tmp3 = warpFeatFlow_v4(pre_2, pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        ###########################################################

        # feat_pre2 = self.backbone(pre_2)[2:]
        # # feat_pre2 = [self.att[ind](warpFeatFlow(i.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))) for ind, i in enumerate(feat_pre2)]
        # feat_pre2 = [warpFeatFlow(i.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)) for ind, i in enumerate(feat_pre2)]
        # feat_pre1 = self.backbone(pre_1)[2:]
        # # feat_pre1 = [self.att[ind](warpFeatFlow(i.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1))) for ind, i in enumerate(feat_pre1)]
        # feat_pre1 = [warpFeatFlow(i.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1)) for ind, i in enumerate(feat_pre1)]
        # feat_img = self.backbone(img)[2:]
        # feat_fuse = [torch.cat([i, j, k], dim=1) for i, j, k in zip(feat_pre2, feat_pre1, feat_img)]
        # feat = self.feat_up(feat_fuse)

        # feat_pre2 = [i.detach() for i in self.backbone(pre_2)[2:]]
        # feat_pre1 = [i.detach() for i in self.backbone(pre_1)[2:]]
        # feat_img = self.backbone(img)[2:]
        # # feat_pre2_depth = [self.pixel_depth_backbone[i](v).detach() for i, v in enumerate(feat_pre2)]
        # # feat_pre1_depth = [self.pixel_depth_backbone[i](v).detach() for i, v in enumerate(feat_pre1)]
        # # feat_img_depth = [self.pixel_depth_backbone[i](v) for i, v in enumerate(feat_img)]
        # feat_pre2_depth = self.pixel_depth_new(feat_pre2).detach()
        # feat_pre2_depth = self.pixel_depth_head(feat_pre2_depth).detach()
        # feat_pre1_depth = self.pixel_depth_new(feat_pre1).detach()
        # feat_pre1_depth = self.pixel_depth_head(feat_pre1_depth).detach()
        # feat_img_depth = self.pixel_depth_new(feat_img)
        # feat_img_depth = self.pixel_depth_head(feat_img_depth)
        # feat_pre2_depth = [feat_pre2_depth, feat_pre2_depth, feat_pre2_depth, feat_pre2_depth]
        # feat_pre1_depth = [feat_pre1_depth, feat_pre1_depth, feat_pre1_depth, feat_pre1_depth]
        # feat_img_depth = [feat_img_depth, feat_img_depth, feat_img_depth, feat_img_depth]
        #
        # ret['pixel_depth_backbone'] = feat_img_depth[::-1]
        #
        #
        # # feat_add_depth_pre2 = [warpFeatFlowWarpDepth(i, (-j).exp(), input['t_cy'], pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)) for i, j in zip(feat_pre2, feat_pre2_depth)]
        # # feat_add_depth_pre1 = [warpFeatFlowWarpDepth(i, (-j).exp(), input['t_cy'], pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1)) for i, j in zip(feat_pre1, feat_pre1_depth)]
        # feat_add_depth_pre2 = [warpFeatFlowWarpDepth(i, (-j).exp(), input['instr'], pre_2_flow[..., 0].unsqueeze(1),
        #                                              pre_2_flow[..., 1].unsqueeze(1)) for i, j in
        #                        zip(feat_pre2, feat_pre2_depth)]
        # feat_add_depth_pre1 = [warpFeatFlowWarpDepth(i, (-j).exp(), input['instr'], pre_1_flow[..., 0].unsqueeze(1),
        #                                              pre_1_flow[..., 1].unsqueeze(1)) for i, j in
        #                        zip(feat_pre1, feat_pre1_depth)]
        #
        # # feat_add_depth_img = [torch.cat([i, j], dim=1) for i, j in zip(feat_img, feat_img_depth)]
        # feat_add_depth_img = [warpFeatFlowWarpDepth(i, (-j).exp(), input['instr'], torch.zeros_like(j), torch.zeros_like(j)) for i, j in
        #                        zip(feat_img, feat_img_depth)]
        # # feat_add_depth_img = [warpFeatFlowWarpDepth(i, input['lidar_depth'], input['instr'], torch.zeros_like(j), torch.zeros_like(j)) for i, j in
        # #                        zip(feat_img, feat_img_depth)]
        #
        #
        # feat_fuse = [torch.cat([i[:, -1:, ...], j[:, -1:, ...], k], dim=1) for i, j, k in zip(feat_add_depth_pre2, feat_add_depth_pre1, feat_add_depth_img)]
        # feat = self.feat_up(feat_fuse)
        #
        # # depth_fused = self.pixel_depth_head_fuse(feat)
        # # ret['pixel_depth_backbone'] = [feat_img_depth[0], depth_fused]



        # feat_pre2 = [i.detach() for i in self.backbone(pre_2)[2:]]
        # feat_pre2 = (self.feat_up(feat_pre2)).detach()
        # feat_pre1 = [i.detach() for i in self.backbone(pre_1)[2:]]
        # feat_pre1 = (self.feat_up(feat_pre1)).detach()
        # feat_img = self.backbone(img)[2:]
        # feat_img = self.feat_up(feat_img)
        # # feat = torch.cat([feat_pre2, feat_pre1, feat_img], dim=1)
        # feat_agg = [feat_pre2, feat_pre1, feat_img]
        # feat = feat_img


        img = input['img']
        pre_1 = input['pre_1']
        pre_2 = input['pre_2']
        pre_1_flow = input['pre_1_flow']
        pre_2_flow = input['pre_2_flow']
        device_id = img.device

        ret = {}


        pre_1_flow_bk = input['pre_1_flow_bk']
        pre_2_flow_bk = input['pre_2_flow_bk']
        recover_1 = warpFeatFlowBK(pre_1_flow, torch.ones_like(pre_1_flow[:, :1, ...]), pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...])
        recover_2 = warpFeatFlowBK(pre_2_flow, torch.ones_like(pre_2_flow[:, :1, ...]), pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...])
        recover_1 = -recover_1[:, :2, ...]
        recover_2 = -recover_2[:, :2, ...]
        pre1_mask = (((recover_1[:, 0:1, ...] - pre_1_flow_bk[:, 0:1, ...])**2 +
                      (recover_1[:, 1:2, ...] - pre_1_flow_bk[:, 1:2, ...])**2) < 9).type(torch.float32)
        pre2_mask = (((recover_2[:, 0:1, ...] - pre_2_flow_bk[:, 0:1, ...])**2 +
                      (recover_2[:, 1:2, ...] - pre_2_flow_bk[:, 1:2, ...])**2) < 9).type(torch.float32)


        feat = self.backbone(img)[2:]

        # self.backbone.eval()
        # with torch.no_grad():
        #     feat_pre2_base = self.backbone(pre_2)[2:]
        #     # feat_pre2_base = self.backbone(pre_2)[2:4]
        #     # feat_pre2_base = self.backbone(pre_2)[2:5]
        #     # feat_pre2_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_2))))]
        # feat_pre2_bk = [warpFeatFlowBK(i.detach(), pre2_mask, pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]
        # # feat_pre2_f = [warpFeatFlow(i.detach(), pre_2_flow[:, :1, ...], pre_2_flow[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]
        #
        # with torch.no_grad():
        #     feat_pre1_base = self.backbone(pre_1)[2:]
        #     # feat_pre1_base = self.backbone(pre_1)[2:4]
        #     # feat_pre1_base = self.backbone(pre_1)[2:5]
        #     # feat_pre1_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_1))))]
        # self.backbone.train()
        # feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
        # # feat_pre1_f = [warpFeatFlow(i.detach(), pre_1_flow[:, :1, ...], pre_1_flow[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
        #
        #
        # feat = [torch.cat([i, j, k], dim=1) for i, j, k in zip(feat_pre2_bk, feat_pre1_bk, feat)]
        # # feat[0] = torch.cat([feat_pre2_bk[0], feat_pre1_bk[0], feat[0]], dim=1)
        # # feat[1] = torch.cat([feat_pre2_bk[1], feat_pre1_bk[1], feat[1]], dim=1)
        # # feat[2] = torch.cat([feat_pre2_bk[2], feat_pre1_bk[2], feat[2]], dim=1)
        # # feat = [torch.cat([i, j, k, m, n], dim=1) for i, j, k, m, n in zip(feat_pre2_bk, feat_pre2_f, feat_pre1_bk, feat_pre1_f, feat)]
        # feat = self.feat_up(feat)

        if not _delete:
            self.backbone.eval()
            with torch.no_grad():
                feat_pre2_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_2))))]
            feat_pre2_bk = [warpFeatFlowBK(i.detach(), pre2_mask, pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]

            with torch.no_grad():
                feat_pre1_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_1))))]
            self.backbone.train()

            '''propagate'''
            # cur_factor, pre_factor = 2/3, 1/3
            # cur_factor, pre_factor = 1/2, 1/2
            # cur_factor, pre_factor = 3/4, 1/4
            cur_factor, pre_factor = 1/3, 2/3

            # feat_pre1_base = [torch.cat([2/3 * feat_pre1_base[0] + 1/3 * feat_pre2_bk[0][:, :-2, ...],
            #                             feat_pre2_bk[0][:, -2:, ...]], dim=1)]
            # feat_pre1_base = [2/3 * feat_pre1_base[0] + 1/3 * feat_pre2_bk[0][:, :-2, ...]]
            feat_pre1_base = [cur_factor * feat_pre1_base[0] + pre_factor * feat_pre2_bk[0][:, :-2, ...]]

            feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]

            # feat[0] = torch.cat([2/3 * feat[0] + 1/3 * feat_pre1_bk[0][:, :-4, ...], feat_pre1_bk[0][:, -4:, ...]], dim=1)
            # feat[0] = torch.cat([2/3 * feat[0] + 1/3 * feat_pre1_bk[0][:, :-2, ...],
            #                      1/3 * feat_pre1_bk[0][:, -2:, ...] + 1/3*1/3*feat_pre2_bk[0][:, -2:, ...]], dim=1)
            feat[0] = torch.cat([cur_factor * feat[0] + pre_factor * feat_pre1_bk[0][:, :-2, ...],
                                pre_factor * feat_pre1_bk[0][:, -2:, ...] + pre_factor*pre_factor*feat_pre2_bk[0][:, -2:, ...]], dim=1)

        feat = self.feat_up(feat)


        # self.backbone.eval()
        # with torch.no_grad():
        #     feat_pre2_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_2))))]
        # feat_pre2_bk = [warpFeatFlowBK(i.detach(), pre2_mask, pre_2_flow_bk[:, :1, ...], pre_2_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre2_base)]
        #
        # with torch.no_grad():
        #     feat_pre1_base = [self.backbone.level2(self.backbone.level1(self.backbone.level0(self.backbone.base_layer(pre_1))))]
        # self.backbone.train()
        #
        # '''propagate cat'''
        # feat_pre1_base = [torch.cat([feat_pre1_base[0], feat_pre2_bk[0]], dim=1)]
        #
        # feat_pre1_bk = [warpFeatFlowBK(i.detach(), pre1_mask, pre_1_flow_bk[:, :1, ...], pre_1_flow_bk[:, 1:2, ...]) for ind, i in enumerate(feat_pre1_base)]
        #
        # feat[0] = torch.cat([feat_pre1_bk[0], feat[0]], dim=1)
        # feat = self.feat_up(feat)




        # ''' use 3d conv for fusion'''
        # feat_pre2 = self.backbone(pre_2)[2:]
        # feat_pre2 = self.feat_up(feat_pre2)
        # feat_pre2 = warpFeatFlow(feat_pre2.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # # feat_pre2 = warpFeatFlow_v4(feat_pre2.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # # feat_pre2 = torch.cat([feat_pre2.detach(), torch.zeros_like(feat_pre2[:, :2, ...])], dim=1)
        # feat_pre1 = self.backbone(pre_1)[2:]
        # feat_pre1 = self.feat_up(feat_pre1)
        # feat_pre1 = warpFeatFlow(feat_pre1.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1))
        # # feat_pre1 = warpFeatFlow_v4(feat_pre1.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1))
        # # feat_pre1 = torch.cat([feat_pre1.detach(), torch.zeros_like(feat_pre1[:, :2, ...])], dim=1)
        # feat_img = self.backbone(img)[2:]
        # feat_img = self.feat_up(feat_img)
        # feat_pre2 = torch.cat([feat_pre2, -2 * torch.ones_like(feat_pre2[:, :1, ...])], dim=1)
        # feat_pre1 = torch.cat([feat_pre1, -1 * torch.ones_like(feat_pre2[:, :1, ...])], dim=1)
        # feat_img = torch.cat([feat_img, torch.zeros_like(feat_img[:, :2, ...]), torch.zeros_like(feat_img[:, :1, ...])], dim=1)
        # # feat_3d = torch.stack([feat_pre2, feat_pre1, feat_img], dim=2)
        # feat_3d = torch.stack([feat_pre2, feat_img, feat_pre1], dim=2)
        # feat = self.fuse_3d(feat_3d).squeeze()

        # feat_img = self.backbone(img)[2:]
        # feat = self.feat_up(feat_img)
        ############################################################
        # '''3 frame'''
        # # tmp = warpFeatFlow_v2(pre_2, pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # # tmp = warpFeatFlow(pre_1, pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1))
        # feat_pre2 = self.backbone(pre_2)[2:]
        # feat_pre2 = self.feat_up(feat_pre2)
        # # feat_pre2 = warpFeatFlow(feat_pre2.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # feat_pre2 = warpFeatFlow_v3(feat_pre2.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1))
        # feat_pre1 = self.backbone(pre_1)[2:]
        # feat_pre1 = self.feat_up(feat_pre1)
        # # feat_pre1 = warpFeatFlow(feat_pre1.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1))
        # feat_pre1 = warpFeatFlow_v3(feat_pre1.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1))
        # feat_img = self.backbone(img)[2:]
        # feat_img = self.feat_up(feat_img)
        # feat = torch.cat([feat_pre2, feat_pre1, feat_img], dim=1)

        # prob_0 = self.att_flow(torch.zeros_like(feat_img[:, :3, ...]))
        # prob_1 = self.att_flow(torch.cat([TransFlow(pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1)), torch.ones_like(pre_1_flow[..., 0].unsqueeze(1))], dim=1))
        # prob_2 = self.att_flow(torch.cat([TransFlow(pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)), 2*torch.ones_like(pre_1_flow[..., 0].unsqueeze(1))], dim=1))
        # prob = F.softmax(torch.cat([prob_0, F.interpolate(prob_1, (96, 320)), F.interpolate(prob_2, (96, 320))], dim=1), dim=1)
        # # feat = feat_img * prob[:, :1, ...].repeat(1, 384, 1, 1) + feat_pre1 * prob[:, 1:2, ...].repeat(1, 384, 1, 1) + feat_pre2 * prob[:, 2:3, ...].repeat(1, 384, 1, 1)
        # feat = feat_img * prob[:, :1, ...] + feat_pre1 * prob[:, 1:2, ...] + feat_pre2 * prob[:, 2:3, ...]

        # feat_pre2 = self.backbone(pre_2)[2:]
        # feat_pre2 = [warpFeatFlow(i, pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)) for i in feat_pre2]
        # feat_img = self.backbone(img)[2:]
        # feat_fuse = [torch.cat([i, j], dim=1) for i, j in zip(feat_pre2, feat_img)]
        # feat = self.feat_up(feat_fuse)


        # flow_cat = torch.cat([pre_2_flow, pre_1_flow, torch.zeros_like(pre_1_flow)], dim=3).permute(0, 3, 1, 2)
        # flow_prob = F.softmax(self.att_flow(flow_cat), dim=1)
        # feat_pre2 = self.backbone(pre_2)
        # feat_pre2 = [warpFeatFlow(i.detach(), pre_2_flow[..., 0].unsqueeze(1), pre_2_flow[..., 1].unsqueeze(1)) for i in feat_pre2]
        # feat_pre1 = self.backbone(pre_1)
        # feat_pre1 = [warpFeatFlow(i.detach(), pre_1_flow[..., 0].unsqueeze(1), pre_1_flow[..., 1].unsqueeze(1)) for i in feat_pre1]
        # feat_img = self.backbone(img)
        # feat_fuse = [i * F.interpolate(flow_prob[:, 0:1, ...], (i.shape[-2], i.shape[-1])) +
        #              j * F.interpolate(flow_prob[:, 1:2, ...], (i.shape[-2], i.shape[-1])) +
        #              k * F.interpolate(flow_prob[:, 2:3, ...], (i.shape[-2], i.shape[-1]))
        #              for i, j, k in zip(feat_pre2, feat_pre1, feat_img)]
        # feat = self.feat_up(feat_fuse[self.first_level:])

        # c = datetime.now()
        # print(c - b, c)


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
            # '''for 30 cards'''
            # masks = torch.ones(inds.size()).type(torch.bool).to(device_id)
        ret.update(self.get_roi_feat(feat,inds,masks,ret,calibs,coord_ranges,cls_ids))
        # ret.update(self.get_roi_feat(feat_agg,input,inds,masks,ret,calibs,coord_ranges,cls_ids))
        return ret


    def get_temploral_roi_feat_by_mask(self, feat_agg, input, box2d_maps, inds, mask, calibs, coord_ranges, cls_ids):
        BATCH_SIZE, _, HEIGHT, WIDE = feat_agg[0].size()
        device_id = feat_agg[0].device
        num_masked_bin = mask.sum()
        res = {}

        '''different size'''
        RoI_align_size = 7
        feat_pre2, feat_pre1, feat_img = feat_agg

        if num_masked_bin != 0:
            # get box2d of each roi region
            scale_box2d_masked = extract_input_from_tensor(box2d_maps, inds, mask)
            # get roi feature
            roi_feature_masked = roi_align(feat_img, scale_box2d_masked, [RoI_align_size, RoI_align_size])
            # get coord range of each roi
            coord_ranges_mask2d = coord_ranges[scale_box2d_masked[:, 0].long()]

            box2d_masked = torch.cat([scale_box2d_masked[:, 0:1],
                                      scale_box2d_masked[:, 1:2] / WIDE * (
                                                  coord_ranges_mask2d[:, 1, 0:1] - coord_ranges_mask2d[:, 0,
                                                                                   0:1]) + coord_ranges_mask2d[:, 0,
                                                                                           0:1],
                                      scale_box2d_masked[:, 2:3] / HEIGHT * (
                                                  coord_ranges_mask2d[:, 1, 1:2] - coord_ranges_mask2d[:, 0,
                                                                                   1:2]) + coord_ranges_mask2d[:, 0,
                                                                                           1:2],
                                      scale_box2d_masked[:, 3:4] / WIDE * (
                                                  coord_ranges_mask2d[:, 1, 0:1] - coord_ranges_mask2d[:, 0,
                                                                                   0:1]) + coord_ranges_mask2d[:, 0,
                                                                                           0:1],
                                      scale_box2d_masked[:, 4:5] / HEIGHT * (
                                                  coord_ranges_mask2d[:, 1, 1:2] - coord_ranges_mask2d[:, 0,
                                                                                   1:2]) + coord_ranges_mask2d[:, 0,
                                                                                           1:2]], 1)
            roi_calibs = calibs[box2d_masked[:, 0].long()]
            # project the coordinate in the normal image to the camera coord by calibs
            coords_in_camera_coord = torch.cat([self.project2rect(roi_calibs, torch.cat(
                [box2d_masked[:, 1:3], torch.ones([num_masked_bin, 1]).to(device_id)], -1))[:, :2],
                                                self.project2rect(roi_calibs, torch.cat([box2d_masked[:, 3:5],
                                                                                         torch.ones(
                                                                                             [num_masked_bin, 1]).to(
                                                                                             device_id)], -1))[:, :2]],
                                               -1)
            coords_in_camera_coord = torch.cat([box2d_masked[:, 0:1], coords_in_camera_coord], -1)
            # generate coord maps
            # coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:,1:2]+i*(coords_in_camera_coord[:,3:4]-coords_in_camera_coord[:,1:2])/6 for i in range(7)],-1).unsqueeze(1).repeat([1,7,1]).unsqueeze(1),
            #                     torch.cat([coords_in_camera_coord[:,2:3]+i*(coords_in_camera_coord[:,4:5]-coords_in_camera_coord[:,2:3])/6 for i in range(7)],-1).unsqueeze(2).repeat([1,1,7]).unsqueeze(1)],1)
            coord_maps = torch.cat([torch.cat([coords_in_camera_coord[:, 1:2] + i * (
                        coords_in_camera_coord[:, 3:4] - coords_in_camera_coord[:, 1:2]) / (RoI_align_size - 1) for i in
                                               range(RoI_align_size)], -1).unsqueeze(1).repeat([1, RoI_align_size, 1]).unsqueeze(1),
                                    torch.cat([coords_in_camera_coord[:, 2:3] + i * (
                                                coords_in_camera_coord[:, 4:5] - coords_in_camera_coord[:, 2:3]) / (
                                                           RoI_align_size - 1) for i in range(RoI_align_size)],
                                              -1).unsqueeze(2).repeat([1, 1, RoI_align_size]).unsqueeze(1)], 1)

            # concatenate coord maps with feature maps in the channel dim
            cls_hots = torch.zeros(num_masked_bin, self.cls_num).to(device_id)
            cls_hots[torch.arange(num_masked_bin).to(device_id), cls_ids[mask].long()] = 1.0

            # roi_feature_masked = torch.cat([roi_feature_masked,coord_maps,cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)
            roi_feature_masked = torch.cat([roi_feature_masked, coord_maps, cls_hots.unsqueeze(-1).unsqueeze(-1).repeat(
                [1, 1, RoI_align_size, RoI_align_size])], 1)

            '''pre frame 1'''
            c = 1


            # compute heights of projected objects
            box2d_height = torch.clamp(box2d_masked[:, 4] - box2d_masked[:, 2], min=1.0)
            scale_depth = torch.clamp((scale_box2d_masked[:, 4] - scale_box2d_masked[:, 2]) * 4, min=1.0) / \
                          torch.clamp(box2d_masked[:, 4] - box2d_masked[:, 2], min=1.0)

            # compute real 3d height
            size3d_offset = self.size_3d(roi_feature_masked)[:, :, 0, 0]
            h3d_log_std = size3d_offset[:, 3:4]
            size3d_offset = size3d_offset[:, :3]

            size_3d = (self.mean_size[cls_ids[mask].long()] + size3d_offset)
            depth_geo = size_3d[:, 0] / box2d_height.squeeze() * roi_calibs[:, 0, 0]

            depth_net_out = self.depth(roi_feature_masked)[:, :, 0, 0]
            depth_geo_log_std = (
                        h3d_log_std.squeeze() + 2 * (roi_calibs[:, 0, 0].log() - box2d_height.log())).unsqueeze(-1)
            depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:, 1:2], depth_geo_log_std], -1), -1,
                                                keepdim=True)

            depth_net_out = torch.cat(
                [(1. / (depth_net_out[:, 0:1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1), depth_net_log_std], -1)


            res['train_tag'] = torch.ones(num_masked_bin).type(torch.bool).to(device_id)
            # res['heading'] = self.heading(roi_feature_masked)[:,:,0,0]
            res['heading'] = self.heading_1(roi_feature_masked)[:, :, 0, 0]

            res['depth'] = depth_net_out


            '''corners offset '''
            res['corners_offset_3d'] = self.corners_offset_3d(roi_feature_masked)[:, :, 0, 0]
            ####################################################################

            res['offset_3d'] = self.offset_3d(roi_feature_masked)[:, :, 0, 0]
            res['size_3d'] = size3d_offset
            res['h3d_log_variance'] = h3d_log_std

            # print('*********', [(k, res[k].shape) for k in res.keys()])
        else:
            res['depth'] = torch.zeros([1, 2]).to(device_id)
            res['offset_3d'] = torch.zeros([1, 2]).to(device_id)
            res['size_3d'] = torch.zeros([1, 3]).to(device_id)
            res['train_tag'] = torch.zeros(1).type(torch.bool).to(device_id)
            res['heading'] = torch.zeros([1, 24]).to(device_id)
            res['h3d_log_variance'] = torch.zeros([1, 1]).to(device_id)

            res['corners_offset_3d'] = torch.zeros([1, 16]).to(device_id)

        return res

    def get_roi_feat_by_mask(self,feat,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids):
        BATCH_SIZE,_,HEIGHT,WIDE = feat.size()
        device_id = feat.device
        num_masked_bin = mask.sum()
        res = {}

        '''different size'''
        RoI_align_size = 7
        # RoI_align_size = 9
        # RoI_align_size = 13
        # RoI_align_size = 19
        # RoI_align_size = 5
        # RoI_align_size = 3
        # RoI_align_size = 2

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
            # noc_depth_out = NOC_depth[:, 0, :, :]
            # # noc_depth_out = noc_depth_out * 16. + 28.
            # noc_depth_out = (1. / (noc_depth_out.sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1).unsqueeze(-1)
            # noc_depth_offset_out = NOC_depth_offset[:, 0, :, :]
            #
            # # noc_depth_out_uncern = NOC_depth[:, 1, :, :]
            # noc_depth_out_uncern = torch.logsumexp(torch.stack([NOC_depth[:, 1, :, :],
            #                                                     depth_geo_log_std.unsqueeze(-1).repeat(1, 7, 7)], -1),
            #                                        -1)
            # noc_depth_offset_out_uncern = NOC_depth_offset[:, 1, :, :]
            #
            # noc_merge_depth_out = noc_depth_out + noc_depth_offset_out
            # noc_merge_depth_out_uncern = torch.logsumexp(torch.stack([noc_depth_offset_out_uncern, noc_depth_out_uncern],
            #                                                          -1), -1)

            '''do not use geo depth'''
            noc_depth_out = NOC_depth[:, 0, :, :]
            # noc_depth_out = noc_depth_out * 16. + 28.
            noc_depth_out = (-noc_depth_out).exp()
            # caddn_noc_depth_out = NOC_depth[:, :96, :, :]
            # noc_depth_out = self.decode_caddn_depth(caddn_noc_depth_out)
            noc_depth_out = noc_depth_out * scale_depth.unsqueeze(-1).unsqueeze(-1)
            noc_depth_offset_out = NOC_depth_offset[:, 0, :, :]
            # noc_depth_offset_out = noc_depth_offset_out * scale_depth.unsqueeze(-1).unsqueeze(-1)

            # noc_depth_out_uncern = NOC_depth[:, 1, :, :]
            # noc_depth_offset_out_uncern = NOC_depth_offset[:, 1, :, :]
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


            # '''do not use geo depth'''
            # caddn_noc_depth_out = NOC_depth[:, :81, :, :]
            # noc_depth_out = self.decode_caddn_depth(caddn_noc_depth_out)
            # noc_depth_offset_out = NOC_depth_offset[:, 0, :, :]
            #
            # noc_depth_out_uncern = NOC_depth[:, 81, :, :]
            # noc_depth_offset_out_uncern = NOC_depth_offset[:, 1, :, :]
            #
            # noc_merge_depth_out = noc_depth_out + noc_depth_offset_out
            # noc_merge_depth_out_uncern = torch.logsumexp(torch.stack([noc_depth_offset_out_uncern, noc_depth_out_uncern],
            #                                                          -1), -1)
            ###################################################################


            # '''surface depth'''
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,1:2],depth_geo_log_std],-1),-1,keepdim=True)
            # depth_net_1 = torch.cat([(1. / (depth_net_out[:,0:1].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,2:3],depth_geo_log_std],-1),-1,keepdim=True)
            # depth_net_2 = torch.cat([(1. / (depth_net_out[:,3:4].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,4:5],depth_geo_log_std],-1),-1,keepdim=True)
            # depth_net_3 = torch.cat([(1. / (depth_net_out[:,5:6].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,6:7],depth_geo_log_std],-1),-1,keepdim=True)
            # depth_net_4 = torch.cat([(1. / (depth_net_out[:,7:8].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)
            # # print("depth_net_4", depth_net_4.shape)
            # surface_depth = torch.cat([depth_net_1[..., 0:1],
            #                            depth_net_2[..., 0:1],
            #                            depth_net_3[..., 0:1],
            #                            depth_net_4[..., 0:1]], -1)
            # uncertainty_depth = torch.cat([depth_net_1[..., 1:2],
            #                                depth_net_2[..., 1:2],
            #                                depth_net_3[..., 1:2],
            #                                depth_net_4[..., 1:2]], -1)
            # # print("surface_depth", surface_depth.shape)
            # vis_depth = torch.softmax(depth_net_out[:,8:12], dim=1)
            # '''surface depth'''
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,1:2],depth_geo_log_std],-1),-1,keepdim=True)
            # depth_net_1 = torch.cat([(1. / (depth_net_out[:,0:1].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)
            # depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,2:3],depth_geo_log_std],-1),-1,keepdim=True)
            # depth_net_2 = torch.cat([(1. / (depth_net_out[:,3:4].sigmoid() + 1e-6) - 1.)+depth_geo.unsqueeze(-1),depth_net_log_std],-1)
            # surface_depth = torch.cat([depth_net_1[..., 0:1],
            #                            depth_net_2[..., 0:1]], -1)
            # uncertainty_depth = torch.cat([depth_net_1[..., 1:2],
            #                                depth_net_2[..., 1:2]], -1)
            # vis_depth = torch.softmax(depth_net_out[:, 8:10], dim=1)


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


            # res['surface_depth'] = surface_depth
            # res['uncertainty_depth'] = uncertainty_depth
            # res['vis_depth'] = vis_depth

            ####################################################################
            '''corners offset '''
            res['corners_offset_3d'] = self.corners_offset_3d(roi_feature_masked)[:,:,0,0]
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

            res['corners_offset_3d'] = torch.zeros([1,16]).to(device_id)



        return res


    # def decode_caddn_depth(self, depth_logits):
    #     self.disc_cfg = {"mode": "LID",
    #                      "num_bins": 80,
    #                      "depth_min": 2.0,
    #                      "depth_max": 46.8}
    #     num_bins = self.disc_cfg['num_bins']
    #     depth_min = self.disc_cfg['depth_min']
    #     depth_max = self.disc_cfg['depth_max']
    #
    #     bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
    #     indices = torch.argmax(depth_logits, dim=1)
    #     depth_map = depth_min + ((((indices + 0.5) * 2) ** 2) * bin_size - 1) / 8
    #
    #     return depth_map


    def decode_caddn_depth(self, depth_logits):
        self.disc_cfg = {"mode": "LID",
                         "num_bins": 96,
                         "depth_min": 1.0,
                         "depth_max": 80}
        num_bins = self.disc_cfg['num_bins']
        depth_min = self.disc_cfg['depth_min']
        depth_max = self.disc_cfg['depth_max']

        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = torch.arange(num_bins).type(torch.float32).cuda()
        depth_scatter = depth_min + ((((indices + 0.5) * 2) ** 2) * bin_size - 1) / 8
        depth_prob = F.softmax(depth_logits, dim=1)
        depth_val = torch.sum(depth_scatter.view(1, num_bins, 1, 1)*depth_prob, dim=1)

        return depth_val



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
    # def get_roi_feat(self,feat,input,inds,mask,ret,calibs,coord_ranges,cls_ids):
    #     BATCH_SIZE,_,HEIGHT,WIDE = feat[0].size()
    #     device_id = feat[0].device
    #     coord_map = torch.cat([torch.arange(WIDE).unsqueeze(0).repeat([HEIGHT,1]).unsqueeze(0),\
    #                     torch.arange(HEIGHT).unsqueeze(-1).repeat([1,WIDE]).unsqueeze(0)],0).unsqueeze(0).repeat([BATCH_SIZE,1,1,1]).type(torch.float).to(device_id)
    #     box2d_centre = coord_map + ret['offset_2d']
    #     box2d_maps = torch.cat([box2d_centre-ret['size_2d']/2,box2d_centre+ret['size_2d']/2],1)
    #     box2d_maps = torch.cat([torch.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat([1,1,HEIGHT,WIDE]).type(torch.float).to(device_id),box2d_maps],1)
    #     #box2d_maps is box2d in each bin
    #     res = self.get_temploral_roi_feat_by_mask(feat,input,box2d_maps,inds,mask,calibs,coord_ranges,cls_ids)
    #     return res


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
