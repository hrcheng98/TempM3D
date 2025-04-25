import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss_for_caddn
import operator

class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'],
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss']}


    def compute_weight(self,current_loss,epoch):
        T=140
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            # time_value = min(((epoch-2)/(T-2)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
                    if loss_weights[current_topic] != loss_weights[current_topic]:
                        for pre_topic in self.loss_graph[current_topic]:
                            print('NAN===============', time_value, control_weight, c_weights[0][self.term2index[pre_topic]], pre_topic, self.term2index[pre_topic])
                    # if torch.isnan(loss_weights[current_topic]):
                    #     print(loss_weights[current_topic], current_topic)
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)

        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class GupnetLoss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch

        self.count = 0
        # self.moving_depth_loss = []

        # self.CaDDN_loss = DDNLoss()

    def forward(self, preds, targets, task_uncertainties=None):

        if targets['mask_2d'].sum() == 0:
            print('mask_2d = 0')
            bbox2d_loss = 0
            bbox3d_loss = 0
            self.stat['offset2d_loss'] = 0
            self.stat['size2d_loss'] = 0
            self.stat['depth_loss'] = 0
            self.stat['offset3d_loss'] = 0
            self.stat['size3d_loss'] = 0
            self.stat['heading_loss'] = 0
        else:
            bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets)


        seg_loss = self.compute_segmentation_loss(preds, targets)
        
        loss = seg_loss + bbox2d_loss + bbox3d_loss

        self.count += 1

        return loss, self.stat
        # return float(loss), self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')


        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        RoI_align_size = 7

        ####################################################################
        '''LOCAL DENSE NOC depths'''
        if 'noc_depth_out' not in input.keys():
        # if False:
            depth_loss = 0
            offset3d_loss = 0
            size3d_loss = 0
            heading_loss = 0
            loss = 0
            print('not good')
        else:
            depth_target = extract_target_from_tensor(target['depth'], target[mask_type])

            abs_noc_depth = input['noc_depth_out'][input['train_tag']]
            noc_depth_offset = input['noc_depth_offset_out'][input['train_tag']]

            abs_noc_depth_target = extract_target_from_tensor(target['abs_noc_depth'], target[mask_type])
            noc_depth_offset_target = extract_target_from_tensor(target['noc_depth_offset'], target[mask_type])
            noc_depth_mask_target = extract_target_from_tensor(target['noc_depth_mask'], target[mask_type])


            '''with uncernt'''
            abs_noc_depth_uncern = input['noc_depth_out_uncern'][input['train_tag']]
            noc_depth_offset_uncern = input['noc_depth_offset_out_uncern'][input['train_tag']]
            abs_noc_depth_loss = laplacian_aleatoric_uncertainty_loss(abs_noc_depth[noc_depth_mask_target],
                                                                       abs_noc_depth_target[noc_depth_mask_target],
                                                                       abs_noc_depth_uncern[noc_depth_mask_target])

            noc_depth_offset_loss = laplacian_aleatoric_uncertainty_loss(noc_depth_offset[noc_depth_mask_target],
                                                                          noc_depth_offset_target[noc_depth_mask_target],
                                                                          noc_depth_offset_uncern[noc_depth_mask_target])
            depth_loss = abs_noc_depth_loss + noc_depth_offset_loss


            noc_merge_depth = input['noc_merge_depth_out'][input['train_tag']]
            noc_merge_depth_uncern = input['noc_merge_depth_out_uncern'][input['train_tag']]
            merge_depth_loss = laplacian_aleatoric_uncertainty_loss(
                                            noc_merge_depth.view(-1, RoI_align_size*RoI_align_size),
                                            depth_target.repeat(1, RoI_align_size*RoI_align_size),
                                            noc_merge_depth_uncern.view(-1, RoI_align_size*RoI_align_size))

            depth_loss += merge_depth_loss

            depth_loss = depth_loss.mean()


            # compute offset3d loss
            offset3d_input = input['offset_3d'][input['train_tag']]
            offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
            offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')


            # compute size3d loss
            size3d_input = input['size_3d'][input['train_tag']]
            size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
            # size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
            #        laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])/3
            # size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
            #        (torch.abs(size3d_input[:,0:1] - size3d_target[:,0:1])).mean()/3
            # print(size3d_input[0], size3d_target[0])
            size3d_loss = F.smooth_l1_loss(size3d_input, size3d_target)
            # size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
            #        (torch.abs(size3d_input[:,0:1] - size3d_target[:,0:1])).mean()/3 + corners_offset_3d_loss


            #size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')+\
            #       laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])
            # compute heading loss
            heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                                target[mask_type],  ## NOTE
                                                target['heading_bin'],
                                                target['heading_res'])
            # print(target['heading_bin'].view(-1)[target[mask_type].view(-1)], torch.argmax(input['heading'][input['train_tag']][:, :12], dim=1))
            # loss = depth_loss + offset3d_loss + size3d_loss + heading_loss
            # loss = depth_loss + offset3d_loss + size3d_loss + heading_loss + corners_offset_3d_loss
            loss = depth_loss + offset3d_loss + size3d_loss + heading_loss

            if depth_loss != depth_loss:
                print('badNAN----------------depth_loss', depth_loss)
                exit(0)
            if offset3d_loss != offset3d_loss:
                print('badNAN----------------offset3d_loss', offset3d_loss)
            if size3d_loss != size3d_loss:
                print('badNAN----------------size3d_loss', size3d_loss)
            if depth_loss != depth_loss:
                print('badNAN----------------heading_loss', heading_loss)

        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        
        return loss


# CUDA_VISIBLE_DEVICES=0,1,2 python tools/train_val.py --config experiments/0302.yaml
# CUDA_VISIBLE_DEVICES=2,3 python tools/train_val.py --config experiments/0302.yaml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_val.py --config experiments/0302.yaml

### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]

#compute heading loss two stage style  

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss    
'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''

import math
def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices


# import kornia
# class DDNLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         """
#         Initializes DDNLoss module
#         Args:
#             weight [float]: Loss function weight
#             alpha [float]: Alpha value for Focal Loss
#             gamma [float]: Gamma value for Focal Loss
#             disc_cfg [dict]: Depth discretiziation configuration
#             fg_weight [float]: Foreground loss weight
#             bg_weight [float]: Background loss weight
#             downsample_factor [int]: Depth map downsample factor
#         """
#         super().__init__()
#         self.disc_cfg = {"mode": "LID",
#                          "num_bins": 80,
#                          "depth_min": 2.0,
#                          "depth_max": 46.8}
#
#         # Set loss function
#         self.alpha = alpha
#         self.gamma = gamma
#         self.loss_func = kornia.losses.FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
#
#     def forward(self, depth_logits, depth_maps):
#         """
#         Gets DDN loss
#         Args:
#             depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
#             depth_maps: torch.Tensor(B, H, W)]: Depth map [m]
#             gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
#         Returns:
#             loss [torch.Tensor(1)]: Depth classification network loss
#             tb_dict [dict[float]]: All losses to log in tensorboard
#         """
#         # Bin depth map to create target
#         depth_target = bin_depths(depth_maps, **self.disc_cfg, target=True)
#
#         # # Apply softmax along depth axis and remove last depth category (> Max Range)
#         # depth_probs = F.softmax(depth_logits, dim=1)
#         # depth_logits = depth_probs[:, :, :-1]
#
#         # bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
#         # indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
#
#         # Compute loss
#         depth_loss = self.loss_func(depth_logits, depth_target)
#
#         return depth_loss



if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

