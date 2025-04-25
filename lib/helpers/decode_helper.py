import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle

def decode_detections(dets, info, calibs, cls_mean_size, threshold, problist=None):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''

    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold: continue


            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]


            # noc_depth_out_prob, noc_depth_offset_out_prob, noc_merge_depth_out_prob = problist
            # im1 = convert_color_map(noc_depth_out_prob[i, j], size=(int(w), int(h)))
            # im2 = convert_color_map(noc_depth_offset_out_prob[i, j], size=(int(w), int(h)))
            # im3 = convert_color_map(noc_merge_depth_out_prob[i, j], size=(int(w), int(h)))
            #
            # img_id = '{:0>6}.png'.format(info['img_id'][i])
            #
            # rgb = cv.imread('/private/personal/pengliang/KITTI3D/training/image_2/{}'.format(img_id))
            # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tools/uncer_vis/{}'.format(img_id), rgb)
            # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tools/uncer_vis/{}_{}_vis.png'.format(img_id, j), im1)
            # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tools/uncer_vis/{}_{}_att.png'.format(img_id, j), im2)
            # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tools/uncer_vis/{}_{}_ins.png'.format(img_id, j), im3)
            # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tools/uncer_vis/{}_{}_rgb.png'.format(img_id, j),
            #            rgb[max(int(y-h/2), 0):int(y+h/2), max(int(x-w/2), 0):int(x+w/2)])



            # 3d bboxs decoding
            # depth decoding
            # '''only use instance depth'''
            # depth = dets[i, j, 6]
            depth = dets[i, j, -2]
            score *= dets[i, j, -1]

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            if True in (dimensions<0.0): continue

            # positions decoding
            x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            # depth = dets[i, j, 6:10]
            # # depth_score = dets[i, j, -4:]
            # depth_score = dets[i, j, -8:-4]
            # vis_depth = dets[i, j, -4:]
            #
            # # heading angle decoding
            # alpha = get_heading_angle(dets[i, j, 7+3:31+3])
            # ry = calibs[i].alpha2ry(alpha, x)
            #
            # # dimensions decoding
            # dimensions = dets[i, j, 31+3:34+3]
            # dimensions += cls_mean_size[int(cls_id)]
            # if True in (dimensions<0.0): continue
            #
            # # positions decoding
            # x3d = dets[i, j, 34+3] * info['bbox_downsample_ratio'][i][0]
            # y3d = dets[i, j, 35+3] * info['bbox_downsample_ratio'][i][1]
            #
            # h, w, l = dimensions
            # depth[0] = depth[0] + l/2 * np.sin(ry)
            # depth[1] = depth[1] + w/2 * np.cos(ry)
            # depth[2] = depth[2] - l/2 * np.sin(ry)
            # depth[3] = depth[3] - w/2 * np.cos(ry)
            # depth = vis_depth[0] * depth[0] + vis_depth[1] * depth[1] + \
            #         vis_depth[2] * depth[2] + vis_depth[3] * depth[3]
            # depth_score = vis_depth[0] * depth_score[0] + vis_depth[1] * depth_score[1] + \
            #               vis_depth[2] * depth_score[2] + vis_depth[3] * depth_score[3]
            #
            # score *= depth_score
            #
            # locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            # locations[1] += dimensions[0] / 2

            #
            # depth = dets[i, j, 6:8]
            # # depth_score = dets[i, j, -2:]
            # depth_score = dets[i, j, -4:-2]
            # vis_depth = dets[i, j, -2:]
            # # print(vis_depth)
            #
            # # heading angle decoding
            # alpha = get_heading_angle(dets[i, j, 7+1:31+1])
            # ry = calibs[i].alpha2ry(alpha, x)
            #
            # # dimensions decoding
            # dimensions = dets[i, j, 31+1:34+1]
            # dimensions += cls_mean_size[int(cls_id)]
            # if True in (dimensions<0.0): continue
            #
            # # positions decoding
            # x3d = dets[i, j, 34+1] * info['bbox_downsample_ratio'][i][0]
            # y3d = dets[i, j, 35+1] * info['bbox_downsample_ratio'][i][1]
            #
            #
            # h, w, l = dimensions
            # depth[0] = depth[0] + l/2 * np.abs(np.sin(ry))
            # depth[1] = depth[1] + w/2 * np.abs(np.cos(ry))
            # # depth_score = 0.5 * depth_score[0] + 0.5 * depth_score[1]
            # # depth = 0.5 * depth[0] + 0.5 * depth[1]
            # depth_score = vis_depth[0] * depth_score[0] + vis_depth[1] * depth_score[1]
            # depth = vis_depth[0] * depth[0] + vis_depth[1] * depth[1]
            #
            #
            # score *= depth_score
            #
            # locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            # locations[1] += dimensions[0] / 2

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results

import cv2 as cv
def convert_color_map(img, mode=cv.INTER_LINEAR, size=49):
    # temp = img / np.max(img) * 255
    # temp = img / 50 * 255
    temp = cv.resize(img.cpu().numpy(), size, interpolation=mode) * 250
    temp = temp.astype(np.uint8)
    im_color = cv.applyColorMap(temp, cv.COLORMAP_JET)
    return im_color

#two stage style
def extract_dets_from_outputs(outputs, K=50):
    RoI_align_size = 7
    # RoI_align_size = 9
    # RoI_align_size = 13
    # RoI_align_size = 19
    # RoI_align_size = 5
    # RoI_align_size = 3
    # RoI_align_size = 2


    # get src outputs
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    batch, channel, height, width = heatmap.size() # get shape

    heading = outputs['heading'].view(batch,K,-1)
    depth = outputs['depth'].view(batch,K,-1)[:,:,0:1]
    # '''surface_depth'''
    # depth = outputs['surface_depth'].view(batch,K,4)
    # '''surface_depth'''
    # depth = outputs['surface_depth'].view(batch,K,2)

    ####################################################################
    '''LOCAL DENSE NOC depths'''
    # noc_depth_out = outputs['noc_depth_out'].view(batch,K,7,7)
    # noc_depth_offset_out = outputs['noc_depth_offset_out'].view(batch,K,7,7)
    noc_depth_out = outputs['noc_depth_out'].view(batch,K,RoI_align_size,RoI_align_size)
    noc_depth_offset_out = outputs['noc_depth_offset_out'].view(batch,K,RoI_align_size,RoI_align_size)
    noc_depth = noc_depth_out + noc_depth_offset_out

    # '''use gird instance depth'''
    # depth = (torch.mean(noc_depth_out.view(batch,K, -1), dim=-1)).unsqueeze(2)


    # noc_depth_out_uncern = outputs['noc_depth_out_uncern'].view(batch,K,7,7)
    # noc_depth_offset_out_uncern = outputs['noc_depth_offset_out_uncern'].view(batch,K,7,7)
    # noc_depth_out_uncern_prob = (-(0.5*noc_depth_out_uncern).exp()).exp()
    # noc_depth_offset_out_uncern_prob = (-(0.5*noc_depth_offset_out_uncern).exp()).exp()
    # merge_prob = noc_depth_out_uncern_prob * noc_depth_offset_out_uncern_prob
    # # merge_prob = noc_depth_out_uncern_prob
    # # merge_prob = noc_depth_offset_out_uncern_prob
    # noc_merge_depth_out_uncern = outputs['noc_merge_depth_out_uncern'].view(batch,K,7,7)
    noc_merge_depth_out_uncern = outputs['noc_merge_depth_out_uncern'].view(batch,K,RoI_align_size,RoI_align_size)
    merge_prob = (-(0.5 * noc_merge_depth_out_uncern).exp()).exp()
    # '''do not use uncern'''
    # merge_prob = torch.ones_like(merge_prob)

    # merge_prob = (-(0.1 * noc_merge_depth_out_uncern).exp()).exp()

    merge_depth = (torch.sum((noc_depth*merge_prob).view(batch,K,-1), dim=-1) /
                   torch.sum(merge_prob.view(batch,K,-1), dim=-1))
    merge_depth = merge_depth.unsqueeze(2)
    # '''use avg depth'''
    # merge_depth = (torch.mean(noc_depth.view(batch, K, -1), dim=-1)).unsqueeze(2)

    # ind = (merge_prob.view(batch,K,-1).max(-1))[-1]
    # merge_depth = (noc_depth.view(batch * K, 7 * 7)).gather(1, ind.view(-1).unsqueeze(-1)).view(batch, K)
    # merge_depth = merge_depth.unsqueeze(2)

    # merge_prob = merge_prob.view(batch,K,-1).mean(-1).unsqueeze(2)
    merge_prob = (merge_prob.view(batch,K,-1).max(-1))[0].unsqueeze(2)
    # merge_prob = (torch.sum(merge_prob.view(batch,K,-1)**2, dim=-1) / \
    #               torch.sum(merge_prob.view(batch,K,-1), dim=-1)).unsqueeze(2)


    #
    # '''
    # for vis
    # '''
    # noc_depth_out_uncern = outputs['noc_depth_out_uncern'].view(batch,K,RoI_align_size,RoI_align_size)
    # noc_depth_out_prob = (-(0.5 * noc_depth_out_uncern).exp()).exp()
    # noc_depth_offset_out_uncern = outputs['noc_depth_offset_out_uncern'].view(batch,K,RoI_align_size,RoI_align_size)
    # noc_depth_offset_out_prob = (-(0.5 * noc_depth_offset_out_uncern).exp()).exp()
    # noc_merge_depth_out_uncern = outputs['noc_merge_depth_out_uncern'].view(batch,K,RoI_align_size,RoI_align_size)
    # noc_merge_depth_out_prob = (-(0.5 * noc_merge_depth_out_uncern).exp()).exp()


    # noc_depth_mean = noc_depth.view(batch,K,-1).mean(-1).unsqueeze(2)
    # noc_depth_mean = torch.exp(-torch.std(noc_depth.view(batch,K,-1), -1) / 8).unsqueeze(2)
    # noc_depth_mean = merge_depth.unsqueeze(2)
    # depth = merge_depth.unsqueeze(2)
    ####################################################################


    size_3d = outputs['size_3d'].view(batch,K,-1)
    offset_3d = outputs['offset_3d'].view(batch,K,-1)

    heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)

    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.view(batch, K, 1).float()

    # grid_conf = outputs['noc_depth_out_uncern'].view(batch,K,RoI_align_size,RoI_align_size)
    # grid_prob = (-(0.5 * grid_conf).exp()).exp()

    # '''aggre'''
    # depth = (torch.sum((noc_depth_out*grid_prob).view(batch,K,-1), dim=-1) /
    #                torch.sum(grid_prob.view(batch,K,-1), dim=-1))
    # depth = depth.unsqueeze(2)

    depth_score = (-(0.5*outputs['depth'].view(batch,K,-1)[:,:,1:2]).exp()).exp()
    # scores = scores.view(batch, K, 1)*depth_score
    scores = scores.view(batch, K, 1)
    # scores = scores.view(batch, K, 1) * (grid_prob.view(batch,K,-1).max(-1))[0].unsqueeze(2)

    # depth_score = (-(0.5*outputs['uncertainty_depth'].view(batch,K,4)).exp()).exp()
    # vis_depth = outputs['vis_depth'].view(batch,K,4)
    # scores = scores.view(batch, K, 1)

    # depth_score = (-(0.5*outputs['uncertainty_depth'].view(batch,K,2)).exp()).exp()
    # scores = scores.view(batch, K, 1)

    # depth_score = (-(0.5*outputs['uncertainty_depth'].view(batch,K,2)).exp()).exp()
    # vis_depth = outputs['vis_depth'].view(batch, K, 2)
    # scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    # detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d], dim=2)
    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, merge_depth, merge_prob], dim=2)
    # detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, merge_depth, merge_prob], dim=2)
    # detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, depth_score, vis_depth], dim=2)
    # detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d, depth_score], dim=2)

    return detections
    # return detections, (noc_depth_out_prob, noc_depth_offset_out_prob, noc_merge_depth_out_prob)


'''
tensor([[13.2713, 13.7092, 14.2469, 13.3254, 12.6964, 12.6252, 13.0032],
        [11.6188, 12.4665, 11.6936, 11.4405, 11.0105, 10.9124, 13.5041],
        [ 9.8601,  9.7868,  9.6269,  9.7418,  9.7347, 10.4236, 12.8945],
        [ 8.7557,  8.2155,  8.6113,  9.0346,  9.3626, 10.2832, 12.2972],
        [ 8.2916,  7.9024,  8.2798,  8.8233,  9.1837, 10.5434, 11.8389],
        [ 8.5925,  8.0375,  8.0744,  8.7797,  9.2106, 10.3822, 10.3835],
        [ 8.7052,  8.1512,  8.1513,  9.0122,  9.1264,  9.1266,  8.9591]],
       device='cuda:0')

'''

#
# def decode_detections(dets, merge_depth, merge_prob, info, calibs, cls_mean_size, threshold):
#     '''
#     NOTE: THIS IS A NUMPY FUNCTION
#     input: dets, numpy array, shape in [batch x max_dets x dim]
#     input: img_info, dict, necessary information of input images
#     input: calibs, corresponding calibs for the input batch
#     output:
#     '''
#     results = {}
#     for i in range(dets.shape[0]):  # batch
#         preds = []
#         for j in range(dets.shape[1]):  # max_dets
#             cls_id = int(dets[i, j, 0])
#             score = dets[i, j, 1]
#             if score < threshold: continue
#
#             # 2d bboxs decoding
#             x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
#             y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
#             w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
#             h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
#             bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
#
#             # 3d bboxs decoding
#             # depth decoding
#             # depth = dets[i, j, -2]
#             # score *= dets[i, j, -1]
#
#             # heading angle decoding
#             alpha = get_heading_angle(dets[i, j, 7:31])
#             ry = calibs[i].alpha2ry(alpha, x)
#
#             # dimensions decoding
#             dimensions = dets[i, j, 31:34]
#             dimensions += cls_mean_size[int(cls_id)]
#             if True in (dimensions < 0.0): continue
#
#             # positions decoding
#             x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
#             y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
#
#
#             single_d = merge_depth[i, j].reshape(-1)
#             single_p = merge_prob[i, j].reshape(-1)
#
#             sort_args = np.argsort(single_d)
#             org_prob = single_p[sort_args]
#             sort_prob = org_prob / np.sum(org_prob)
#             sort_dep = single_d[sort_args]
#             #
#             # count_prob, candidate = 0, [0]
#             # for k in range(49):
#             #     count_prob += sort_prob[k]
#             #     if count_prob > 0.3333:
#             #     # if count_prob > 0.99:
#             #         count_prob = 0
#             #         candidate.append(k)
#
#
#             # sort_args = np.argsort(single_p)
#             # org_prob = single_p[sort_args]
#             # sort_prob = org_prob / np.sum(org_prob)
#             # sort_dep = single_d[sort_args]
#             # # print(sort_prob, sort_dep)
#
#             # candidate = [0, 17, 34, 50]
#             # candidate = [0, 25, 50]
#             candidate = [0, 50]
#
#             # print('====start')
#             for k in range(1, len(candidate)):
#             # for k in range(1, 2):
#                 dep_range = sort_dep[candidate[k-1]:candidate[k]]
#                 prob_range = org_prob[candidate[k-1]:candidate[k]]
#
#                 single_merge_dep = np.sum(dep_range * prob_range) / np.sum(prob_range)
#                 # single_merge_prob = np.max(prob_range) * score
#                 single_merge_prob = np.sum(prob_range ** 2) / np.sum(prob_range) * score
#                 # print(single_merge_dep, single_merge_prob)
#
#                 locations = calibs[i].img_to_rect(x3d, y3d, single_merge_dep).reshape(-1)
#                 locations[1] += dimensions[0] / 2
#                 preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, single_merge_prob])
#             # print('====end========')
#
#             # if np.max(sort_dep) - np.min(sort_dep) > 1:
#             #     locations = calibs[i].img_to_rect(x3d, y3d, single_merge_dep-0.5).reshape(-1)
#             #     locations[1] += dimensions[0] / 2
#             #     preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, single_merge_prob*0.9])
#             #
#             #     locations = calibs[i].img_to_rect(x3d, y3d, single_merge_dep+0.5).reshape(-1)
#             #     locations[1] += dimensions[0] / 2
#             #     preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, single_merge_prob*0.9])
#             #
#             # if np.max(sort_dep) - np.min(sort_dep) > 2:
#             #     locations = calibs[i].img_to_rect(x3d, y3d, single_merge_dep - 1).reshape(-1)
#             #     locations[1] += dimensions[0] / 2
#             #     preds.append(
#             #         [cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, single_merge_prob * 0.8])
#             #
#             #     locations = calibs[i].img_to_rect(x3d, y3d, single_merge_dep + 1).reshape(-1)
#             #     locations[1] += dimensions[0] / 2
#             #     preds.append(
#             #         [cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, single_merge_prob * 0.8])
#
#             # args_inter = np.argmin(np.abs(single_merge_dep - sort_dep))
#             # dep_range = sort_dep[0:args_inter]
#             # prob_range = org_prob[0:args_inter]
#             #
#             # cur_dep = np.sum(dep_range * prob_range) / np.sum(prob_range)
#             #
#             # locations = calibs[i].img_to_rect(x3d, y3d, cur_dep).reshape(-1)
#             # locations[1] += dimensions[0] / 2
#             # preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() +
#             #              [ry, single_merge_prob*np.exp(-abs(cur_dep - single_merge_dep))])
#             #
#             # dep_range = sort_dep[args_inter:]
#             # prob_range = org_prob[args_inter:]
#             #
#             # cur_dep = np.sum(dep_range * prob_range) / np.sum(prob_range)
#             #
#             # locations = calibs[i].img_to_rect(x3d, y3d, cur_dep).reshape(-1)
#             # locations[1] += dimensions[0] / 2
#             # preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() +
#             #              [ry, single_merge_prob*np.exp(-abs(cur_dep - single_merge_dep))])
#
#
#
#             # locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
#             # locations[1] += dimensions[0] / 2
#             #
#             # preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
#         results[info['img_id'][i]] = preds
#     return results

#
# # two stage style
# def extract_dets_from_outputs(outputs, K=50):
#     # get src outputs
#     heatmap = outputs['heatmap']
#     size_2d = outputs['size_2d']
#     offset_2d = outputs['offset_2d']
#
#     batch, channel, height, width = heatmap.size()  # get shape
#
#     heading = outputs['heading'].view(batch, K, -1)
#     depth = outputs['depth'].view(batch, K, -1)[:, :, 0:1]
#
#     ####################################################################
#     '''LOCAL DENSE NOC depths'''
#     noc_depth_out = outputs['noc_depth_out'].view(batch, K, 7, 7)
#     noc_depth_offset_out = outputs['noc_depth_offset_out'].view(batch, K, 7, 7)
#     merge_depth = noc_depth_out + noc_depth_offset_out
#
#     noc_merge_depth_out_uncern = outputs['noc_merge_depth_out_uncern'].view(batch, K, 7, 7)
#     merge_prob = (-(0.5 * noc_merge_depth_out_uncern).exp()).exp()
#     ####################################################################
#
#     size_3d = outputs['size_3d'].view(batch, K, -1)
#     offset_3d = outputs['offset_3d'].view(batch, K, -1)
#
#     heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
#
#     # perform nms on heatmaps
#     heatmap = _nms(heatmap)
#     scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)
#
#     offset_2d = _transpose_and_gather_feat(offset_2d, inds)
#     offset_2d = offset_2d.view(batch, K, 2)
#     xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
#     ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]
#
#     xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
#     ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]
#
#     cls_ids = cls_ids.view(batch, K, 1).float()
#     depth_score = (-(0.5 * outputs['depth'].view(batch, K, -1)[:, :, 1:2]).exp()).exp()
#     # scores = scores.view(batch, K, 1)*depth_score
#     scores = scores.view(batch, K, 1)
#
#     # check shape
#     xs2d = xs2d.view(batch, K, 1)
#     ys2d = ys2d.view(batch, K, 1)
#     xs3d = xs3d.view(batch, K, 1)
#     ys3d = ys3d.view(batch, K, 1)
#
#     size_2d = _transpose_and_gather_feat(size_2d, inds)
#     size_2d = size_2d.view(batch, K, 2)
#
#     detections = torch.cat(
#         [cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d], dim=2)
#     return detections, merge_depth.cpu().numpy(), merge_prob.cpu().numpy()

############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    if torch.__version__ == '1.6.0':
        topk_ys = (topk_inds // width).int().float()
    else:
        topk_ys = (topk_inds / width).int().float()
    # topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    if torch.__version__ == '1.6.0':
        topk_cls_ids = (topk_ind // K).int()
    else:
        topk_cls_ids = (topk_ind / K).int()
    # topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti import KITTI
    from torch.utils.data import DataLoader

    dataset = KITTI('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
