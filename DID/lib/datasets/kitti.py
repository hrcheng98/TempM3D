import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.utils import get_angle_from_box3d,check_range
from lib.datasets.kitti_utils import get_objects_from_label
from lib.datasets.kitti_utils import Calibration
from lib.datasets.kitti_utils import get_affine_transform
from lib.datasets.kitti_utils import affine_transform
from lib.datasets.kitti_utils import compute_box_3d
import pdb

import cv2 as cv
import torchvision.ops.roi_align as roi_align
import math
from lib.datasets.kitti_utils import Object3d

from datetime import datetime



def convert_to_3d(self, depth, P2, upsample_factor, x_start, y_start):
    '''
    :param depth: depth map of current frame cropped area. SHAPE: A*B
    :param P2: projection matrix of left RGB camera.  SHAPE: 4*3
    :param upsample_factor: upsample factor of the cropped area.
    :param x_start: start coordinates in image coordinates of x.
    :param y_start: start coordinates in image coordinates of y.
    :return:
            points: 3D coordinates in real world of cropped area.   SHAPE: N*3
            uv_points: corresponding 2D coordinates in image coordinates of 3D points  SHAPE: N*2
    '''
    fx = P2[0][0] * upsample_factor
    fy = P2[1][1] * upsample_factor
    cx = P2[0][2] * upsample_factor
    cy = P2[1][2] * upsample_factor

    b_x = P2[0][3] * upsample_factor / (-fx)
    b_y = P2[1][3] * upsample_factor / (-fy)
    # print(fx, fy, cx, cy)

    x_tile = np.array(range(depth.shape[1])).reshape(1, -1) + x_start
    points_x = np.tile(x_tile, [depth.shape[0], 1])

    y_tile = np.array(range(depth.shape[0])).reshape(-1, 1) + y_start
    points_y = np.tile(y_tile, [1, depth.shape[1]])

    points_x = points_x.reshape((-1, 1))
    points_y = points_y.reshape((-1, 1))
    depth = depth.reshape((-1, 1))

    # # -------mask-------
    # mask = np.where(depth != np.inf, True, False)
    # points_x = points_x[mask].reshape((-1, 1))
    # points_y = points_y[mask].reshape((-1, 1))
    # depth = depth[mask].reshape((-1, 1))

    uv_points = np.concatenate([points_x, points_y], axis=1)

    points_x = (points_x - cx) / fx
    points_y = (points_y - cy) / fy

    points_x = points_x * depth + b_x
    points_y = points_y * depth + b_y

    points = np.concatenate([points_x, points_y, depth], axis=1)

    return points, uv_points




class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])
        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        ''' 
        ##l,w,h
        self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ],
                                       [1.52563191462 ,1.62856739989, 3.88311640418],
                                       [1.73698127    ,0.59706367   , 1.76282397   ]])                              
                              
        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(root_dir, cfg['data_dir'], 'ImageSets', split + '.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, cfg['data_dir'], 'testing' if split == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode


    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def project_3d(self, p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
        """
        Projects a 3D box into 2D vertices

        Args:
            p2 (nparray): projection matrix of size 4x3
            x3d: x-coordinate of center of object
            y3d: y-coordinate of center of object
            z3d: z-cordinate of center of object
            w3d: width of object
            h3d: height of object
            l3d: length of object
            ry3d: rotation w.r.t y-axis
        """

        # compute rotational matrix around yaw axis
        R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
                      [0, 1, 0],
                      [-math.sin(ry3d), 0, +math.cos(ry3d)]])

        # 3D bounding box corners
        x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
        y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
        z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

        x_corners += -l3d / 2
        y_corners += -h3d / 2
        z_corners += -w3d / 2

        # bounding box in object co-ordinate
        corners_3d = np.array([x_corners, y_corners, z_corners])

        # rotate
        corners_3d = R.dot(corners_3d)

        # translate
        corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))

        # corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
        # corners_2D = p2.dot(corners_3D_1)
        # corners_2D = corners_2D / corners_2D[2]
        #
        # bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
        #
        # verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T
        #
        # if return_3d:
        #     return verts3d, corners_3d
        # else:
        #     return verts3d
        return corners_3d


    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)


        RoI_align_size = 7

        ####################################################################
        '''LOCAL DENSE NOC depths'''
        if index > 7480:
            '''for test'''
            d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(7480), -1) / 256.
            # d = cv.imread('/private_data/personal/pengliang/caddn_label_depth_2/depth_2/{:0>6}.png'.format(7480), -1) / 256.
        else:
            # d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(index), -1) / 256.
            d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/training/lidar_depth/{:0>6}.png'.format(index), -1) / 256.
            # d = cv.imread('/private_data/personal/pengliang/caddn_label_depth_2/depth_2/{:0>6}.png'.format(index), -1) / 256.
            # d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/testing/lidar_depth/{:0>6}.png'.format(index), -1) / 256.

        dst_W, dst_H = img_size
        pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
        pad_wr = dst_W - pad_w - d.shape[1]
        d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
        d = Image.fromarray(d)
        ####################################################################

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False

        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                d = d.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1 + self.scale)
                # crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

        depth_scale_factor = crop_size[1] / self.resolution[1]

        ####################################################################
        '''LOCAL DENSE NOC depths'''
        # '''sparse lidar depth'''
        # d2 = (Image.fromarray(d)).transform(tuple(self.resolution.tolist()),
        #                     method=Image.AFFINE,
        #                     data=tuple(trans_inv.reshape(-1).tolist()),
        #                     resample=Image.NEAREST)
        # d2 = np.array(d2)
        # down_d = cv.resize(d2, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
        #                    interpolation=cv.INTER_NEAREST)
        '''dense lidar depth'''
        d2 = d.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            # resample=Image.BILINEAR)
                            resample=Image.NEAREST)
        d2 = np.array(d2)
        # d2 = cv.warpAffine(d, trans_inv, tuple(self.resolution.tolist()), flags=cv.INTER_LINEAR)
        down_d = cv.resize(d2, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
                           interpolation=cv.INTER_AREA)
        ####################################################################

        coord_range = np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W


        calib = self.get_calib(index)

        # new_P2 = calib.P2.copy()
        # new_P2[[0, 1], 2] -= np.array(img_size) / 2 - center
        # new_P2[0, 0] /= (crop_size/self.resolution)[0]
        # new_P2[1, 1] /= (crop_size/self.resolution)[1]


        features_size = self.resolution // self.downsample# W * H
        #  ============================   get labels   ==============================
        if self.split!='test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32) # C * H * W
            size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
            depth = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)

            new_loc_3d = np.zeros((self.max_objs, 3), dtype=np.float32) # for debug
            import vis
            visBox = vis.Vis3d([-30, 30], [0, 100], 100)
            p_3d, _ = convert_to_3d(1, np.array(d2 * depth_scale_factor), calib.P2, 1, 0, 0)
            visBox.add_points(p_3d[:, [0, 2]], (255, 255, 0), thk=2)



            # if torch.__version__ == '1.10.0+cu113':
            if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0', '1.4.0']:
                mask_2d = np.zeros((self.max_objs), dtype=np.bool)
            else:
                mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            # '''for 30 cards'''
            # mask_2d = np.zeros((self.max_objs), dtype=np.bool)
            mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            ####################################################################
            '''LOCAL DENSE NOC depths'''
            # abs_noc_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
            # noc_depth_offset = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
            # noc_depth_mask = np.zeros((self.max_objs, 7, 7), dtype=np.bool)
            abs_noc_depth = np.zeros((self.max_objs, RoI_align_size, RoI_align_size), dtype=np.float32)
            noc_depth_offset = np.zeros((self.max_objs, RoI_align_size, RoI_align_size), dtype=np.float32)
            noc_depth_mask = np.zeros((self.max_objs, RoI_align_size, RoI_align_size), dtype=np.bool)
            bbox_list = []
            ####################################################################
            #
            ####################################################################
            '''Corner 3d points'''
            # corners_offset_3d = np.zeros((self.max_objs, 8, 2), dtype=np.float32)
            corners_offset_3d = np.zeros((self.max_objs, 16), dtype=np.float32)
            ####################################################################



            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue
    
                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                # '''remove far'''
                # if objects[i].pos[-1] > 60:
                #     continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # bbox_list.append(bbox_2d)
    
                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                ###########################################################################
                '''Corner 3d points'''
                points_3d = self.project_3d(calib.P2, (objects[i].pos)[0], (objects[i].pos)[1]-objects[i].h / 2, (objects[i].pos)[2],
                                             objects[i].w, objects[i].h, objects[i].l, objects[i].ry)
                corners_3d, _ = calib.rect_to_img(points_3d.T)
                stack_corners = []
                for ii in range(8):
                    stack_corners.append(affine_transform(corners_3d[ii].reshape(-1), trans))
                    stack_corners[ii] /= self.downsample
                stack_corners = np.stack(stack_corners, axis=0)
                # stack_corners = np.concatenate(stack_corners, axis=0)
                # print('center_3d: {}, stack_corners: {}, corners_3d:{}, points_3d:{}'.format(center_3d, stack_corners, corners_3d, points_3d))
                ###########################################################################
            
                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
    
                # generate the radius of gaussian heatmap
                w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                radius = gaussian_radius((w, h))
                radius = max(0, int(radius))
    
                if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
                    draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                    continue
    
                cls_id = self.cls2id[objects[i].cls_type]
                cls_ids[i] = cls_id
                draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
    
                # encoding 2d/3d offset & 2d size
                indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                offset_2d[i] = center_2d - center_heatmap
                size_2d[i] = 1. * w, 1. * h
    
                # encoding depth
                depth[i] = objects[i].pos[-1]

                # encoding heading angle
                #heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)


                new_loc_3d[i, 2] = objects[i].pos[-1] * depth_scale_factor
                new_loc_3d[i, 0] = ((center_3d[0]*self.downsample - calib.P2[0, 2])*new_loc_3d[i, 2] - calib.P2[0, 3]) / calib.P2[0, 0]
                new_loc_3d[i, 1] = ((center_3d[1]*self.downsample - calib.P2[1, 2])*new_loc_3d[i, 2] - calib.P2[1, 3]) / calib.P2[1, 1]
                p_3d, _ = convert_to_3d(1, np.array(d2 * depth_scale_factor), calib.P2, 1, 0, 0)
                visBox.add_bev_box(new_loc_3d[i, [0, 2]], np.array([objects[i].w, objects[i].l], dtype=np.float32),
                                  -(calib.alpha2ry(heading_angle, (bbox_2d[0]*self.downsample+bbox_2d[2]*self.downsample)/2)-np.pi/2), (255, 255, 255), thk=2)
                visBox.add_bev_box(new_loc_3d[i, [0, 2]], np.array([objects[i].w, objects[i].l], dtype=np.float32),
                                  -(objects[i].ry-np.pi/2), (255, 255, 0), thk=2)


                # encoding 3d offset & size_3d
                ###########################################################################
                '''Corner 3d points'''
                stack_corners_offset = stack_corners - center_heatmap[np.newaxis, :]
                corners_offset_3d[i] = stack_corners_offset.reshape(-1)
                ###########################################################################

                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size

                # print('corners_offset_3d: {}, offset_3d: {}'.format(corners_offset_3d[i], offset_3d[i]))


                #objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                    mask_2d[i] = 1

                ####################################################################
                '''LOCAL DENSE NOC depths'''
                down_d_copy = down_d.copy()
                # bbox_2d_int = bbox_2d.astype(np.int32)
                # bbox_2d_int[bbox_2d_int < 0] = 0
                # roi_depth = down_d_copy[bbox_2d_int[1]:bbox_2d_int[3] + 1, bbox_2d_int[0]:bbox_2d_int[2] + 1]
                # roi_depth_ind = (roi_depth < depth[i] - 3) | (roi_depth > depth[i] + 3)
                # down_d_copy[bbox_2d_int[1]:bbox_2d_int[3] + 1, bbox_2d_int[0]:bbox_2d_int[2] + 1][roi_depth_ind] = 0
                #
                # '''for dense lidar depth can only use INTER_NEAREST mode !!!!!!'''
                # abs_noc_depth[i] = cv.resize(down_d_copy[bbox_2d_int[1]:bbox_2d_int[3]+1, bbox_2d_int[0]:bbox_2d_int[2]+1],
                #                           (7, 7), interpolation=cv.INTER_NEAREST)
                # noc_depth_mask[i] = abs_noc_depth[i] > 0
                # noc_depth_offset[i] = depth[i] - abs_noc_depth[i]

                # roi_depth = roi_align(torch.from_numpy(down_d_copy).unsqueeze(0).unsqueeze(0).type(torch.float32),
                #                       [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
                roi_depth = roi_align(torch.from_numpy(down_d_copy).unsqueeze(0).unsqueeze(0).type(torch.float32),
                                      [torch.tensor(bbox_2d).unsqueeze(0)], [RoI_align_size, RoI_align_size]).numpy()[0, 0]
                roi_depth_ind = (roi_depth > depth[i] - 3) & \
                                (roi_depth < depth[i] + 3) & \
                                (roi_depth > 0)
                # roi_depth_ind = (roi_depth > 0)
                roi_depth[~roi_depth_ind] = 0
                abs_noc_depth[i] = roi_depth
                noc_depth_mask[i] = roi_depth_ind
                noc_depth_offset[i] = depth[i] - abs_noc_depth[i]

                ####################################################################


            # '''cannot fit sparse lidar depth'''
            # roi_noc_depth = roi_align(torch.from_numpy(down_d).unsqueeze(0).unsqueeze(0).type(torch.float32), [torch.tensor(np.array(bbox_list))], [7, 7])

            # if np.sum(mask_2d) == 0:
            #     print('badfile', '/pvc_data/personal/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(index))

            targets = {
                   'depth': depth,
                   'size_2d': size_2d,
                   'heatmap': heatmap,
                   'offset_2d': offset_2d,
                   'indices': indices,
                   'size_3d': size_3d,
                   'offset_3d': offset_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'cls_ids': cls_ids,
                   'mask_2d': mask_2d,

                   'abs_noc_depth': abs_noc_depth,
                   'noc_depth_mask': noc_depth_mask,
                   'noc_depth_offset': noc_depth_offset,

                   'pixel_depth': torch.from_numpy(down_d).unsqueeze(0).type(torch.float32) / depth_scale_factor / 4,

                   'corners_offset_3d': corners_offset_3d,

                   # 'depth_surface': depth_surface,
                       }
        else:
            targets = {}

        map = visBox.get_map()
        cv.imwrite('tmp.png', map)
        visBox.reset()

        inputs = img

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size/features_size}
        b = datetime.now()
        # print('data loading', b-a ,b )

        # print([(k, inputs[k].shape) for k in inputs.keys()])
        # print([(k, targets[k].shape) for k in targets.keys()])
        # print('===============')
        return inputs, calib.P2, coord_range, targets, info   #calib.P2


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    cfg = {'random_flip':0.0, 'random_crop':1.0, 'scale':0.4, 'shift':0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist':['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center':False,
           'data_dir': '/pvc_user/pengliang/DID/DID-main/KITTI_pvc',
           'root_dir': '/pvc_user/pengliang/DID/DID-main'
           }
    dataset = KITTI('/pvc_user/pengliang/DID/DID-main/KITTI_pvc', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    # for batch_idx, (inputs, targets, info) in enumerate(dataloader):
    for batch_idx, (inputs, P2, coord_range, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        # img = (img * dataset.std + dataset.mean) * 255
        # img = Image.fromarray(img.astype(np.uint8))
        # img.show()
        # # print(targets['size_3d'][0][0])
        #
        # # test heatmap
        # heatmap = targets['heatmap'][0]  # image id
        # heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        # heatmap.show()
        #
        # break


    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
