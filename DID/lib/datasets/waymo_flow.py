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
from lib.datasets.utils import get_angle_from_box3d, check_range
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

from  datetime import  datetime

class WaymoFlow(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        print('=======================')
        # basic configuration
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        self.resolution = np.array([1920//2, 1280//2])  # W * H
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
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       # [1.52563191462, 1.62856739989, 3.88311640418],
                                       [1.8, 2.0, 4.5],
                                       [1.73698127, 0.59706367, 1.76282397]])

        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split

        # path configuration
        self.data_dir = '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3'
        if self.split == 'test':
            self.data_dir = '/pvc_data/personal/pengliang/waymo_kitti_format/validation'
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        # self.label_dir = os.path.join(self.data_dir, 'filter_label_0')

        self.idx_list = [i[:-4] for i in sorted(os.listdir(self.image_dir))]

        # data augmentation configuration
        self.data_augmentation = True if split in ['train', 'trainval'] else False
        self.random_flip = cfg['random_flip']
        self.random_crop = cfg['random_crop']
        self.scale = cfg['scale']
        self.shift = cfg['shift']

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 4

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file).convert('RGB')  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def get_rot_surface_depth(self, ry, points_set):
        def _roty(t):
            ''' Rotation about the y-axis. '''
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s],
                             [0, 1, 0],
                             [-s, 0, c]])

        R = _roty(ry)
        rot_points = (np.dot(R, points_set.T)).T

        return rot_points

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
        a = datetime.now()
        # print('a', a)

        cv.setNumThreads(0)
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # index = 168
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)

        # pre_1_path = os.path.join(
        #     '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2',
        #     '{:0>6}.png'.format(int(self.idx_list[item-1])))
        # pre_2_path = os.path.join(
        #     '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2',
        #     '{:0>6}.png'.format(int(self.idx_list[item-2])))
        # cur_path = os.path.join(
        #     '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2',
        #     '{:0>6}.png'.format(index))
        pre_1_path = os.path.join(
            '/pvc_data/personal/pengliang/waymo_kitti_format/validation/image_2',
            '{:0>6}.png'.format(int(self.idx_list[item-1])))
        pre_2_path = os.path.join(
            '/pvc_data/personal/pengliang/waymo_kitti_format/validation/image_2',
            '{:0>6}.png'.format(int(self.idx_list[item-2])))
        cur_path = os.path.join(
            '/pvc_data/personal/pengliang/waymo_kitti_format/validation/image_2',
            '{:0>6}.png'.format(index))


        if ((index - int(self.idx_list[item-1]))) > 20 or ((index - int(self.idx_list[item-1])) < 0):
            pre_1_path = cur_path
        if ((int(self.idx_list[item-1]) - int(self.idx_list[item-2]) > 20)) or ((int(self.idx_list[item-1]) - int(self.idx_list[item-2]) < 0)) or ((int(self.idx_list[item]) - int(self.idx_list[item-2]) < 0)):
            pre_2_path = pre_1_path

        if not os.path.exists(pre_1_path):
            pre_1 = img.copy()
        else:
            pre_1 = cv.imread(pre_1_path)
        if not os.path.exists(pre_2_path):
            pre_2 = pre_1.copy()
        else:
            pre_2 = cv.imread(pre_2_path)

        # print('b', datetime.now())

        '''propogate frame'''
        # print('propogate frame')
        pre_2_1_flow = pre_1_path.replace('image_2', 'waymo_flow')
        pre_1_0_flow = cur_path.replace('image_2', 'waymo_flow')

        if not os.path.exists(pre_2_1_flow.replace('.png', '_u.png')):
            pre_2_1_flow = pre_1_0_flow

        # print(pre_2_1_flow.replace('.png', '_u.png'))
        pre_2_flow_u = cv.imread(pre_2_1_flow.replace('.png', '_u.png'), -1) / 10. - 1000.
        pre_1_flow_u = cv.imread(pre_1_0_flow.replace('.png', '_u.png'), -1) / 10. - 1000.
        pre_2_flow_v = cv.imread(pre_2_1_flow.replace('.png', '_v.png'), -1) / 10. - 1000.
        pre_1_flow_v = cv.imread(pre_1_0_flow.replace('.png', '_v.png'), -1) / 10. - 1000.
        pre_2_flow = np.stack([pre_2_flow_u, pre_2_flow_v], axis=2)
        pre_1_flow = np.stack([pre_1_flow_u, pre_1_flow_v], axis=2)

        pre_2_flow_u_bk = cv.imread(pre_2_1_flow.replace('.png', '_bk_u.png'), -1) / 10. - 1000.
        pre_1_flow_u_bk = cv.imread(pre_1_0_flow.replace('.png', '_bk_u.png'), -1) / 10. - 1000.
        pre_2_flow_v_bk = cv.imread(pre_2_1_flow.replace('.png', '_bk_v.png'), -1) / 10. - 1000.
        pre_1_flow_v_bk = cv.imread(pre_1_0_flow.replace('.png', '_bk_v.png'), -1) / 10. - 1000.
        pre_2_flow_bk = np.stack([pre_2_flow_u_bk, pre_2_flow_v_bk], axis=2)
        pre_1_flow_bk = np.stack([pre_1_flow_u_bk, pre_1_flow_v_bk], axis=2)

        # print('c', datetime.now())


        # '''for original RAFT 1/1'''
        # pad_ht = (((img_size[1] // 8) + 1) * 8 - img_size[1]) % 8
        # pad_wd = (((img_size[0] // 8) + 1) * 8 - img_size[0]) % 8
        # pad_v = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        #
        # pre_2_flow = pre_2_flow[pad_v[2]:pad_v[2] + img_size[1], pad_v[0]:pad_v[0] + img_size[0], :]
        # pre_1_flow = pre_1_flow[pad_v[2]:pad_v[2] + img_size[1], pad_v[0]:pad_v[0] + img_size[0], :]
        # pre_2_flow_bk = pre_2_flow_bk[pad_v[2]:pad_v[2] + img_size[1], pad_v[0]:pad_v[0] + img_size[0], :]
        # pre_1_flow_bk = pre_1_flow_bk[pad_v[2]:pad_v[2] + img_size[1], pad_v[0]:pad_v[0] + img_size[0], :]



        RoI_align_size = 7

        ####################################################################
        '''LOCAL DENSE NOC depths'''
        depth_path = '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/depth_dense/{:0>6}.png'.format(index)
        if os.path.exists(depth_path):
            d = cv.imread(depth_path, -1) / 256.
        else:
            d = cv.imread('/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/depth_dense/000000.png', -1) / 256.

        dst_W, dst_H = img_size
        pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
        pad_wr = dst_W - pad_w - d.shape[1]
        d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')

        cat_all = np.concatenate([img, pre_1, pre_2,
                                  pre_1_flow, pre_2_flow, pre_1_flow_bk, pre_2_flow_bk,
                                  d[..., np.newaxis]
                                  ], 2).astype(np.float32)

        # print('d', datetime.now())


        ####################################################################

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                # img = img.transpose(Image.FLIP_LEFT_RIGHT)
                # d = d.transpose(Image.FLIP_LEFT_RIGHT)
                cat_all = cv.flip(cat_all, 1)
                cat_all[..., [9, 11, 13, 15]] = -cat_all[..., [9, 11, 13, 15]]

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                # crop_size = img_size * np.clip(np.random.randn()*self.scale + 1, 1 - self.scale, 1)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        '''transformed cy'''
        # t_y = center[1] - crop_size[1]/2

        t_x = center[0] - crop_size[0] / 2
        t_y = center[1] - crop_size[1] / 2


        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        cat_all = cv.warpAffine(cat_all, trans, tuple(self.resolution.tolist()))

        cat_all[..., [9, 11, 13, 15]] = cat_all[..., [9, 11, 13, 15]] * (self.resolution[0] / crop_size[0])
        cat_all[..., [10, 12, 14, 16]] = cat_all[..., [10, 12, 14, 16]] * (self.resolution[1] / crop_size[1])


        depth_scale_factor = crop_size[1] / self.resolution[1]
        crop_scale_factor_list = crop_size / self.resolution

        # '''scale lidar depth'''
        # lidar_d = np.array(lidar_d) * depth_scale_factor

        if random_crop_flag and random_flip_flag:
            c = 1

        ####################################################################
        '''LOCAL DENSE NOC depths'''
        down_d = cv.resize(cat_all[..., -1], (self.resolution[0] // self.downsample, self.resolution[1] // self.downsample),
                           interpolation=cv.INTER_AREA)
        ####################################################################


        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        # image encoding
        img = np.transpose((cat_all[..., 0:3].astype(np.float32) / 255.0 - self.mean) / self.std, [2, 0, 1])
        pre_1 = np.transpose((cat_all[..., 3:6].astype(np.float32) / 255.0 - self.mean) / self.std, [2, 0, 1])
        pre_2 = np.transpose((cat_all[..., 6:9].astype(np.float32) / 255.0 - self.mean) / self.std, [2, 0, 1])

        calib = self.get_calib(index)
        features_size = self.resolution // self.downsample  # W * H


        #  ============================   get labels   ==============================
        if self.split != 'test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi
            # labels encoding
            heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32)  # C * H * W
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
            # if torch.__version__ == '1.10.0+cu113':
            if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0']:
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

            ####################################################################
            '''Corner 3d points'''
            # corners_offset_3d = np.zeros((self.max_objs, 8, 2), dtype=np.float32)
            corners_offset_3d = np.zeros((self.max_objs, 16), dtype=np.float32)
            ####################################################################

            '''visibility depth'''
            alpha = np.zeros((self.max_objs, 1), dtype=np.float32)
            # depth_surface = np.zeros((self.max_objs, 4), dtype=np.float32)
            depth_surface = np.zeros((self.max_objs, 2), dtype=np.float32)

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

                bbox_list.append(bbox_2d)

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                     dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample

                ###########################################################################
                '''Corner 3d points'''
                points_3d = self.project_3d(calib.P2, (objects[i].pos)[0], (objects[i].pos)[1] - objects[i].h / 2,
                                            (objects[i].pos)[2],
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
                # heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi
                heading_bin[i], heading_res[i] = angle2class(heading_angle)

                # surface_points = np.array([[objects[i].l / 2,  0, 0],
                #                            [0,                 0, -objects[i].w / 2],
                #                            [-objects[i].l / 2, 0, 0],
                #                            [0,                 0, objects[i].w / 2],
                #                            ])
                # rot_surface_points = self.get_rot_surface_depth(objects[i].ry, surface_points)
                # rot_surface_points = rot_surface_points + objects[i].pos
                # # depth_surface[i] = rot_surface_points[:, 2]
                # depth_surface[i] = (np.sort(rot_surface_points[:, 2]))[:2]

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

                # objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
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
                                      [torch.tensor(bbox_2d).unsqueeze(0)], [RoI_align_size, RoI_align_size]).numpy()[
                    0, 0]
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

        '''transformed cy'''
        # org_cy = calib.P2[1, 2]
        # t_cy = (org_cy - t_y) / depth_scale_factor

        org_fx, org_fy, org_cx, org_cy = calib.P2[0, 0], calib.P2[1, 1], calib.P2[0, 2], calib.P2[1, 2]
        t_fx = org_fx / crop_scale_factor_list[0]
        t_fy = org_fy / crop_scale_factor_list[1]
        t_cx = (org_cx - t_x) / crop_scale_factor_list[0]
        t_cy = (org_cy - t_y) / crop_scale_factor_list[1]
        instr = np.array([t_fx, t_fy, t_cx, t_cy]).astype(np.float32)

        # '''already remap to the original image size domain'''
        # calib.P2[0, 0] = calib.P2[0, 0] * (self.resolution[1] / crop_size[1])

        # collect return data

        inputs = {'img': img,
                  'pre_1': pre_1,
                  'pre_2': pre_2,
                  'pre_1_flow': np.transpose(cat_all[..., [9, 10]], [2, 0, 1]),
                  'pre_2_flow': np.transpose(cat_all[..., [11, 12]], [2, 0, 1]),
                  'pre_1_flow_bk': np.transpose(cat_all[..., [13, 14]], [2, 0, 1]),
                  'pre_2_flow_bk': np.transpose(cat_all[..., [15, 16]], [2, 0, 1]),

                  'instr': instr,

                  # 'lidar_depth': torch.from_numpy(lidar_d).unsqueeze(0).type(torch.float32),

                  }
        # inputs = img

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        b = datetime.now()
        # print(b , b-a)

        return inputs, calib.P2, coord_range, targets, info  # calib.P2



#
# class KITTI_aug(KITTI):
#     def __init__(self, root_dir, split, cfg):
#         super(KITTI_aug, self).__init__(root_dir, split, cfg)
#
#     def get_label(self, idx):
#         label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
#         assert os.path.exists(label_file)
#         return get_objects_from_label(label_file)
#
#     def convert_to_3d(self, depth, P2, upsample_factor, x_start, y_start):
#         '''
#         :param depth: depth map of current frame cropped area. SHAPE: A*B
#         :param P2: projection matrix of left RGB camera.  SHAPE: 4*3
#         :param upsample_factor: upsample factor of the cropped area.
#         :param x_start: start coordinates in image coordinates of x.
#         :param y_start: start coordinates in image coordinates of y.
#         :return:
#                 points: 3D coordinates in real world of cropped area.   SHAPE: N*3
#                 uv_points: corresponding 2D coordinates in image coordinates of 3D points  SHAPE: N*2
#         '''
#         fx = P2[0][0] * upsample_factor
#         fy = P2[1][1] * upsample_factor
#         cx = P2[0][2] * upsample_factor
#         cy = P2[1][2] * upsample_factor
#
#         b_x = P2[0][3] * upsample_factor / (-fx)
#         b_y = P2[1][3] * upsample_factor / (-fy)
#         # print(fx, fy, cx, cy)
#
#         x_tile = np.array(range(depth.shape[1])).reshape(1, -1) + x_start
#         points_x = np.tile(x_tile, [depth.shape[0], 1])
#
#         y_tile = np.array(range(depth.shape[0])).reshape(-1, 1) + y_start
#         points_y = np.tile(y_tile, [1, depth.shape[1]])
#
#         points_x = points_x.reshape((-1, 1))
#         points_y = points_y.reshape((-1, 1))
#         depth = depth.reshape((-1, 1))
#
#         # # -------mask-------
#         # mask = np.where(depth != np.inf, True, False)
#         # points_x = points_x[mask].reshape((-1, 1))
#         # points_y = points_y[mask].reshape((-1, 1))
#         # depth = depth[mask].reshape((-1, 1))
#
#         uv_points = np.concatenate([points_x, points_y], axis=1)
#
#         points_x = (points_x - cx) / fx
#         points_y = (points_y - cy) / fy
#
#         points_x = points_x * depth + b_x
#         points_y = points_y * depth + b_y
#
#         points = np.concatenate([points_x, points_y, depth], axis=1)
#
#         return points, uv_points
#
#
#     def augment_gt(self, source, pad_dense_lidar, p2, org_label, rand_depth):
#
#         # import open3d as o3d
#         # def path2cloud(raw_path, P2):
#         #     raw_xyz = (np.load(raw_path))['velodyne_depth']
#         #     points = self.convert_to_3d(raw_xyz, P2, 1, 0, 0)[0]
#         #     points = points[points[:, 2] > 0]
#         #
#         #     pcd = o3d.geometry.PointCloud()
#         #     pcd.points = o3d.utility.Vector3dVector(points)
#         #     return pcd
#         #
#         # def _rotationMatrixToEulerAngles(R):
#         #     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#         #
#         #     singular = sy < 1e-6
#         #
#         #     if not singular:
#         #         x = math.atan2(R[2, 1], R[2, 2])
#         #         y = math.atan2(-R[2, 0], sy)
#         #         z = math.atan2(R[1, 0], R[0, 0])
#         #     else:
#         #         x = math.atan2(-R[1, 2], R[1, 1])
#         #         y = math.atan2(-R[2, 0], sy)
#         #         z = 0
#         #
#         #     return np.array([x, y, z])
#         #
#         # def icp(raw, target, init_x, init_z, threshold=6.0):
#         #     # threshold = 1.0  # 移动范围的阀值
#         #     trans_init = np.asarray([[1, 0, 0, init_x],  # 4x4 identity matrix，这是一个转换矩阵，
#         #                              [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
#         #                              [0, 0, 1, init_z],  # 这个矩阵为初始变换
#         #                              [0, 0, 0, 1]])
#         #     # 运行icp
#         #     reg_p2p = o3d.registration.registration_icp(
#         #         raw, target, threshold, trans_init,
#         #         o3d.registration.TransformationEstimationPointToPoint())
#         #
#         #     return reg_p2p.transformation
#         #
#         #
#         # raw_path = '/private/personal/pengliang/KITTI_raw/KITTI_raw/2011_09_26/2011_09_26_drive_0036_sync/proj_depth/velodyne/image_02/0000000151.npz'
#         # cur_path = '/private/personal/pengliang/KITTI_raw/KITTI_raw/2011_09_26/2011_09_26_drive_0036_sync/proj_depth/velodyne/image_02/0000000152.npz'
#         # # cur_path = '/private/personal/pengliang/KITTI_raw/KITTI_raw/2011_09_26/2011_09_26_drive_0036_sync/proj_depth/velodyne/image_02/0000000151.npz'
#         # raw_pc = path2cloud(raw_path, p2)
#         # cur_pc = path2cloud(cur_path, p2)
#         #
#         # '''obtain fg points'''
#         # fg_p_ind = np.asarray(cur_pc.points)[:, 1] < 0.5
#         # cur_pc.points = o3d.utility.Vector3dVector(np.asarray(cur_pc.points)[fg_p_ind])
#         # fg_p_ind = np.asarray(raw_pc.points)[:, 1] < 0.5
#         # raw_pc.points = o3d.utility.Vector3dVector(np.asarray(raw_pc.points)[fg_p_ind])
#         #
#         # trans = icp(raw_pc, cur_pc, 0, 0)
#         #
#         # R, T = trans[:3, :3], trans[:3, 3]
#         # rot_x, rot_y, rot_z = _rotationMatrixToEulerAngles(R)
#
#         imH, imW = source.shape[0], source.shape[1]
#
#         # pad_h = imH - dense_lidar.shape[0]
#         # pad_w = (imW - dense_lidar.shape[1]) // 2
#         # pad_dense_lidar = np.zeros((imH, imW))
#         # pad_dense_lidar[pad_h:, pad_w:pad_w + dense_lidar.shape[1]] = dense_lidar
#
#         points, uv_points_org = self.convert_to_3d(pad_dense_lidar, p2, 1, 0, 0)
#
#         points[..., 2] = points[..., 2] + rand_depth
#         # points = np.concatenate([points, np.ones((len(points), 1))], axis=1)
#         # points = ((trans @ points.T).T)[:, :3]
#
#         points_coor = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
#         points_new = (p2 @ points_coor.T).T
#         uv_points = points_new[:, :2] / points_new[:, 2:3]
#
#         # new_rgb = np.zeros((imH, imW, 3), dtype=np.uint8)
#         new_rgb = np.zeros((imH, imW, 4), dtype=np.float32)
#
#         coor_ind = (uv_points[..., 0] >= 0) & \
#                    (uv_points[..., 1] >= 0) & \
#                    (uv_points[..., 0] < imW) & \
#                    (uv_points[..., 1] < imH) & \
#                    (points[..., 2] > 0)
#         new_rgb[uv_points[coor_ind, 1].astype(np.int32), uv_points[coor_ind, 0].astype(np.int32), :3] = \
#             source[uv_points_org[coor_ind, 1], uv_points_org[coor_ind, 0]]
#         '''depth'''
#         new_rgb[uv_points[coor_ind, 1].astype(np.int32), uv_points[coor_ind, 0].astype(np.int32), 3] = \
#             pad_dense_lidar[uv_points_org[coor_ind, 1], uv_points_org[coor_ind, 0]] - rand_depth
#
#         new_rgb_inter = new_rgb.copy()
#         invalid_uv = np.where((new_rgb[..., 0] == 0) & (new_rgb[..., 1] == 0) & (new_rgb[..., 2] == 0))
#         invalid_uv = np.stack([invalid_uv[1], invalid_uv[0]], axis=1)
#
#         invalid_uv_ind = (invalid_uv[:, 0] > 0) & \
#                          (invalid_uv[:, 1] > 0) & \
#                          (invalid_uv[:, 0] < imW - 1) & \
#                          (invalid_uv[:, 1] < imH - 1)
#         invalid_uv = invalid_uv[invalid_uv_ind]
#
#         # rgb_v, mask = [], []
#         # for i in range(-1, 2):
#         #     for j in range(-1, 2):
#         #         tmp_rgb = new_rgb[invalid_uv[..., 1] + i, invalid_uv[..., 0] + j]
#         #         rgb_v.append(tmp_rgb)
#         #         mask.append((tmp_rgb[:, 0] > 0) & (tmp_rgb[:, 1] > 0) & (tmp_rgb[:, 2] > 0))
#
#         rgb_v = [new_rgb[invalid_uv[..., 1] + i, invalid_uv[..., 0] + j]
#                     for i in range(-1, 2) for j in range(-1, 2)]
#         mask = [tmp_rgb[:, 3] > 0 for tmp_rgb in rgb_v]
#
#         new_rgb_inter[invalid_uv[..., 1].astype(np.int32), invalid_uv[..., 0].astype(np.int32)] = \
#             np.sum(rgb_v, axis=0) / np.expand_dims(np.sum(mask, axis=0), axis=1)
#
#
#         new_rgb_inter[new_rgb_inter != new_rgb_inter] = 0
#         new_rgb_inter[new_rgb_inter[..., 0] == 0, :] = np.concatenate([source[new_rgb_inter[..., 0] == 0, :],
#                                                                        np.expand_dims(pad_dense_lidar, axis=2)[new_rgb_inter[..., 0] == 0, :]], axis=1)
#         source_aug = new_rgb_inter[..., :3].astype(np.uint8)
#         pad_dense_lidar = new_rgb_inter[..., 3]
#         pad_dense_lidar[pad_dense_lidar < 0] = 0
#
#         new_objects = []
#         for object in org_label:
#             object.pos[2] = object.pos[2] + rand_depth
#             cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY = object.pos[0], \
#                                                     object.pos[1] - object.h/2, \
#                                                     object.pos[2], \
#                                                     object.w, object.h, object.l, object.ry
#             verts3d, corners_3d = self.project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)
#
#             x = min(verts3d[:, 0])
#             y = min(verts3d[:, 1])
#             x2 = max(verts3d[:, 0])
#             y2 = max(verts3d[:, 1])
#             box2d = np.array((x, y, x2, y2), dtype=np.float32)
#
#             '''constrain'''
#             box2d = np.expand_dims(box2d, axis=0)
#             box2d[box2d < 0] = 0
#             box2d[box2d[:, 0] > imW-1, 0] = imW-1
#             box2d[box2d[:, 1] > imH-1, 1] = imH-1
#             box2d[box2d[:, 2] > imW-1, 2] = imW-1
#             box2d[box2d[:, 3] > imH-1, 3] = imH-1
#             box2d = box2d[0]
#
#             ign_flag = 0
#             # any boxes behind camera plane?
#             if np.any(corners_3d[2, :] <= 0):
#                 ign_flag = 1
#             if (box2d[2] - box2d[0] < 10) or (box2d[3] - box2d[1] < 10):
#                 ign_flag = 1
#             if box2d[2] < 30 or box2d[0] > imW - 30:
#                 ign_flag = 1
#             if object.box2d[2] < 30 or object.box2d[0] > imW - 30:
#                 ign_flag = 1
#
#             if ign_flag:
#                 object.cls_type = 'UnKnown'
#                 box2d = np.array((0, 0, 0, 0), dtype=np.float32)
#
#             object.box2d = box2d
#             new_objects.append(Object3d(object.to_kitti_format()))
#
#         return Image.fromarray(source_aug), \
#                Image.fromarray(pad_dense_lidar), \
#                new_objects
#
#     def project_3d(self, p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
#         """
#         Projects a 3D box into 2D vertices
#
#         Args:
#             p2 (nparray): projection matrix of size 4x3
#             x3d: x-coordinate of center of object
#             y3d: y-coordinate of center of object
#             z3d: z-cordinate of center of object
#             w3d: width of object
#             h3d: height of object
#             l3d: length of object
#             ry3d: rotation w.r.t y-axis
#         """
#
#         # compute rotational matrix around yaw axis
#         R = np.array([[+math.cos(ry3d), 0, +math.sin(ry3d)],
#                       [0, 1, 0],
#                       [-math.sin(ry3d), 0, +math.cos(ry3d)]])
#
#         # 3D bounding box corners
#         x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
#         y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
#         z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])
#
#         x_corners += -l3d / 2
#         y_corners += -h3d / 2
#         z_corners += -w3d / 2
#
#         # bounding box in object co-ordinate
#         corners_3d = np.array([x_corners, y_corners, z_corners])
#
#         # rotate
#         corners_3d = R.dot(corners_3d)
#
#         # translate
#         corners_3d += np.array([x3d, y3d, z3d]).reshape((3, 1))
#
#         corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))
#         corners_2D = p2.dot(corners_3D_1)
#         corners_2D = corners_2D / corners_2D[2]
#
#         bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
#
#         verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).astype(float).T
#
#         if return_3d:
#             return verts3d, corners_3d
#         else:
#             return verts3d
#
#
#     def __getitem__(self, item):
#         #  ============================   get inputs   ===========================
#         index = int(self.idx_list[item])  # index mapping, get real data id
#         # image loading
#
#         # index = 10
#
#         img = self.get_image(index)
#         img_size = np.array(img.size)
#
#         ####################################################################
#         '''LOCAL DENSE NOC depths'''
#         # d = cv.imread('/private/pengliang/KITTI3D/training/lidar_depth/{:0>6}.png'.format(index), -1) / 256.
#         # d = cv.imread('/private/pengliang/depth_dense/{:0>6}.png'.format(index), -1) / 256.
#         # d = cv.imread('/ssd/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(index), -1) / 256.
#         d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(index), -1) / 256.
#
#         dst_W, dst_H = img_size
#         pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
#         pad_wr = dst_W - pad_w - d.shape[1]
#         d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')
#         d = Image.fromarray(d)
#         ####################################################################
#
#
#         ####################################################################
#         '''Augment in depth'''
#         img_np = np.asarray(img)
#         d_np = np.asarray(d)
#         calib = self.get_calib(index)
#         objects = self.get_label(index)
#         p2 = calib.P2
#         if np.random.random() < 0.5:
#             rand_shift_depth = abs(np.random.randint(int(3 * 100))) / 100.
#             # rand_shift_depth = -abs(np.random.randint(int(3 * 100))) / 100.
#             img, d, objects = self.augment_gt(img_np.copy(), d_np.copy(), p2, objects,
#                                               rand_shift_depth)
#         ####################################################################
#
#
#         # data augmentation for image
#         center = np.array(img_size) / 2
#         crop_size = img_size
#         random_crop_flag, random_flip_flag = False, False
#         if self.data_augmentation:
#             if np.random.random() < self.random_flip:
#                 random_flip_flag = True
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)
#                 d = d.transpose(Image.FLIP_LEFT_RIGHT)
#
#             if np.random.random() < self.random_crop:
#                 random_crop_flag = True
#                 crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
#                 center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
#                 center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
#
#         # add affine transformation for 2d images.
#         trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
#         img = img.transform(tuple(self.resolution.tolist()),
#                             method=Image.AFFINE,
#                             data=tuple(trans_inv.reshape(-1).tolist()),
#                             resample=Image.BILINEAR)
#
#         ####################################################################
#         '''LOCAL DENSE NOC depths'''
#         # '''sparse lidar depth'''
#         # d2 = (Image.fromarray(d)).transform(tuple(self.resolution.tolist()),
#         #                     method=Image.AFFINE,
#         #                     data=tuple(trans_inv.reshape(-1).tolist()),
#         #                     resample=Image.NEAREST)
#         # d2 = np.array(d2)
#         # down_d = cv.resize(d2, (self.resolution[0]//self.downsample, self.resolution[1]//self.downsample),
#         #                    interpolation=cv.INTER_NEAREST)
#         '''dense lidar depth'''
#         d2 = d.transform(tuple(self.resolution.tolist()),
#                          method=Image.AFFINE,
#                          data=tuple(trans_inv.reshape(-1).tolist()),
#                          resample=Image.BILINEAR)
#         d2 = np.array(d2)
#         # d2 = cv.warpAffine(d, trans_inv, tuple(self.resolution.tolist()), flags=cv.INTER_LINEAR)
#         down_d = cv.resize(d2, (self.resolution[0] // self.downsample, self.resolution[1] // self.downsample),
#                            interpolation=cv.INTER_AREA)
#         ####################################################################
#
#         coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
#         # image encoding
#         img = np.array(img).astype(np.float32) / 255.0
#         img = (img - self.mean) / self.std
#         img = img.transpose(2, 0, 1)  # C * H * W
#
#         features_size = self.resolution // self.downsample  # W * H
#         #  ============================   get labels   ==============================
#         print(self.split)
#         if self.split != 'test':
#             # data augmentation for labels
#             if random_flip_flag:
#                 calib.flip(img_size)
#                 for object in objects:
#                     [x1, _, x2, _] = object.box2d
#                     object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
#                     object.ry = np.pi - object.ry
#                     object.pos[0] *= -1
#                     if object.ry > np.pi:  object.ry -= 2 * np.pi
#                     if object.ry < -np.pi: object.ry += 2 * np.pi
#             # labels encoding
#             heatmap = np.zeros((self.num_classes, features_size[1], features_size[0]), dtype=np.float32)  # C * H * W
#             size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
#             offset_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
#             depth = np.zeros((self.max_objs, 1), dtype=np.float32)
#             heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
#             heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
#             src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
#             size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
#             offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
#             height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
#             cls_ids = np.zeros((self.max_objs), dtype=np.int64)
#             indices = np.zeros((self.max_objs), dtype=np.int64)
#             # if torch.__version__ == '1.10.0+cu113':
#             if torch.__version__ in ['1.10.0+cu113', '1.10.0', '1.6.0']:
#                 mask_2d = np.zeros((self.max_objs), dtype=np.bool)
#             else:
#                 mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
#             # '''for 30 cards'''
#             # mask_2d = np.zeros((self.max_objs), dtype=np.bool)
#             mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
#             object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
#
#             ####################################################################
#             '''LOCAL DENSE NOC depths'''
#             abs_noc_depth = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
#             noc_depth_offset = np.zeros((self.max_objs, 7, 7), dtype=np.float32)
#             noc_depth_mask = np.zeros((self.max_objs, 7, 7), dtype=np.bool)
#             bbox_list = []
#             ####################################################################
#
#             '''visibility depth'''
#             alpha = np.zeros((self.max_objs, 1), dtype=np.float32)
#             # depth_surface = np.zeros((self.max_objs, 4), dtype=np.float32)
#             depth_surface = np.zeros((self.max_objs, 2), dtype=np.float32)
#
#             for i in range(object_num):
#                 # filter objects by writelist
#                 if objects[i].cls_type not in self.writelist:
#                     continue
#
#                 # filter inappropriate samples by difficulty
#                 if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
#                     continue
#
#                 # process 2d bbox & get 2d center
#                 bbox_2d = objects[i].box2d.copy()
#                 # add affine transformation for 2d boxes.
#                 bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
#                 bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
#                 # modify the 2d bbox according to pre-compute downsample ratio
#                 bbox_2d[:] /= self.downsample
#
#                 bbox_list.append(bbox_2d)
#
#                 # process 3d bbox & get 3d center
#                 center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
#                                      dtype=np.float32)  # W * H
#                 center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
#                 center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
#                 center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
#                 center_3d = center_3d[0]  # shape adjustment
#                 center_3d = affine_transform(center_3d.reshape(-1), trans)
#                 center_3d /= self.downsample
#
#
#                 # ###########################################################################
#                 # '''Corner 3d points'''
#                 # corners_3d = self.project_3d(p2, (objects[i].pos)[0], (objects[i].pos)[1] - -objects[i].h / 2, (objects[i].pos)[2],
#                 #                              objects[i].w, objects[i].h, objects[i].l, objects[i].ry)
#                 # stack_corners = []
#                 # for ii in range(8):
#                 #     stack_corners.append(affine_transform(corners_3d[ii].reshape(-1), trans))
#                 #     stack_corners[ii] /= self.downsample
#                 # stack_corners = np.stack(stack_corners, axis=0)
#                 # print('center_3d: {}, stack_corners: {}'.format(center_3d, stack_corners))
#                 # ###########################################################################
#
#
#                 # generate the center of gaussian heatmap [optional: 3d center or 2d center]
#                 center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
#                 if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: continue
#                 if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: continue
#
#                 # generate the radius of gaussian heatmap
#                 w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
#                 radius = gaussian_radius((w, h))
#                 radius = max(0, int(radius))
#
#                 if objects[i].cls_type in ['Van', 'Truck', 'DontCare']:
#                     draw_umich_gaussian(heatmap[1], center_heatmap, radius)
#                     continue
#
#                 cls_id = self.cls2id[objects[i].cls_type]
#                 cls_ids[i] = cls_id
#                 draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)
#
#                 # encoding 2d/3d offset & 2d size
#                 indices[i] = center_heatmap[1] * features_size[0] + center_heatmap[0]
#                 offset_2d[i] = center_2d - center_heatmap
#                 size_2d[i] = 1. * w, 1. * h
#
#                 # encoding depth
#                 depth[i] = objects[i].pos[-1]
#
#                 # encoding heading angle
#                 # heading_angle = objects[i].alpha
#                 heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
#                 if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
#                 if heading_angle < -np.pi: heading_angle += 2 * np.pi
#                 heading_bin[i], heading_res[i] = angle2class(heading_angle)
#
#                 # surface_points = np.array([[objects[i].l / 2,  0, 0],
#                 #                            [0,                 0, -objects[i].w / 2],
#                 #                            [-objects[i].l / 2, 0, 0],
#                 #                            [0,                 0, objects[i].w / 2],
#                 #                            ])
#                 # rot_surface_points = self.get_rot_surface_depth(objects[i].ry, surface_points)
#                 # rot_surface_points = rot_surface_points + objects[i].pos
#                 # # depth_surface[i] = rot_surface_points[:, 2]
#                 # depth_surface[i] = (np.sort(rot_surface_points[:, 2]))[:2]
#
#                 # encoding 3d offset & size_3d
#                 offset_3d[i] = center_3d - center_heatmap
#                 src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
#                 mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
#                 size_3d[i] = src_size_3d[i] - mean_size
#
#                 # objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
#                 if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
#                     mask_2d[i] = 1
#
#                 ####################################################################
#                 '''LOCAL DENSE NOC depths'''
#                 down_d_copy = down_d.copy()
#                 # bbox_2d_int = bbox_2d.astype(np.int32)
#                 # bbox_2d_int[bbox_2d_int < 0] = 0
#                 # roi_depth = down_d_copy[bbox_2d_int[1]:bbox_2d_int[3] + 1, bbox_2d_int[0]:bbox_2d_int[2] + 1]
#                 # roi_depth_ind = (roi_depth < depth[i] - 3) | (roi_depth > depth[i] + 3)
#                 # down_d_copy[bbox_2d_int[1]:bbox_2d_int[3] + 1, bbox_2d_int[0]:bbox_2d_int[2] + 1][roi_depth_ind] = 0
#                 #
#                 # '''for dense lidar depth can only use INTER_NEAREST mode !!!!!!'''
#                 # abs_noc_depth[i] = cv.resize(down_d_copy[bbox_2d_int[1]:bbox_2d_int[3]+1, bbox_2d_int[0]:bbox_2d_int[2]+1],
#                 #                           (7, 7), interpolation=cv.INTER_NEAREST)
#                 # noc_depth_mask[i] = abs_noc_depth[i] > 0
#                 # noc_depth_offset[i] = depth[i] - abs_noc_depth[i]
#
#                 roi_depth = roi_align(torch.from_numpy(down_d_copy).unsqueeze(0).unsqueeze(0).type(torch.float32),
#                                       [torch.tensor(bbox_2d).unsqueeze(0)], [7, 7]).numpy()[0, 0]
#                 roi_depth_ind = (roi_depth > depth[i] - 3) & \
#                                 (roi_depth < depth[i] + 3) & \
#                                 (roi_depth > 0)
#                 roi_depth[~roi_depth_ind] = 0
#                 abs_noc_depth[i] = roi_depth
#                 noc_depth_mask[i] = roi_depth_ind
#                 noc_depth_offset[i] = depth[i] - abs_noc_depth[i]
#
#                 ####################################################################
#
#             # '''cannot fit sparse lidar depth'''
#             # roi_noc_depth = roi_align(torch.from_numpy(down_d).unsqueeze(0).unsqueeze(0).type(torch.float32), [torch.tensor(np.array(bbox_list))], [7, 7])
#
#             targets = {
#                 'depth': depth,
#                 'size_2d': size_2d,
#                 'heatmap': heatmap,
#                 'offset_2d': offset_2d,
#                 'indices': indices,
#                 'size_3d': size_3d,
#                 'offset_3d': offset_3d,
#                 'heading_bin': heading_bin,
#                 'heading_res': heading_res,
#                 'cls_ids': cls_ids,
#                 'mask_2d': mask_2d,
#
#                 'abs_noc_depth': abs_noc_depth,
#                 'noc_depth_mask': noc_depth_mask,
#                 'noc_depth_offset': noc_depth_offset,
#
#                 # 'depth_surface': depth_surface,
#             }
#         else:
#             targets = {}
#
#         # '''already remap to the original image size domain'''
#         # calib.P2[0, 0] = calib.P2[0, 0] * (self.resolution[1] / crop_size[1])
#
#         # collect return data
#         inputs = img
#         info = {'img_id': index,
#                 'img_size': img_size,
#                 'bbox_downsample_ratio': img_size / features_size}
#         return inputs, calib.P2, coord_range, targets, info  # calib.P2


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    cfg = {'random_flip': 0.0, 'random_crop': 1.0, 'scale': 0.4, 'shift': 0.1, 'use_dontcare': False,
           'class_merging': False, 'writelist': ['Pedestrian', 'Car', 'Cyclist'], 'use_3d_center': False}
    dataset = KITTI('../../data', 'train', cfg)
    dataloader = DataLoader(dataset=dataset, batch_size=1)
    print(dataset.writelist)

    for batch_idx, (inputs, targets, info) in enumerate(dataloader):
        # test image
        img = inputs[0].numpy().transpose(1, 2, 0)
        img = (img * dataset.std + dataset.mean) * 255
        img = Image.fromarray(img.astype(np.uint8))
        img.show()
        # print(targets['size_3d'][0][0])

        # test heatmap
        heatmap = targets['heatmap'][0]  # image id
        heatmap = Image.fromarray(heatmap[0].numpy() * 255)  # cats id
        heatmap.show()

        break

    # print ground truth fisrt
    objects = dataset.get_label(0)
    for object in objects:
        print(object.to_kitti_format())
