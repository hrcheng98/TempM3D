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

from datetime import datetime



tmp_p = '/pvc_data/personal/pengliang'
train_3D_mapping_file_path = tmp_p + '/WeakM3D_official/data/kitti/data_file/train_mapping.txt'
kitti_3D_rand_file_path = tmp_p + '/WeakM3D_official/data/kitti/data_file/train_rand.txt'
train_3D_file_path = tmp_p + '/KITTI3D/train.txt'
val_3D_file_path = tmp_p + '/KITTI3D/val.txt'
kitti_raw_data_dir = '/pvc_data/personal/pengliang/kitti_raw'


def build_train_val_set():
    train_mapping = np.loadtxt(train_3D_mapping_file_path, dtype=str)
    kitti_rand = np.loadtxt(kitti_3D_rand_file_path, delimiter=',')
    train_3D = np.loadtxt(train_3D_file_path).astype(np.uint16)
    val_3D = np.loadtxt(val_3D_file_path).astype(np.uint16)

    train_3D_mapping = train_mapping[(kitti_rand[train_3D] - 1).astype(np.uint16)]
    val_3D_mapping = train_mapping[(kitti_rand[val_3D] - 1).astype(np.uint16)]

    mapping = []
    for i in range(7481):
        if len(train_3D) > 0 and i == train_3D[0]:
            mapping.append(train_3D_mapping[0])
            train_3D_mapping = train_3D_mapping[1:]
            train_3D = train_3D[1:]
        elif len(val_3D) > 0 and i == val_3D[0]:
            mapping.append(val_3D_mapping[0])
            val_3D_mapping = val_3D_mapping[1:]
            val_3D = val_3D[1:]
        else:
            print('error')
    return mapping


class KITTIFlow(data.Dataset):
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
        self.cls_mean_size = np.array([[1.76255119, 0.66068622, 0.84422524],
                                       [1.52563191462, 1.62856739989, 3.88311640418],
                                       [1.73698127, 0.59706367, 1.76282397]])

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
        # self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.mean = np.array([0.406, 0.485, 0.456], dtype=np.float32)
        self.std = np.array([0.225, 0.229, 0.224], dtype=np.float32)

        # others
        self.downsample = 4

        self.mapping = build_train_val_set()
        self.pre_frames = cfg['pre_frames']

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

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
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img_file = os.path.join(self.image_dir, '%06d.png' % index)

        '''
        obtain prior image
        '''
        pre_frame_img_path = []

        if self.split != 'test':
            seq = self.mapping[index]
            # pre_frame_img_path.append(os.path.join(kitti_raw_data_dir, seq[0], seq[1], 'image_02/data/{:0>10}.png'.format(int(seq[2]))))
            for j in range(self.pre_frames+1):
                path = os.path.join(kitti_raw_data_dir, seq[0], seq[1], 'image_02/data/{:0>10}.png'.format(int(seq[2]) - j))
                pre_frame_img_path.append(path)
        else:
            pre_frame_img_path.append(img_file)
            for j in range(1, self.pre_frames+1):
                path = os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_3d_pre/data_object_prev_2/testing/prev_2', '{:0>6}_{:0>2}.png'.format(index, j))
                pre_frame_img_path.append(path)


        for j in range(1, self.pre_frames + 1):
            if not os.path.exists(pre_frame_img_path[j]):
                pre_frame_img_path[j] = pre_frame_img_path[j-1]


        pre_frame_imgs = []
        for j in range(self.pre_frames + 1):
            rgb_img = cv.imread(pre_frame_img_path[j])
            pre_frame_imgs.append(rgb_img)
        img_size = np.array([pre_frame_imgs[0].shape[1], pre_frame_imgs[0].shape[0]])


        pad_ht = (((img_size[1] // 8) + 1) * 8 - img_size[1]) % 8
        pad_wd = (((img_size[0] // 8) + 1) * 8 - img_size[0]) % 8
        pad_v = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]

        pre_frame_img_flow_path = [p.replace('kitti_raw', 'kitti_raw_flow') for p in pre_frame_img_path[:-1]]
        pre_frame_img_flow, pre_frame_img_flow_bk = [], []
        for j in range(self.pre_frames):
            if os.path.exists(pre_frame_img_flow_path[j].replace('.png', '_u.png')):
                flow_u = cv.imread(pre_frame_img_flow_path[j].replace('.png', '_u.png'), -1) / 10. - 1000.
                flow_v = cv.imread(pre_frame_img_flow_path[j].replace('.png', '_v.png'), -1) / 10. - 1000.
                flow_u_bk = cv.imread(pre_frame_img_flow_path[j].replace('.png', '_bk_u.png'), -1) / 10. - 1000.
                flow_v_bk = cv.imread(pre_frame_img_flow_path[j].replace('.png', '_bk_v.png'), -1) / 10. - 1000.

                flow = np.stack([flow_u, flow_v], axis=2)
                flow_bk = np.stack([flow_u_bk, flow_v_bk], axis=2)

                flow = flow[pad_v[2]:pad_v[2] + img_size[1], pad_v[0]:pad_v[0] + img_size[0], :]
                flow_bk = flow_bk[pad_v[2]:pad_v[2] + img_size[1], pad_v[0]:pad_v[0] + img_size[0], :]
            else:
                flow = np.zeros_like(pre_frame_img_flow[-1])
                flow_bk = np.zeros_like(pre_frame_img_flow_bk[-1])

            pre_frame_img_flow.append(flow)
            pre_frame_img_flow_bk.append(flow_bk)
        ####################################################################

        RoI_align_size = 7

        ####################################################################
        '''LOCAL DENSE NOC depths'''
        if index > 7480:
            '''for test'''
            d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(7480), -1) / 256.
        else:
            d = cv.imread('/pvc_data/personal/pengliang/KITTI3D/depth_dense/{:0>6}.png'.format(index), -1) / 256.


        dst_W, dst_H = img_size
        pad_h, pad_w = dst_H - d.shape[0], (dst_W - d.shape[1]) // 2
        pad_wr = dst_W - pad_w - d.shape[1]
        d = np.pad(d, ((pad_h, 0), (pad_w, pad_wr)), mode='edge')


        cat_all = np.concatenate([*pre_frame_imgs,
                                  *pre_frame_img_flow,
                                  *pre_frame_img_flow_bk,
                                  d[..., np.newaxis]
                                  ], 2).astype(np.float32)
        flow_u_index = [(self.pre_frames+1)*3 + j*2 for j in range(self.pre_frames*2)]
        flow_v_index = [(self.pre_frames+1)*3 + j*2+1 for j in range(self.pre_frames*2)]


        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False

        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                cat_all = cv.flip(cat_all, 1)
                cat_all[..., flow_u_index] = -cat_all[..., flow_u_index]

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)


        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        cat_all = cv.warpAffine(cat_all, trans, tuple(self.resolution.tolist()))

        cat_all[..., flow_u_index] = cat_all[..., flow_u_index] * (self.resolution[0] / crop_size[0])
        cat_all[..., flow_v_index] = cat_all[..., flow_v_index] * (self.resolution[1] / crop_size[1])


        ####################################################################
        '''LOCAL DENSE NOC depths'''
        down_d = cv.resize(cat_all[..., -1], (self.resolution[0] // self.downsample, self.resolution[1] // self.downsample),
                           interpolation=cv.INTER_AREA)
        ####################################################################


        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        # image encoding
        rgb_imgs = []
        for j in range(self.pre_frames+1):
            trans_rgb_img = np.transpose((cat_all[..., j*3:(j+1)*3].astype(np.float32) / 255.0 - self.mean) / self.std, [2, 0, 1])
            rgb_imgs.append(trans_rgb_img)
        rgb_imgs = np.stack(rgb_imgs, axis=0)

        flow_imgs = []
        for j in range(self.pre_frames*2):
            trans_flow_img = np.transpose(cat_all[..., (self.pre_frames+1)*3+j*2: (self.pre_frames+1)*3+(j+1)*2], [2, 0, 1])
            flow_imgs.append(trans_flow_img)
        flow_imgs = np.stack(flow_imgs, axis=0)


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
            abs_noc_depth = np.zeros((self.max_objs, RoI_align_size, RoI_align_size), dtype=np.float32)
            noc_depth_offset = np.zeros((self.max_objs, RoI_align_size, RoI_align_size), dtype=np.float32)
            noc_depth_mask = np.zeros((self.max_objs, RoI_align_size, RoI_align_size), dtype=np.bool)
            ####################################################################
            #
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                    continue

                '''remove far'''
                if objects[i].pos[-1] > 60:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                # modify the 2d bbox according to pre-compute downsample ratio
                bbox_2d[:] /= self.downsample

                # bbox_list.append(bbox_2d)

                # process 3d bbox & get 3d center
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                     dtype=np.float32)  # W * H
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                center_3d /= self.downsample


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


                offset_3d[i] = center_3d - center_heatmap
                src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]
                size_3d[i] = src_size_3d[i] - mean_size


                # objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                    mask_2d[i] = 1

                ####################################################################
                '''LOCAL DENSE NOC depths'''
                down_d_copy = down_d.copy()
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
            }
        else:
            targets = {}


        inputs = {'rgb_imgs': rgb_imgs,
                  'flow_imgs': flow_imgs,
                  }
        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        return inputs, calib.P2, coord_range, targets, info  # calib.P2



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
