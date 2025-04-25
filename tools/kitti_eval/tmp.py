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
from copy import deepcopy


# from lib.helpers.decode_helper import calc_iou

class KITTI(data.Dataset):
    def __init__(self, root_dir, split, cfg):
        # basic configuration
        self.num_classes = 3
        self.max_objs = 250
        self.class_name = ['Pedestrian', 'Car', 'Cyclist']
        self.cls2id = {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
        # self.resolution = np.array([1280, 384])  # W * H
        self.resolution = np.array([1920, 1280])  # W * H
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
        # self.cls_mean_size = np.array([[1.76255119    ,0.66068622   , 0.84422524   ], # Pedestrian
        #                                [1.52563191462 ,1.62856739989, 3.88311640418], # car
        #                                [1.73698127    ,0.59706367   , 1.76282397   ]])  # Cyclist
        # self.cls_mean_size = np.array([[1.7490246045546842    ,0.853621465972869  , 0.9080261886739016   ],
        #                                [1.7920510000005416, 2.102084234827723, 4.798515546143519],
        #                                [1.768972990996961    ,0.833707902634183   , 1.7669195287317714   ]]) # waymo
        self.cls_mean_size = np.array([[1.74902460, 0.85362147, 0.90802619],
                                       [1.79205100, 2.10208423, 4.79851555],
                                       [1.76897299, 0.83370790, 1.76691953]])
        # data split loading
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(root_dir, 'ImageSets', split + '.txt')
        # self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        # self.data_dir = os.path.join(root_dir, 'object', 'testing' if split == 'test' else 'training')
        # self.image_dir = os.path.join(self.data_dir, 'image_2')
        # self.depth_dir = os.path.join(self.data_dir, 'depth')
        # self.calib_dir = os.path.join(self.data_dir, 'calib')
        # self.label_dir = os.path.join(self.data_dir, 'label_2')

        self.data_dir = os.path.join(root_dir, 'validation' if split == 'val' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_0')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'filter_label_0')
        split_dir = os.path.join(root_dir, split + '_tiny.txt')
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

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
        # if not os.path.exists(img_file):
        #     print(img_file)
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

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)

        img_size = np.array(img.size)

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        if self.data_augmentation:
            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                crop_size = img_size * np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = img[..., :3]
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

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
            # depth_conf = np.zeros((self.max_objs, 1), dtype=np.float32)
            heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
            heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
            src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
            offset_3d = np.zeros((self.max_objs, 2), dtype=np.float32)
            height2d = np.zeros((self.max_objs, 1), dtype=np.float32)
            cls_ids = np.zeros((self.max_objs), dtype=np.int64)
            indices = np.zeros((self.max_objs), dtype=np.int64)
            # mask_2d = np.zeros((self.max_objs), dtype=np.uint8)
            # mask_3d = np.zeros((self.max_objs), dtype=np.uint8)
            mask_2d = np.zeros((self.max_objs), dtype=np.bool)
            mask_3d = np.zeros((self.max_objs), dtype=np.bool)
            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs
            idx = 0
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown':
                    continue
                if objects[i].pos[-1] < 2:
                    continue

                # OBMO
                obj = objects[i]

                # x_z_ratio = obj.pos[0] / obj.pos[2]
                # y_z_ratio = (obj.pos[1] - 0.5 * obj.h) / obj.pos[2]

                # #epsilon = [-3.0, -2.4, -1.8, -1.2, -0.6, 0.6, 1.2, 1.8, 2.4, 3.0]
                # epsilon = [-0.04*obj.pos[2], -0.02*obj.pos[2], -0.01*obj.pos[2], 0.01*obj.pos[2], 0.02*obj.pos[2], 0.04*obj.pos[2]]
                # center_2d = np.array([(obj.box2d[0] + obj.box2d[2]) / 2, (obj.box2d[1] + obj.box2d[3]) / 2], dtype=np.float32)  # W * H
                # center_3d = obj.pos + [0, -obj.h / 2, 0]  # real 3D center in 3D space
                # center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                # pts_2d_ori, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                # centerx_off = pts_2d_ori[0][0] - center_2d[0]
                # centery_off = pts_2d_ori[0][1] - center_2d[1]
                # bbox_h = obj.box2d[3] - obj.box2d[1]
                # bbox_w = obj.box2d[2] - obj.box2d[0]
                # ori_bbox = np.array([obj.box2d[0], obj.box2d[1], obj.box2d[2], obj.box2d[3]])
                # obj.score = 1.0

                def record(obj, index):
                    # process 2d bbox & get 2d center
                    bbox_2d = obj.box2d.copy()
                    # add affine transformation for 2d boxes.
                    bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)
                    # modify the 2d bbox according to pre-compute downsample ratio
                    bbox_2d[:] /= self.downsample

                    # process 3d bbox & get 3d center
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                         dtype=np.float32)  # W * H
                    center_3d = obj.pos + [0, -obj.h / 2, 0]  # real 3D center in 3D space
                    center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    center_3d = affine_transform(center_3d.reshape(-1), trans)
                    center_3d /= self.downsample

                    # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                    center_heatmap = center_3d.astype(np.int32) if self.use_3d_center else center_2d.astype(np.int32)
                    if center_heatmap[0] < 0 or center_heatmap[0] >= features_size[0]: return
                    if center_heatmap[1] < 0 or center_heatmap[1] >= features_size[1]: return

                    # generate the radius of gaussian heatmap
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    radius = gaussian_radius((w, h))
                    radius = max(0, int(radius))

                    if obj.cls_type in ['Van', 'Truck', 'DontCare']:
                        draw_umich_gaussian(heatmap[1], center_heatmap, radius)
                        return

                    cls_id = self.cls2id[obj.cls_type]
                    cls_ids[index] = cls_id
                    draw_umich_gaussian(heatmap[cls_id], center_heatmap, radius)

                    # encoding 2d/3d offset & 2d size
                    indices[index] = center_heatmap[1] * features_size[0] + center_heatmap[0]
                    offset_2d[index] = center_2d - center_heatmap
                    size_2d[index] = 1. * w, 1. * h

                    # encoding depth
                    depth[index] = obj.pos[-1]
                    # depth_conf[index] = obj.score

                    # encoding heading angle
                    # heading_angle = objects[i].alpha
                    heading_angle = calib.ry2alpha(obj.ry, (obj.box2d[0] + obj.box2d[2]) / 2)
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin[index], heading_res[index] = angle2class(heading_angle)

                    # encoding 3d offset & size_3d
                    offset_3d[index] = center_3d - center_heatmap
                    src_size_3d[index] = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
                    mean_size = self.cls_mean_size[self.cls2id[obj.cls_type]]
                    size_3d[index] = src_size_3d[index] - mean_size

                    # objects[i].trucation <=0.5 and objects[i].occlusion<=2 and (objects[i].box2d[3]-objects[i].box2d[1])>=25:
                    if obj.trucation <= 0.5 and obj.occlusion <= 2:
                        mask_2d[index] = 1

                record(obj, idx)
                idx = idx + 1
                # for e in epsilon:
                #     new_obj = deepcopy(obj)
                #     new_obj.pos[2] = obj.pos[2] + e
                #     new_obj.pos[0] = new_obj.pos[2] * x_z_ratio
                #     new_obj.pos[1] = new_obj.pos[2] * y_z_ratio + 0.5 * obj.h

                #     pts_2d, _ = calib.rect_to_img(np.array([[new_obj.pos[0], new_obj.pos[1] - 0.5 * obj.h, new_obj.pos[2]]]))
                #     new_obj.box2d[0] = pts_2d[0][0] - centerx_off - bbox_w / 2
                #     new_obj.box2d[2] = pts_2d[0][0] - centerx_off + bbox_w / 2
                #     new_obj.box2d[1] = pts_2d[0][1] - centery_off - bbox_h / 2
                #     new_obj.box2d[3] = pts_2d[0][1] - centery_off + bbox_h / 2

                #     # new_bbox = np.array([new_obj.box2d[0], new_obj.box2d[1], new_obj.box2d[2], new_obj.box2d[3]])
                #     # new_obj.score = calc_iou(ori_bbox, new_bbox)
                #     # new_obj.score = 1 - np.abs(e) / 4
                #     if new_obj.score<=0:
                #         continue

                #     record(new_obj, idx)
                #     idx = idx + 1

            targets = {'depth': depth,
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
                       #    'depth_conf': depth_conf,
                       }
        else:
            targets = {}
        # collect return data
        inputs = img
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
        # img = Image.fromarray(img.astype(np.uint8))
        img = Image.fromarray(img.astype(np.bool))
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
