import os
import numpy as np
import cv2 as cv
from tqdm import tqdm

all_f_path = '/data/wuxiaopei/DataSets/gpu11_pcdet_data/waymo_sfd_seguv/training/depth_dense_scale_equal/'
dst_path = '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/depth_dense'

sample_f_path = '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2'
all_f = [os.path.join(all_f_path, i) for i in sorted(os.listdir(sample_f_path))]

for f in tqdm(all_f):
    empty_depth = np.zeros((1280//2, 1920//2))
    dense_path = cv.imread(f, -1) / 256.
    empty_depth[210:210+320, :] = dense_path
    empty_depth = cv.resize(empty_depth, (1920, 1280), interpolation=cv.INTER_NEAREST)
    cv.imwrite(os.path.join(dst_path, f.split('/')[-1]), (empty_depth*256).astype(np.uint16))
    # exit(0)