from tqdm import tqdm
import numpy as np
import os
import cv2 as cv

# split = ['training', 7481]
# split = ['testing', 7512]
split = ['testing', 7518]

for i in tqdm(range(split[1])):
    pre_2_flow = np.load(
        os.path.join('/pvc_data/personal/pengliang/private_data/RAFT/KITTI3D_flow/{}'.format(split[0]),
                     '{:0>6}.npy'.format(i * 2 + 0)))
    pre_1_flow = np.load(
        os.path.join('/pvc_data/personal/pengliang/private_data/RAFT/KITTI3D_flow/{}'.format(split[0]),
                     '{:0>6}.npy'.format(i * 2 + 1)))

    assert np.min(pre_2_flow) > -1000.
    assert np.min(pre_1_flow) > -1000.

    cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all/{}'.format(split[0]),
                         '{:0>6}_u.png'.format(i * 2 + 0)), ((pre_2_flow[..., 0]+1000.)*10).astype(np.uint16))
    cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all/{}'.format(split[0]),
                         '{:0>6}_u.png'.format(i * 2 + 1)), ((pre_1_flow[..., 0]+1000.)*10).astype(np.uint16))
    cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all/{}'.format(split[0]),
                         '{:0>6}_v.png'.format(i * 2 + 0)), ((pre_2_flow[..., 1]+1000.)*10).astype(np.uint16))
    cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all/{}'.format(split[0]),
                         '{:0>6}_v.png'.format(i * 2 + 1)), ((pre_1_flow[..., 1]+1000.)*10).astype(np.uint16))


# for i in tqdm(range(split[1])):
#     pre_2_flow = np.load(
#         os.path.join('/pvc_data/personal/pengliang/private_data/RAFT/KITTI3D_flow_bk/{}'.format(split[0]),
#                      '{:0>6}.npy'.format(i * 2 + 0)))
#     pre_1_flow = np.load(
#         os.path.join('/pvc_data/personal/pengliang/private_data/RAFT/KITTI3D_flow_bk/{}'.format(split[0]),
#                      '{:0>6}.npy'.format(i * 2 + 1)))
#
#     assert np.min(pre_2_flow) > -1000.
#     assert np.min(pre_1_flow) > -1000.
#
#     cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all_bk/{}'.format(split[0]),
#                          '{:0>6}_u.png'.format(i * 2 + 0)), ((pre_2_flow[..., 0]+1000.)*10).astype(np.uint16))
#     cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all_bk/{}'.format(split[0]),
#                          '{:0>6}_u.png'.format(i * 2 + 1)), ((pre_1_flow[..., 0]+1000.)*10).astype(np.uint16))
#     cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all_bk/{}'.format(split[0]),
#                          '{:0>6}_v.png'.format(i * 2 + 0)), ((pre_2_flow[..., 1]+1000.)*10).astype(np.uint16))
#     cv.imwrite(os.path.join('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all_bk/{}'.format(split[0]),
#                          '{:0>6}_v.png'.format(i * 2 + 1)), ((pre_1_flow[..., 1]+1000.)*10).astype(np.uint16))
exit(0)