import re
import io
import os
import argparse
import cv2
import glob
import numpy as np
import torch
from nets.dip import DIP

from nets.utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm

from datetime import datetime
import torch.nn.functional as F


DEVICE = 'cuda'


def flow_to_image_ndmax(flow, max_flow=256, isBGR2RGB=True):
    # flow shape (H, W, C)
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    im_rgb = (im * 255).astype(np.uint8)
    if isBGR2RGB:
        im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
    return im_rgb


def warp_cv2(img_prev, flow):
    # calculate mat
    w = int(img_prev.shape[1])
    h = int(img_prev.shape[0])
    flow = np.float32(flow)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flow
    print('pixel_map', pixel_map.shape)
    new_frame = cv2.remap(img_prev, pixel_map, None, cv2.INTER_LINEAR)
    return new_frame


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def demo_kittiTestTrain():
    model = DIP(max_offset=256, mixed_precision=False, test_mode=True)
    model = torch.nn.DataParallel(model)
    model.cuda()
    warm_start = True

    pre_train = torch.load('/pvc_user/pengliang/DIP/DIP_kitti.pth')
    model.load_state_dict(pre_train, strict=False)

    model.eval()


    #split = ['testing', 'testing_DIP_flow', 7518]
    split = ['training', 'training_DIP_flow_v2', 7481]
    kitti_test_dir = '/pvc_data/personal/pengliang/KITTI3D/{}/image_2'.format(split[0])
    # kitti_test_pre_data_dir = '/pvc_data/personal/pengliang/KITTI3D/kitti_3d_pre/data_object_prev_2/{}/prev_2'.format(split[0])
    kitti_test_pre_data_dir = '/pvc_user/chenghaoran/tempM3D/DID/KITTI_pvc/kitti_3d_pre/data_object_prev_2/{}/prev_2'.format(split[0])
    
    # kitti_test_pre_data_dir = '/pvc_data/personal/pengliang/KITTI3D/kitti_3d_pre_DIP/data_object_prev_2/{}/prev_2'.format(
    #     split[0])
    if not os.path.exists(kitti_test_pre_data_dir):
        os.makedirs(kitti_test_pre_data_dir)



    with torch.no_grad():
        for idx in tqdm(range(split[2])):
            str_idx = '{:0>6}'.format(idx)
            # flow_id = ['00', '01', '02', '03']
            flow_id = ['04', '05']
            for j, _ in enumerate(flow_id[:-1]):
                if not os.path.exists(os.path.join(kitti_test_pre_data_dir, str_idx + '_' + flow_id[j + 1] + '.png')):
                    continue

                im_name1 = os.path.join(kitti_test_pre_data_dir, str_idx + '_' + flow_id[j + 1] + '.png')
                if flow_id[j] == '00':
                    im_name2 = os.path.join(kitti_test_dir, str_idx + '.png')
                else:
                    im_name2 = os.path.join(kitti_test_pre_data_dir, str_idx + '_' + flow_id[j] + '.png')

                img1 = cv2.imread(im_name1, cv2.IMREAD_COLOR)
                img2 = cv2.imread(im_name2, cv2.IMREAD_COLOR)
                # '''resize'''
                img1 = cv2.resize(img1, (img1.shape[1]//4, img1.shape[0]//4))
                img2 = cv2.resize(img2, (img2.shape[1]//4, img2.shape[0]//4))


                image1 = torch.from_numpy(img1).permute(2, 0, 1).float()
                image2 = torch.from_numpy(img2).permute(2, 0, 1).float()
                image1 = image1[None].to(DEVICE)
                image2 = image2[None].to(DEVICE)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                torch.cuda.synchronize()
                t1 = datetime.now()
                flow_up = model(image1 = image1, image2 = image2, iters=20, init_flow=None)
                t2 = datetime.now()
                # print(t2, t2-t1)
                # flow_up = padder.unpad(flow_up)
                # flo = flow_up[0].view(2, flow_up[0].shape[-2], flow_up[0].shape[-1])
                # flo = flo.permute(1, 2, 0).cpu().numpy()
                # color_flow = flow_to_image(flo, clip_flow=None, convert_to_bgr=True)
                # cv2.imwrite('tmp_flow.jpg', color_flow)
                # exit(0)



                flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy() * 4.

                torch.cuda.synchronize()
                t1 = datetime.now()
                flow_up = model(image1 = image2, image2 = image1, iters=20, init_flow=None)
                t2 = datetime.now()
                # print(t2, t2-t1)

                flow_uv_bk = flow_up[0].permute(1, 2, 0).cpu().numpy() * 4.

                assert np.min(flow_uv) > -1000.
                assert np.min(flow_uv_bk) > -1000.
                # assert np.min(flow_uv) > -500.
                # assert np.min(flow_uv_bk) > -500.

                if flow_id[j] == '00':
                    im_name2 = os.path.join(kitti_test_pre_data_dir, str_idx + '_00.png')

                save_name_u = im_name2.replace(split[0], split[1]).replace('.png', '_u.png')
                save_name_v = im_name2.replace(split[0], split[1]).replace('.png', '_v.png')
                save_name_u_bk = im_name2.replace(split[0], split[1]).replace('.png', '_bk_u.png')
                save_name_v_bk = im_name2.replace(split[0], split[1]).replace('.png', '_bk_v.png')

                cv2.imwrite(save_name_u, ((flow_uv[..., 0] + 1000.) * 10).astype(np.uint16))
                cv2.imwrite(save_name_v, ((flow_uv[..., 1] + 1000.) * 10).astype(np.uint16))
                cv2.imwrite(save_name_u_bk, ((flow_uv_bk[..., 0] + 1000.) * 10).astype(np.uint16))
                cv2.imwrite(save_name_v_bk, ((flow_uv_bk[..., 1] + 1000.) * 10).astype(np.uint16))


                torch.cuda.empty_cache()



def demo_waymo():
    model = DIP(max_offset=256, mixed_precision=False, test_mode=True)
    model = torch.nn.DataParallel(model)
    model.cuda()
    warm_start = True

    pre_train = torch.load('/pvc_user/pengliang/DIP/DIP_kitti.pth')
    model.load_state_dict(pre_train, strict=False)

    model.eval()


    raw_img_root_dir = '/pvc_data/personal/pengliang/waymo_kitti_format/training/image_2'
    img_root_dir = '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2'
    # img_root_dir = '/pvc_data/personal/pengliang/waymo_kitti_format/validation/image_2'
    all_img = sorted(os.listdir(img_root_dir))

    with torch.no_grad():
        # for ind, v in enumerate(tqdm(all_img)):
        # for ind, v in enumerate(tqdm(all_img[:10000])):
        # for ind, v in enumerate(tqdm(all_img[10000:20000])):
        # for ind, v in enumerate(tqdm(all_img[20000:30001])):
        # for ind, v in enumerate(tqdm(all_img[30001:])):

        # for ind, v in enumerate(tqdm(all_img[len(all_img)//4:])):
        # for ind, v in enumerate(tqdm(all_img[len(all_img)*2//3:])):
        # for ind, v in enumerate(tqdm(all_img[1350:len(all_img)//3])): # totoal 160000
        # for ind, v in enumerate(tqdm(all_img[1350+8000:len(all_img)//3])):

        # for ind, v in enumerate(tqdm(all_img[0 : len(all_img)//4])):
        #     ind += 0
        # for ind, v in enumerate(tqdm(all_img[len(all_img)//4 : len(all_img)//4*2])):
        #     ind += len(all_img)//4
        # for ind, v in enumerate(tqdm(all_img[len(all_img)//4*2 : len(all_img)//4*3])):
        #     ind += len(all_img)//4*2
        # for ind, v in enumerate(tqdm(all_img[len(all_img)//4*3 : ])):
        #     ind += len(all_img)//4*3

        # th_i = 0
        # # th_i = 1
        # # th_i = 2
        # # th_i = 3
        # # th_i = 4
        # # th_i = 5
        # # th_i = 6
        # # th_i = 7
        # # CUDA_VISIBLE_DEVICES=3 python demo_kitti.py
        # for ind, v in enumerate(tqdm(all_img[len(all_img)//8*th_i : len(all_img)//8*(th_i+1)])):
        #     ind += len(all_img)//8*th_i

        start_len = len(all_img) - 2000
        # th_i = 0
        # th_i = 1
        # th_i = 2
        # th_i = 3
        # th_i = 4
        # th_i = 5
        # th_i = 6
        th_i = 7
        # CUDA_VISIBLE_DEVICES=3 python demo_kitti.py
        for ind, v in enumerate(tqdm(all_img[start_len+2000//8*th_i : start_len+2000//8*(th_i+1)])):
            ind += len(all_img)//8*th_i

            # print(ind, v)
            cur_path = os.path.join(img_root_dir, v)
            # pre_1_path = os.path.join(img_root_dir, all_img[ind-1])
            # pre_2_path = os.path.join(img_root_dir, all_img[ind-2])
            # pre_1_path = os.path.join(img_root_dir, all_img[ind-3])
            # pre_2_path = os.path.join(img_root_dir, all_img[ind-6])
            pre_1_path = os.path.join(raw_img_root_dir, '{:0>6}.png'.format(int(v[:-4])-1))
            pre_2_path = os.path.join(raw_img_root_dir, '{:0>6}.png'.format(int(v[:-4])-2))
            if not os.path.exists(pre_1_path):
                pre_1_path = cur_path
            if not os.path.exists(pre_2_path):
                pre_2_path = pre_1_path


            cur_index, pre_1_index, pre_2_index = int(v[:-4]), int(pre_1_path[-10:-4]), int(pre_2_path[-10:-4])
            if (cur_index - pre_1_index > 20) or (cur_index - pre_1_index) < 0:
                pre_1_path = cur_path
                pre_1_index = int(pre_1_path[-10:-4])
            if (pre_1_index - pre_2_index > 20) or (pre_1_index - pre_2_index < 0):
                pre_2_path = pre_1_path
                pre_2_index = int(pre_2_path[-10:-4])

            print(cur_path, pre_1_path, pre_2_path, th_i)

            save_name_u = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_u.png')
            save_name_v = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_v.png')
            save_name_u_bk = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_bk_u.png')
            save_name_v_bk = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_bk_v.png')
            if os.path.exists(save_name_u) and os.path.exists(save_name_v) and os.path.exists(save_name_u_bk) and os.path.exists(save_name_v_bk):
                continue



            img1 = cv2.imread(pre_1_path, cv2.IMREAD_COLOR)
            img2 = cv2.imread(cur_path, cv2.IMREAD_COLOR)
            img1 = cv2.resize(img1, (1920 // 2, 1280 // 2))
            img2 = cv2.resize(img2, (1920 // 2, 1280 // 2))

            image1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            image2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            image1 = image1[None].to(DEVICE)
            image2 = image2[None].to(DEVICE)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_up = model(image1, image2, iters=20, init_flow=None)
            flow_up = F.interpolate(flow_up, (1280, 1920)) * 2
            flow_uv = flow_up[0].permute(1,2,0).cpu().numpy()

            # color_flow = flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=True)
            # cv2.imwrite('tmp_flow.jpg', color_flow)

            flow_up = model(image2, image1, iters=20, init_flow=None)
            flow_up = F.interpolate(flow_up, (1280, 1920)) * 2
            flow_uv_bk = flow_up[0].permute(1, 2, 0).cpu().numpy()

            # color_flow = flow_to_image(flow_uv_bk, clip_flow=None, convert_to_bgr=True)
            # cv2.imwrite('tmp_flow2.jpg', color_flow)
            # exit(0)

            # assert np.min(flow_uv) > -1000.
            # assert np.min(flow_uv_bk) > -1000.
            if np.min(flow_uv) < -1000:
                print(np.min(flow_uv))
                flow_uv[flow_uv < -1000] = -1000
            if np.min(flow_uv_bk) < -1000:
                print(np.min(flow_uv_bk))
                flow_uv_bk[flow_uv_bk < -1000] = -1000

            # save_name_u = cur_path.replace('image_2', 'waymo_flow').replace('.png', '_u.png')
            # save_name_v = cur_path.replace('image_2', 'waymo_flow').replace('.png', '_v.png')
            # save_name_u_bk = cur_path.replace('image_2', 'waymo_flow').replace('.png', '_bk_u.png')
            # save_name_v_bk = cur_path.replace('image_2', 'waymo_flow').replace('.png', '_bk_v.png')

            # save_name_u = cur_path.replace('image_2', 'waymo_flow_skip3').replace('.png', '_u.png')
            # save_name_v = cur_path.replace('image_2', 'waymo_flow_skip3').replace('.png', '_v.png')
            # save_name_u_bk = cur_path.replace('image_2', 'waymo_flow_skip3').replace('.png', '_bk_u.png')
            # save_name_v_bk = cur_path.replace('image_2', 'waymo_flow_skip3').replace('.png', '_bk_v.png')

            # save_name_u = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_u.png')
            # save_name_v = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_v.png')
            # save_name_u_bk = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_bk_u.png')
            # save_name_v_bk = cur_path.replace('image_2', 'waymo_flow_skip1').replace('.png', '_bk_v.png')

            # print(save_name_u, save_name_v, save_name_u_bk, save_name_v_bk)
            # exit(0)

            cv2.imwrite(save_name_u, ((flow_uv[..., 0] + 1000.) * 10).astype(np.uint16))
            cv2.imwrite(save_name_v, ((flow_uv[..., 1] + 1000.) * 10).astype(np.uint16))
            cv2.imwrite(save_name_u_bk, ((flow_uv_bk[..., 0] + 1000.) * 10).astype(np.uint16))
            cv2.imwrite(save_name_v_bk, ((flow_uv_bk[..., 1] + 1000.) * 10).astype(np.uint16))



    # with torch.no_grad():
    #     images = ['/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2/000000.png',
    #               '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2/000001.png'
    #               ]
    #     for imfile1, imfile2 in zip(images[:-1], images[1:]):
    #         image1 = load_image_waymo(imfile1)
    #         image2 = load_image_waymo(imfile2)
    #
    #         padder = InputPadder(image1.shape)
    #         image1, image2 = padder.pad(image1, image2)
    #
    #         flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
    #         flow_up = F.interpolate(flow_up, (1280, 1920)) * 2
    #         u = flow_up[0, 0, ...]
    #         v = flow_up[0, 1, ...]
    #         v_coord, u_coord = torch.meshgrid(torch.arange(1280), torch.arange(1920))
    #         new_u_coord = (u_coord.cuda() + u).cpu().numpy().reshape(-1)
    #         new_v_coord = (v_coord.cuda() + v).cpu().numpy().reshape(-1)
    #         org_img = F.interpolate(image1, (1280, 1920))[0].permute(1, 2, 0).cpu().numpy()
    #         v_coord = v_coord.cpu().numpy().reshape(-1)
    #         u_coord = u_coord.cpu().numpy().reshape(-1)
    #
    #         new_img = np.zeros((1280, 1920, 3))
    #         new_u_coord = np.clip(np.around(new_u_coord).astype(np.int32), 0, 1919)
    #         new_v_coord = np.clip(np.around(new_v_coord).astype(np.int32), 0, 1279)
    #         new_img[v_coord, u_coord, :] = org_img[new_v_coord, new_u_coord, :]
    #
    #         cv2.imwrite('tmp.png', new_img.astype(np.uint8))




if __name__ == '__main__':
    demo_kittiTestTrain()
    #demo_waymo()
