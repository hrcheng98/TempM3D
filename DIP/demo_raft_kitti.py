import sys

sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import torch.nn.functional as F

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    # img = cv2.resize(img, (1920//2, 1280//2))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def load_image_waymo(imfile):
    # img = np.array(Image.open(imfile)).astype(np.uint8)
    img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = cv2.resize(img, (1920 // 2, 1280 // 2))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# def viz(img, flo):
#     img = img[0].permute(1, 2, 0).cpu().numpy()
#     flo = flo[0].permute(1, 2, 0).cpu().numpy()
#
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([img, flo], axis=0)
#
#     # import matplotlib.pyplot as plt
#     # plt.imshow(img_flo / 255.0)
#     # plt.show()
#
#     cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
#     cv2.waitKey()

def viz(img, flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    cv2.imwrite('tmp.png', flo[:, :, [2, 1, 0]].astype(np.uint8))


# # a = np.load('/pvc_data/personal/pengliang/private_data/RAFT/KITTI3D_flow_bk/training/{:0>6}.npy'.format(6915*2))
# a_u = cv2.imread('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all/training/{:0>6}_u.png'.format(6915*2+1), -1) / 10. - 1000.
# a_v = cv2.imread('/pvc_data/personal/pengliang/KITTI3D/kitti_flow_all/training/{:0>6}_v.png'.format(6915*2+1), -1) / 10. - 1000.
# a = np.stack([a_u, a_v], axis=2)
# flo = flow_viz.flow_to_image(-a)
# cv2.imwrite('tmp_00.png', flo[:, :, [2,1,0]].astype(np.uint8))
# c = 1


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            #
            # u = flow_up[0, 0, ...]
            # v = flow_up[0, 1, ...]
            # v_coord, u_coord = torch.meshgrid(torch.arange(376), torch.arange(1248))
            # new_u_coord = u_coord.cuda() + u
            # new_v_coord = v_coord.cuda() + v
            # new_u_coord = torch.clamp(new_u_coord, 0, 1247).type(torch.long).cpu().numpy()
            # new_v_coord = torch.clamp(new_v_coord, 0, 375).type(torch.long).cpu().numpy()
            # new_img = np.zeros((376, 1248, 3))
            # org_img = image1[0].permute(1,2,0).cpu().numpy()
            # new_img[new_v_coord.reshape(-1), new_u_coord.reshape(-1), :] = \
            #     org_img[v_coord.reshape(-1).type(torch.long).cpu().numpy(), u_coord.reshape(-1).type(torch.long).cpu().numpy(), :]
            #
            # viz(image1, flow_up)

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            u = flow_up[0, 0, ...]
            v = flow_up[0, 1, ...]
            v_coord, u_coord = torch.meshgrid(torch.arange(376), torch.arange(1248))
            new_u_coord = (u_coord.cuda() + u).cpu().numpy().reshape(-1)
            new_v_coord = (v_coord.cuda() + v).cpu().numpy().reshape(-1)
            org_img = image1[0].permute(1, 2, 0).cpu().numpy()
            v_coord = v_coord.cpu().numpy().reshape(-1)
            u_coord = u_coord.cpu().numpy().reshape(-1)

            # new_img = np.zeros((376, 1248, 3))
            # u_low = np.clip(new_u_coord.astype(np.int32), 0, 1247)
            # u_high = np.clip(new_u_coord.astype(np.int32) + 1, 0, 1247)
            # v_low = np.clip(new_v_coord.astype(np.int32), 0, 375)
            # v_high = np.clip(new_v_coord.astype(np.int32)+1, 0, 375)
            #
            # new_img[v_coord, u_coord, :] = 1/4*(org_img[v_low, u_low, :] + org_img[v_low, u_high, :] +
            #                                     org_img[v_high, u_low, :] + org_img[v_high, u_high, :])
            new_img = np.zeros((376, 1248, 3))
            new_u_coord = np.clip(np.around(new_u_coord).astype(np.int32), 0, 1247)
            new_v_coord = np.clip(np.around(new_v_coord).astype(np.int32), 0, 375)
            new_img[v_coord, u_coord, :] = org_img[new_v_coord, new_u_coord, :]
            viz(image1, flow_up)


def demo_kitti3d(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    img_pair = np.loadtxt('kitti3d_training_flow_seq.txt', dtype=str)

    with torch.no_grad():
        # for imfile1, imfile2, save_flow_path in tqdm(zip(img_pair[:, 0], img_pair[:, 1], img_pair[:, 2])):
        for imfile2, imfile1, save_flow_path in tqdm(zip(img_pair[:, 0], img_pair[:, 1], img_pair[:, 2])):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # np.save(save_flow_path.replace('png', 'npy').replace('/personal/', '/'), flow_uv)
            np.save(save_flow_path.replace('png', 'npy').replace('/personal/', '/').replace('KITTI3D_flow',
                                                                                            'KITTI3D_flow_bk'), flow_uv)
            # viz(image1, flow_up)


def demo_kittiTestTrain(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    split = ['testing', 'testing_flow', 7518]
    # split = ['training', 'training_flow', 7481]
    kitti_test_dir = '/pvc_data/personal/pengliang/KITTI3D/{}/image_2'.format(split[0])
    kitti_test_pre_data_dir = '/pvc_data/personal/pengliang/KITTI3D/kitti_3d_pre/data_object_prev_2/{}/prev_2'.format(
        split[0])

    with torch.no_grad():
        for idx in tqdm(range(split[2])):
            str_idx = '{:0>6}'.format(idx)
            flow_id = ['00', '01', '02', '03']
            for j, _ in enumerate(flow_id[:-1]):
                if not os.path.exists(os.path.join(kitti_test_pre_data_dir, str_idx + '_' + flow_id[j + 1] + '.png')):
                    continue

                im_name1 = os.path.join(kitti_test_pre_data_dir, str_idx + '_' + flow_id[j + 1] + '.png')
                if flow_id[j] == '00':
                    im_name2 = os.path.join(kitti_test_dir, str_idx + '.png')
                else:
                    im_name2 = os.path.join(kitti_test_pre_data_dir, str_idx + '_' + flow_id[j] + '.png')
                image1 = load_image(im_name1)
                image2 = load_image(im_name2)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()

                flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
                flow_uv_bk = flow_up[0].permute(1, 2, 0).cpu().numpy()

                assert np.min(flow_uv) > -1000.
                assert np.min(flow_uv_bk) > -1000.

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


def demo_kittiRaw(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    kitti_raw_data_dir = '/pvc_data/personal/pengliang/kitti_raw'
    kitti_raw_data_dir_flow = '/pvc_data/personal/pengliang/kitti_raw_flow'

    with torch.no_grad():
        data_dir = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        for d1 in tqdm(data_dir):
            data_dir2 = sorted(os.listdir(os.path.join(kitti_raw_data_dir, d1)))
            for d2 in tqdm(data_dir2):
                if os.path.exists(os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points')):
                    im_ind = sorted(os.listdir(os.path.join(kitti_raw_data_dir, d1, d2, 'velodyne_points', 'data')))

                    im2 = [os.path.join(kitti_raw_data_dir, d1, d2, 'image_02', 'data', i.replace('bin', 'png')) for i
                           in im_ind]
                    im_name = np.array(im2)

                    to_save_dir = os.path.join(kitti_raw_data_dir_flow, d1, d2, 'image_02', 'data')
                    if not os.path.exists(to_save_dir):
                        os.makedirs(to_save_dir)

                    for idx in tqdm(range(len(im_name) - 1)):
                        image1 = load_image(im_name[idx])
                        image2 = load_image(im_name[idx + 1])
                        padder = InputPadder(image1.shape)
                        image1, image2 = padder.pad(image1, image2)

                        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                        flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()

                        flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
                        flow_uv_bk = flow_up[0].permute(1, 2, 0).cpu().numpy()

                        assert np.min(flow_uv) > -1000.
                        assert np.min(flow_uv_bk) > -1000.

                        save_name_u = im_name[idx + 1].replace('kitti_raw', 'kitti_raw_flow').replace('.png', '_u.png')
                        save_name_v = im_name[idx + 1].replace('kitti_raw', 'kitti_raw_flow').replace('.png', '_v.png')
                        save_name_u_bk = im_name[idx + 1].replace('kitti_raw', 'kitti_raw_flow').replace('.png',
                                                                                                         '_bk_u.png')
                        save_name_v_bk = im_name[idx + 1].replace('kitti_raw', 'kitti_raw_flow').replace('.png',
                                                                                                         '_bk_v.png')

                        cv2.imwrite(save_name_u, ((flow_uv[..., 0] + 1000.) * 10).astype(np.uint16))
                        cv2.imwrite(save_name_v, ((flow_uv[..., 1] + 1000.) * 10).astype(np.uint16))
                        cv2.imwrite(save_name_u_bk, ((flow_uv_bk[..., 0] + 1000.) * 10).astype(np.uint16))
                        cv2.imwrite(save_name_v_bk, ((flow_uv_bk[..., 1] + 1000.) * 10).astype(np.uint16))


def demo_kitti3d_test(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    img_pair = np.loadtxt('kitti3d_testing_flow_seq.txt', dtype=str)

    with torch.no_grad():
        # for imfile1, imfile2, save_flow_path in tqdm(zip(img_pair[:, 0], img_pair[:, 1], img_pair[:, 2])):
        for imfile2, imfile1, save_flow_path in tqdm(zip(img_pair[:, 0], img_pair[:, 1], img_pair[:, 2])):
            # image1 = load_image(imfile1.replace('/personal/', '/'))
            # image2 = load_image(imfile2.replace('/personal/', '/'))
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()
            # np.save(save_flow_path.replace('png', 'npy').replace('/personal/', '/'), flow_uv)
            # np.save(save_flow_path.replace('png', 'npy').replace('/personal/', '/').replace('KITTI3D_flow', 'KITTI3D_flow_bk'), flow_uv)
            np.save(save_flow_path.replace('png', 'npy').replace('/private/personal/pengliang',
                                                                 '/pvc_data/personal/pengliang/private_data').replace(
                'KITTI3D_flow', 'KITTI3D_flow_bk'), flow_uv)


def demo_waymo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    img_root_dir = '/pvc_data/personal/pengliang/waymo_kitti_format/training_sampled_3/image_2'
    all_img = sorted(os.listdir(img_root_dir))

    with torch.no_grad():
        for ind, v in tqdm(enumerate(all_img)):
            cur_path = os.path.join(img_root_dir, v)
            pre_1_path = os.path.join(img_root_dir, all_img[ind - 1])
            pre_2_path = os.path.join(img_root_dir, all_img[ind - 2])

            cur_index, pre_1_index, pre_2_index = int(v[:-4]), int(pre_1_path[-10:-4]), int(pre_2_path[-10:-4])
            if (cur_index - pre_1_index > 20) or (cur_index - pre_1_index) < 0:
                pre_1_path = cur_path
                pre_1_index = int(pre_1_path[-10:-4])
            if (pre_1_index - pre_2_index > 20) or (pre_1_index - pre_2_index < 0):
                pre_2_path = pre_1_path
                pre_2_index = int(pre_2_path[-10:-4])

            ##################################################
            image1 = load_image_waymo(pre_1_path)
            image2 = load_image_waymo(cur_path)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            flow_up = F.interpolate(flow_up, (1280, 1920)) * 2
            flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()
            np.save(cur_path.replace('.png', '_01.npy').replace('image_2', 'waymo_flow'), flow_uv)

            ##################################################
            image1 = load_image_waymo(pre_2_path)
            image2 = load_image_waymo(cur_path)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
            flow_up = F.interpolate(flow_up, (1280, 1920)) * 2
            flow_uv = flow_up[0].permute(1, 2, 0).cpu().numpy()
            np.save(cur_path.replace('.png', '_02.npy').replace('image_2', 'waymo_flow'), flow_uv)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # demo(args)
    # demo_waymo(args)
    # demo_kitti3d(args)
    # demo_kittiRaw(args)
    demo_kittiTestTrain(args)
    # demo_kitti3d_test(args)
    # demo_waymo(args)
