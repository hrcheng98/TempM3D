import os
import json
import numpy as np
import cv2 as cv


import os
import shutil
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# a = "/Users/pengliang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/e2e0bfb3b6a6a46041b50cc3a7e10592/Message/MessageTemp/3666e5beca3a79bcb3f7f213423e527d/File/CT骨窗/20张病灶图/001.json"
# with open(a, 'r') as fcc_file:
#     fcc_data = json.load(fcc_file)
#
# img_path = "/Users/pengliang/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/e2e0bfb3b6a6a46041b50cc3a7e10592/Message/MessageTemp/3666e5beca3a79bcb3f7f213423e527d/File/CT骨窗/20张原图/001.png"
# # img = cv.imread(img_path, -1) / 255.
# img = np.zeros((fcc_data['imageHeight'], fcc_data['imageWidth']))
# print(img.shape)
# obj_num = len(fcc_data['shapes'])
# for i in range(obj_num):
#     points = np.array(fcc_data['shapes'][i]['points'])
#     # cv.polylines(img, [points.reshape(-1,1,2).astype(np.int32)], True, (0, 255, 255), 2)
#     # cv.fillConvexPoly(img, points.reshape((-1,1,2)).astype(np.int32), (0, 255, 255), 2)
#     # cv.fillConvexPoly(img, [points.reshape((-1,1,2)).astype(np.int32)], 255)
#     # cv.fillConvexPoly(img, points.reshape((-1,1,2)).astype(np.int32), 255)
#     cv.fillPoly(img, [points.reshape((-1,1,2)).astype(np.int32)], 255)
#     print(img.shape)
#     cv.imshow('test', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()


def mask_label(json_path):
    with open(json_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        mask = np.zeros((fcc_data['imageHeight'], fcc_data['imageWidth']))
        obj_num = len(fcc_data['shapes'])
        for i in range(obj_num):
            points = np.array(fcc_data['shapes'][i]['points'])
            cv.fillPoly(mask, [points.reshape((-1, 1, 2)).astype(np.int32)], 1)

        return mask


data_dir = '/pvc_data/personal/pengliang/mask_jgy/'
image_f = [os.path.join(data_dir+'/Images', i) for i in sorted(os.listdir(os.path.join(data_dir+'/Images')))]
label_f = [os.path.join(data_dir+'/Labels', i) for i in sorted(os.listdir(os.path.join(data_dir+'/Labels')))]
all_f = np.array([i[:-5] for i in os.listdir(os.path.join(data_dir+'/Labels'))])

np.random.shuffle(all_f)
train_set = all_f[:16]
# train_set = all_f[:1]
val_set = all_f[16:]

class jgy_dataset(data.Dataset):
    def __init__(self, mode):
        self.imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]}
        if mode == 'train':
            self.data_set = train_set
        else:
            self.data_set = val_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, ind):
        img = cv.imread(os.path.join(data_dir+'/Images/{}.png'.format(self.data_set[ind]))) / 255.
        img = cv.resize(img, (512, 512), cv.INTER_AREA)
        # process_img = []
        # for j in range(3):
        #     process_img.append((img[:, :, j] - self.imagenet_stats['mean'][j]) / self.imagenet_stats['std'][j])
        # process_img = np.stack(process_img, axis=2)
        # process_img = np.transpose(process_img, [2, 0, 1])
        process_img = np.transpose(img, [2, 0, 1])

        label = mask_label(os.path.join(data_dir+'/Labels/{}.json'.format(self.data_set[ind])))
        label = cv.resize(label, (512, 512), cv.INTER_NEAREST)

        return {
                'img': process_img.astype(np.float32),
                'label': label.astype(np.float32)
                }





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)


def focal_loss(input, target, alpha=0.25, gamma=2.):
    '''
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        alpha: hyper param, default in 0.25
        gamma: hyper param, default in 2.0
    Reference: Focal Loss for Dense Object Detection, ICCV'17
    '''

    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    # print(torch.sum(pos_inds), torch.sum(neg_inds), torch.sum(pos_inds)+torch.sum(neg_inds))

    loss = 0

    pos_loss = torch.log(input) * torch.pow(1 - input, gamma) * pos_inds * alpha
    neg_loss = torch.log(1 - input) * torch.pow(input, gamma) * neg_inds * (1 - alpha)

    num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss.mean()


def train(restore_path=None):
    model = UNet(3, 1, bilinear=True)

    if restore_path:
        model.load_state_dict(torch.load(restore_path), strict=False)
    model.cuda()

    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train_dataset = jgy_dataset('train')
    val_dataset = jgy_dataset('val')
    Train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    Val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))

    for epoch_idx in range(50000):
        model.train()
        for batch_idx, sample in enumerate(Train_dataloader):
            global_step = len(Train_dataloader) * epoch_idx + batch_idx
            img, label = sample['img'], sample['label']
            img = img.cuda()
            label = label.cuda()

            prob = model(img)
            prob = F.sigmoid(prob.squeeze(1))
            # loss = F.binary_cross_entropy_with_logits(prob, label)
            # loss = F.smooth_l1_loss(prob, label)
            loss = focal_loss(prob, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('epoch: {}, batch: {}, loss: {}'.format(epoch_idx, batch_idx, float(loss)))

        with torch.no_grad():
            model.eval()
            good_cnt, bad_cnt = 0, 0
            for batch_idx, sample in enumerate(Val_dataloader):
                img, label = sample['img'], sample['label']
                img = img.cuda()
                label = label.cuda()

                prob = model(img)
                prob = F.sigmoid(prob.squeeze(1))
                tmp = 1
        #         good = 1 if prob[0, 1] > prob[0, 0] else 0
        #         if good == int(label[0, 0]):
        #             good_cnt += 1
        #         else:
        #             bad_cnt += 1
        #     print("correct: {}, wrong: {}, accuracy: {}".format(good_cnt, bad_cnt, float(good_cnt)/(good_cnt+bad_cnt)))

        # b
        # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/lib/models/tmp.png', (label[0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv.imwrite('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/lib/models/tmp2.png', (img[0].permute(1,2,0).detach().cpu().numpy()*100).astype(np.uint8))

if __name__ == '__main__':
    train()