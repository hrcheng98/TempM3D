import os
import shutil
from tqdm import tqdm
import numpy as np

import torch
import torch.utils.data as data
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


data_dir = '/private_data/personal/pengliang/JGY_model/data_1'
health_all_f = [os.path.join(data_dir+'/health', i) for i in sorted(os.listdir(os.path.join(data_dir+'/health')))]
ill_all_f = [os.path.join(data_dir+'/ill', i) for i in sorted(os.listdir(os.path.join(data_dir+'/ill')))]
# train_dict = {i:0 for i in health_all_f[:-50]}
# train_dict.update({i:1 for i in ill_all_f[:-50]})
# val_dict = {i:0 for i in health_all_f[-50:]}
# val_dict.update({i:1 for i in ill_all_f[-50:]})

train_dict = {i:0 for i in health_all_f[:-100]}
train_dict.update({i:1 for i in ill_all_f[:-100]})
val_dict = {i:0 for i in health_all_f[-100:]}
val_dict.update({i:1 for i in ill_all_f[-100:]})

class jgy_dataset(data.Dataset):
    def __init__(self, mode):
        self.imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]}
        if mode == 'train':
            self.data_dict = train_dict
        else:
            self.data_dict = val_dict

        self.img_list, self.label = [], []
        for i, v in self.data_dict.items():
            self.img_list.append(i)
            self.label.append(v)

    def __len__(self):
        return len(self.data_dict.keys())

    def __getitem__(self, ind):
        img = cv.imread(self.img_list[ind]) / 255.
        # img = cv.resize(img, (512, 512), cv.INTER_AREA)
        # img = cv.resize(img, (256, 256), cv.INTER_AREA)
        img = cv.resize(img, (128, 128), cv.INTER_AREA)
        process_img = []
        for j in range(3):
            process_img.append((img[:, :, j] - self.imagenet_stats['mean'][j]) / self.imagenet_stats['std'][j])
        process_img = np.stack(process_img, axis=2)
        process_img = np.transpose(process_img, [2, 0, 1])

        label = np.array([self.label[ind]])

        return {
                'img': process_img.astype(np.float32),
                'label': label.astype(np.float32)
                }


class JGYModel(nn.Module):
    def __init__(self, features=None):
        super(JGYModel, self).__init__()
        self.features = features

        self.ill = nn.Sequential(
                    # nn.Linear(512 * 16 * 16, 256),
                    # nn.Linear(512 * 8 * 8, 256),
                    nn.Linear(512 * 4 * 4, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 2) # to get sin and cos
                )

    def forward(self, x):
        x = self.features(x) # 512 x 7 x 7

        # x = x.view(-1, 512 * 16 * 16)
        # x = x.view(-1, 512 * 8 * 8)
        x = x.view(-1, 512 * 4 * 4)
        ill = self.ill(x)
        ill_prob = F.sigmoid(ill)

        return ill_prob


def train(restore_path=None):
    from torchvision.models import vgg
    vgg = vgg.vgg19_bn()
    vgg.load_state_dict(torch.load('/private_data/personal/pengliang/vgg19_bn.pth'))
    model = JGYModel(vgg.features)

    if restore_path:
        model.load_state_dict(torch.load(restore_path), strict=False)
    model.cuda()

    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train_dataset = jgy_dataset('train')
    val_dataset = jgy_dataset('val')
    Train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    Val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0, drop_last=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))

    for epoch_idx in range(50000):
        model.train()
        for batch_idx, sample in enumerate(Train_dataloader):
            global_step = len(Train_dataloader) * epoch_idx + batch_idx
            img, label = sample['img'], sample['label']
            img = img.cuda()
            label = label.cuda()

            prob = model(img)
            loss = F.cross_entropy(prob, label.type(torch.long)[:, 0])
            # loss = F.smooth_l1_loss(prob[:, 0], label[:, 0])

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
                good = 1 if prob[0, 1] > prob[0, 0] else 0
                if good == int(label[0, 0]):
                    good_cnt += 1
                else:
                    bad_cnt += 1
            print("correct: {}, wrong: {}, accuracy: {}".format(good_cnt, bad_cnt, float(good_cnt)/(good_cnt+bad_cnt)))




if __name__ == '__main__':
    train()



# health1_path = '/Users/pengliang/Downloads/JGY_model/30例胸椎结核和正常人的MRI/正常人/正常胸椎'
#
# all_dir1 = os.listdir(health1_path)
#
# data_dir = '/Users/pengliang/Downloads/JGY_model/data_1'
# agg_health_dir = '/Users/pengliang/Downloads/JGY_model/data_1/health'
# os.makedirs(data_dir, exist_ok=True)
# os.makedirs(agg_health_dir, exist_ok=True)
#
# count = 0
# for dir in tqdm(all_dir1):
#     tmp_dir = os.path.join(health1_path, dir)
#     if not os.path.isdir(tmp_dir):
#         continue
#     tmp_f = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
#     for f in tmp_f:
#         if f[-3:] == 'jpg':
#             fsize = os.path.getsize(f)
#             fsize = fsize / float(1024)
#             if fsize > 50:
#                 shutil.copy(f, os.path.join(agg_health_dir, '{:0>6}.jpg'.format(count)))
#                 count += 1


# ill1_path = '/Users/pengliang/Downloads/JGY_model/30例胸椎结核和正常人的MRI/脊柱结核病人/胸椎结核病人'
#
# all_dir1 = os.listdir(ill1_path)
#
# data_dir = '/Users/pengliang/Downloads/JGY_model/data_1'
# agg_ill_dir = '/Users/pengliang/Downloads/JGY_model/data_1/ill'
# os.makedirs(data_dir, exist_ok=True)
# os.makedirs(agg_ill_dir, exist_ok=True)
#
# count = 0
# for dir in tqdm(all_dir1):
#     tmp_dir = os.path.join(ill1_path, dir)
#     if not os.path.isdir(tmp_dir):
#         continue
#     tmp_f = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)]
#     for f in tmp_f:
#         if f[-3:] == 'jpg':
#             fsize = os.path.getsize(f)
#             fsize = fsize / float(1024)
#             if fsize > 50:
#                 shutil.copy(f, os.path.join(agg_ill_dir, '{:0>6}.jpg'.format(count)))
#                 count += 1
