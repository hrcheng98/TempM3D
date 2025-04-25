import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti import KITTI

from lib.datasets.kitti_flow_opt import KITTIFlow
from lib.datasets.waymo_flow import WaymoFlow
# from lib.datasets.kitti_flow_opt_new import KITTIFlow

from lib.datasets.waymo import Waymo

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



'''pvc data useage'''
def build_dataloader(cfg):
    # --------------  build kitti dataset ----------------
    if cfg['type'] == 'kitti':
        # train_set = KITTI(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_set = KITTIFlow(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        # train_set = KITTIFlow(root_dir=cfg['root_dir'], split='trainval', cfg=cfg)
        # train_set = KITTI(root_dir=cfg['root_dir'], split='trainval', cfg=cfg)
        # train_set = KITTI_aug(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  # num_workers=12,
                                  num_workers=32,
                                  shuffle=True,
                                  # worker_init_fn=my_worker_init_fn,
                                  pin_memory=True,
                                  drop_last=True)
        # val_set = KITTI(root_dir=cfg['root_dir'], split='val', cfg=cfg)
        val_set = KITTIFlow(root_dir=cfg['root_dir'], split='val', cfg=cfg)
        val_loader = DataLoader(dataset=val_set,
                                 batch_size=cfg['batch_size'],
                                 # num_workers=2,
                                 # num_workers=4,
                                 # num_workers=12,
                                 num_workers=32,
                                 # num_workers=0,
                                 shuffle=False,
                                # worker_init_fn=my_worker_init_fn,
                                pin_memory=True,
                                 drop_last=True)
        # test_set = KITTI(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        # test_set = KITTIFlow(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        # test_loader = DataLoader(dataset=test_set,
        #                          batch_size=cfg['batch_size'],
        #                          # num_workers=12,
        #                          num_workers=32,
        #                          shuffle=False,
        #                          # worker_init_fn=my_worker_init_fn,
        #                          pin_memory=True,
        #                          # pin_memory=False,
        #                          drop_last=False)
        #                          # drop_last=True)
        test_loader = None
        return train_loader, val_loader, test_loader

    elif cfg['type'] == 'waymo':
        # train_set = Waymo(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        train_set = WaymoFlow(root_dir=cfg['root_dir'], split='train', cfg=cfg)
        print(16)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  # num_workers=0,
                                  # num_workers=2,
                                  # num_workers=16,
                                  num_workers=32,
                                  # num_workers=64,
                                  # num_workers=8,
                                  # num_workers=4,
                                  shuffle=True,
                                  worker_init_fn=my_worker_init_fn,
                                  pin_memory=True,
                                  # pin_memory=False,
                                  drop_last=False)
        # test_set = Waymo(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        test_set = WaymoFlow(root_dir=cfg['root_dir'], split='test', cfg=cfg)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 # num_workers=16,
                                 # num_workers=0,
                                 num_workers=32,
                                 # num_workers=8,
                                 shuffle=False,
                                 # worker_init_fn=my_worker_init_fn,
                                 pin_memory=True,
                                 # pin_memory=False,
                                 drop_last=False)
        # return train_loader, train_loader, test_loader
        return train_loader, test_loader, test_loader

    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

