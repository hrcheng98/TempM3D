import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from datetime import datetime

parser = argparse.ArgumentParser(description='implementation of GUPNet')
parser.add_argument('--config', type=str, default='experiments/1025.yaml')
args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    # logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))
    logger = create_logger('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tmp.log')

    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)

    # for iter in range(120, 300, 10):
    for iter in range(120, 180, 10):
        # cfg['tester']['resume_model'] = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/{}/checkpoints/checkpoint_epoch_{}.pth'.format(cfg['trainer']['log_dir'], iter)
        cfg['tester']['resume_model'] = '/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/{}/checkpoints/checkpoint_epoch_{}.pth'.format('log_kitti_test_4', iter)
        tester = Tester(cfg['tester'], model, test_loader, logger)
        # tester.test(label_path='/private_data/personal/pengliang/OpenPCDet/pred/data')
        # tester.test(label_path='/private/personal/pengliang/OpenPCDet/pred/data')
        tester.test(label_path='/private/pengliang/OpenPCDet/pred/data')


if __name__ == '__main__':
    main()