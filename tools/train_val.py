import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)



import yaml
import logging
import argparse
import pytz


from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from datetime import datetime




parser = argparse.ArgumentParser(description='implementation of GUPNet')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--config', type=str, default = 'experiments/config.yaml')

# Multi-gpu training
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size', type=int, help='number of nodes for distributed training', default=1)
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)
parser.add_argument('--dist_url', type=str, help='url used to set up distributed training',
                    default='tcp://127.0.0.1:5678')
parser.add_argument('--dist_backend', type=str, help='distributed backend', default='nccl')
parser.add_argument('--gpu', type=int, help='GPU id to use.', default=None)


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

    
    current_time = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    work_dir = 'work_dir/TIP/' + args.config.split('/')[-1].split('.')[0] + '/' + current_time
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    cfg['trainer']['log_dir'] = work_dir + '/' + cfg['trainer']['log_dir']
    cfg['trainer']['out_dir'] = work_dir + '/' + cfg['trainer']['out_dir']

    os.makedirs(cfg['trainer']['log_dir'],exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'],'train.log'))
    # logger = create_logger('/pvc_user/pengliang/GUPNet_master/GUPNet-main/code/tmp.log')
    
    #  build dataloader
    # train_loader, val_loader, _ = build_dataloader(cfg['dataset'])
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'],train_loader.dataset.cls_mean_size)

    # evaluation mode
    if args.evaluate:
        tester = Tester(cfg['tester'], model, val_loader, logger)
        # tester = Tester(cfg['tester'], model, test_loader, logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger)
    trainer.train()
    # tester = Tester(cfg['tester'], model, val_loader, logger)
    # tester.test()






import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.rank)

    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.rank % ngpus_per_node == 0:
        os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))

    #  build dataloader
    from lib.datasets.kitti_flow_opt import KITTI
    from torch.utils.data import DataLoader

    train_set = KITTI(root_dir=cfg['dataset']['root_dir'], split='train', cfg=cfg['dataset'])
    val_set = KITTI(root_dir=cfg['dataset']['root_dir'], split='val', cfg=cfg['dataset'])
    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    cfg['dataset']['batch_size'] = int(cfg['dataset']['batch_size'] / ngpus_per_node)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['dataset']['batch_size'],
                              num_workers=12,
                              shuffle=False,
                              sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=cfg['dataset']['batch_size'],
                            sampler=val_sampler,
                             num_workers=12,
                             shuffle=False,
                            pin_memory=True,
                             drop_last=True)

    # build model
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)


    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    cudnn.benchmark = True

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      device=torch.device("cuda:{}".format(args.gpu)),
                      dist=True,
                      rank=args.rank)
    trainer.train()


def dist_main():
    torch.cuda.empty_cache()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))




if __name__ == '__main__':
    main()
    # dist_main()