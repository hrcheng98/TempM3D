import os
import shutil
from tqdm import tqdm
import numpy as np

import torch

def train(restore_path=None):
    while True:
        a = torch.ones((10, 10)).cuda()
        b = torch.ones((10, 10)).cuda()
        c = a* b


if __name__ == '__main__':
    train()

