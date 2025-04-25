import numpy as np
import os
from tqdm import tqdm

all_f = [i[:-4] for i  in sorted(os.listdir('/private/personal/pengliang/KITTI3D/testing/image_2'))]
for f in tqdm(all_f):
    if os.path.exists(os.path.join('/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v3/data', f+'.txt')):
        c = 1
    else:
        print(os.path.join('/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v3/data', f+'.txt'))
        np.savetxt(os.path.join('/private/personal/pengliang/history_submit/DID-M3D/DID-M3D_v3/data', f+'.txt'), np.array([]), fmt='%s')
