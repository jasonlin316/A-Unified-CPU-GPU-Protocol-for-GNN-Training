import argparse
import threading
import time

import torch
import numpy as np
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array

FEATS_TYPE = torch.float16
MAG_FEATS_KEY = 'MAG240_feat_full'
MAG_FEATS_SHAPE = (244160499, 768)

def fetch_mag_from_shm():
    return get_shared_mem_array(MAG_FEATS_KEY, MAG_FEATS_SHAPE, dtype=FEATS_TYPE)


def host_datas(e):
    tik = time.time()

    feats_disk = np.memmap(
        args.data_path+'/mag240M/full.npy',
        mode="r",
        dtype="float16",
        shape=MAG_FEATS_SHAPE,
    )
    print('Start Loading features, needs 400G available memory')
    feats_mem = torch.from_numpy(feats_disk[:])
    print('Start moving features to shm')
    feats_shm = create_shared_mem_array(MAG_FEATS_KEY, MAG_FEATS_SHAPE, dtype=FEATS_TYPE)
    feats_shm[:] = feats_mem
    del feats_mem

    tok = time.time()
    print(f'Loading data finished, Total execution time:{tok - tik: .1f}s')

    print('Press Ctrl+C to exit')
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/gangda')
    args = parser.parse_args()

    event = threading.Event()
    threading.Thread(target=host_datas, args=[event], daemon=True).start()
    try:
        event.wait()
    except KeyboardInterrupt:
        print('Release shm')
