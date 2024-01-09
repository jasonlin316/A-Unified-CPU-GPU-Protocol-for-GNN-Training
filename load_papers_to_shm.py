import argparse
import threading
import time

import dgl
import torch
from dgl.data import AsNodePredDataset
from dgl.utils.shared_mem import create_shared_mem_array, get_shared_mem_array
from ogb.nodeproppred import DglNodePropPredDataset

from utils import Dict

PAPER_FEATS_KEY = 'PAPER_feat_full2'
PAPER_EDGES_KEY = 'PAPER_edges_coo2'
PAPER_TRAIN_IDX_KEY = 'PAPER_train_idx2'
PAPER_LABEL_KEY = 'PAPER_label_full2'
PAPER_FEATS_SHAPE = (111059956, 128)
PAPER_EDGES_SHAPE = (2, 1615685872)
PAPER_TRAIN_IDX_SHAPE = (1207179,)
PAPER_LABEL_SHAPE = (111059956,)
NUM_CLASSES = 172

def fetch_papers_from_shm():
    feat = get_shared_mem_array(PAPER_FEATS_KEY, PAPER_FEATS_SHAPE, dtype=torch.float32)
    edges = get_shared_mem_array(PAPER_EDGES_KEY, PAPER_EDGES_SHAPE, dtype=torch.long)
    label = get_shared_mem_array(PAPER_LABEL_KEY, PAPER_LABEL_SHAPE, dtype=torch.float32)
    train_idx = get_shared_mem_array(PAPER_TRAIN_IDX_KEY, PAPER_TRAIN_IDX_SHAPE, dtype=torch.long)

    g = dgl.graph((edges[0], edges[1]))
    g.ndata['feat'] = feat
    g.ndata['label'] = label

    dataset = Dict({
        'num_classes': NUM_CLASSES,
        'train_idx': train_idx
    })
    return dataset, g


def host_datas(e):
    tik = time.time()
    print('Start Loading features')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M', args.data_path))
    g = dataset[0]
    print('Loading feature finished')
    print('Start moving features to shm')
    feat = create_shared_mem_array(PAPER_FEATS_KEY, PAPER_FEATS_SHAPE, dtype=torch.float32)
    edges = create_shared_mem_array(PAPER_EDGES_KEY, PAPER_EDGES_SHAPE, dtype=torch.long)
    label = create_shared_mem_array(PAPER_LABEL_KEY, PAPER_LABEL_SHAPE, dtype=torch.float32)
    train_idx = create_shared_mem_array(PAPER_TRAIN_IDX_KEY, PAPER_TRAIN_IDX_SHAPE, dtype=torch.long)
    feat[:] = g.ndata['feat']
    edges[:] = torch.stack(g.edges())
    label[:] = g.ndata['label']
    train_idx[:] = dataset.train_idx
    del dataset
    tok = time.time()
    print(f'Loading data finished, Total execution time:{tok - tik: .1f}s')

    print('Press Ctrl+D to exit')
    breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/gangda')
    args = parser.parse_args()
    event = threading.Event()
    threading.Thread(target=host_datas,
                     args=[event],
                     daemon=True).start()
    try:
        event.wait()
    except KeyboardInterrupt:
        print('Release shm')
