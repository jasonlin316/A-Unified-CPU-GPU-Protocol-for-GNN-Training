import argparse

import dgl

import torch
from dgl.data import AsNodePredDataset, RedditDataset
from ogb.lsc import MAG240MDataset
from ogb.nodeproppred import DglNodePropPredDataset

from load_mag_to_shm import fetch_mag_from_shm
from load_papers_to_shm import fetch_papers_from_shm


def download(name, data_path):
    if name == 'mag240M':
        MAG240MDataset(root=data_path+'/mag240M')
    elif 'ogbn' in name:
        DglNodePropPredDataset(name, data_path)
    elif name == 'reddit':
        RedditDataset(raw_dir=data_path)


def get_data(name, data_path):
    if name == 'mag240M':
        dataset = MAG240MDataset(root=data_path+'/mag240M')
        print('Start Loading Graph Structure')
        (g,), _ = dgl.load_graphs(data_path+'/mag240M/graph.dgl')
        g = g.formats(["csc"])
        print('Graph Structure Loading Finished!')
        paper_offset = dataset.num_authors + dataset.num_institutions
        dataset.train_idx = torch.from_numpy(dataset.get_idx_split("train")) + paper_offset
        g.ndata["feat"] = fetch_mag_from_shm()
        g.ndata["label"] = torch.cat([torch.empty((paper_offset,), dtype=torch.long),
                                      torch.LongTensor(dataset.paper_label[:])])
        print('Graph Feature/Label Loading Finished!')
    elif name == 'ogbn-papers100M':
        dataset, g = fetch_papers_from_shm()
    elif 'ogbn' in name:
        dataset = AsNodePredDataset(DglNodePropPredDataset(name, data_path))
        g = dataset[0]
    elif name == 'reddit':
        dataset = RedditDataset(raw_dir=data_path)
        g = dataset[0]
        train_idx = g.ndata['train_mask'].nonzero().view(-1)
        return dataset.num_classes, train_idx, g
    else:
        raise NotImplementedError
    """
    Note 1: This func avoid creating certain graph formats in each sub-process to save memory
    Note 2: This func will init CUDA. It is not possible to use CUDA in a child process 
            created by fork(), if CUDA has been initialized in the parent process. 
    """
    # g.create_formats_()
    return dataset.num_classes, dataset.train_idx, g


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products')
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/gangda')
    args = parser.parse_args()
    download(args.dataset, args.data_path)
