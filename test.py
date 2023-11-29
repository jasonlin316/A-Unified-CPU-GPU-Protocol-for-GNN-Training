import argparse
import random

import dgl
import torch
from dgl import transforms, NID
from dgl.data import AsNodePredDataset
from dgl.dataloading import NeighborSampler, DataLoader, ShaDowKHopSampler, Sampler
from dgl.sampling.utils import EidExcluder
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm
import dgl.sparse as dglsp

parser = argparse.ArgumentParser()
parser.add_argument('--sampler',
                    type=str,
                    default='neighbor',
                    choices=["neighbor", "shadow"])
parser.add_argument('--batch_size',
                    type=int,
                    default=1024,)
parser.add_argument('--device',
                    type=int,
                    default=0,)
parser.add_argument('--dataset',
                    type=str,
                    default='ogbn-products',
                    choices=["ogbn-papers100M", "ogbn-products", "mag240M"])
parser.add_argument('--matrix', action='store_true')
args = parser.parse_args()

fanouts = [15, 10, 5]
# fanouts = [30, 20, 10]
# fanouts = [5, 10, 15]

dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, '/home/jason/DDP_GNN/dataset/'))
g = dataset[0]
train_idx = dataset.train_idx

if args.sampler.lower() == 'neighbor':
    sampler = NeighborSampler(
        fanouts,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
elif args.sampler.lower() == 'shadow':
    sampler = ShaDowKHopSampler(  # CPU sampling is 2x faster than GPU sampling
        fanouts,
        output_device=0,  # comment out in CPU sampling version
        prefetch_node_feats=["feat"],
    )
else:
    raise NotImplementedError

if args.batch_size == -1:
    args.batch_size = len(train_idx)
loader = DataLoader(
    g,
    train_idx.to(args.device),
    sampler,
    batch_size=args.batch_size,
    device=args.device,
    drop_last=False,
    use_uva=True,
    shuffle=False,
)

runs = 10
avg_traversed = []
if args.matrix:
    for run in range(runs):
        num_traversed_edges = []
        for nid, (input_nodes, output_nodes, blocks) in tqdm(enumerate(loader)):
            X = torch.ones(blocks[0].num_src_nodes(), dtype=torch.float).to(args.device)
            for block in blocks:
                A = dglsp.spmatrix(torch.stack(block.edges()),
                                   shape=(block.num_src_nodes(), block.num_dst_nodes()))
                X = A.T @ X
            num_traversed_edges.append(X)
        avg_traversed.append(torch.cat(num_traversed_edges))
else:
    for run in range(runs):
        num_traversed_edges = []
        for nid, (input_nodes, output_nodes, blocks) in tqdm(enumerate(loader)):
            num_edges = 0
            for block in blocks:
                num_edges += block.num_edges()
            num_traversed_edges.append(num_edges)
        avg_traversed.append(torch.tensor(num_traversed_edges))

avg_traversed = torch.stack(avg_traversed).float().mean(dim=0)
file_name = '{}_{}_{}_{}'.format(args.dataset, args.sampler, str(fanouts), args.batch_size) + \
            ('_matrix' if args.matrix else '')
torch.save(avg_traversed.cpu(), "processed/{}.pt".format(file_name))
breakpoint()
