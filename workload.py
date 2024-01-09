import argparse
import os

import torch
from dgl.dataloading import NeighborSampler, DataLoader, ShaDowKHopSampler, Sampler
from tqdm import tqdm
import dgl.sparse as dglsp

from dataset import get_data

PROCESS_DIR = "processed/"
if not os.path.exists(PROCESS_DIR):
    os.makedirs(PROCESS_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--sampler',
                    type=str,
                    default='neighbor',
                    choices=["neighbor", "shadow"])
parser.add_argument('--batch_size',
                    type=int,
                    default=1024,)
parser.add_argument('--runs',
                    type=int,
                    default=10,)
parser.add_argument('--device',
                    type=int,
                    default=0,)
parser.add_argument('--data_path',
                    type=str,
                    default='/data/gangda')
parser.add_argument('--dataset',
                    type=str,
                    default='ogbn-products')
parser.add_argument('--edge',
                    action='store_true',
                    help='estimate workload by the number of edges')
args = parser.parse_args()

fanouts = [15, 10, 5]

num_classes, train_idx, g = get_data(args.dataset, args.data_path)

if args.sampler.lower() == 'neighbor':
    sampler = NeighborSampler(
        fanouts,
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

avg_traversed = []
if args.edge:
    for run in range(args.runs):
        num_traversed_edges = []
        for nid, (input_nodes, output_nodes, blocks) in tqdm(enumerate(loader)):
            num_edges = 0
            for block in blocks:
                num_edges += block.num_edges()
            num_traversed_edges.append(num_edges)
        avg_traversed.append(torch.tensor(num_traversed_edges))
else:
    for run in range(args.runs):
        num_traversed_edges = []
        for nid, (input_nodes, output_nodes, blocks) in tqdm(enumerate(loader)):
            X = torch.ones(input_nodes.shape, dtype=torch.float).to(args.device)
            if hasattr(blocks, '__len__'):
                for block in blocks:
                    A = dglsp.spmatrix(torch.stack(block.edges()),
                                       shape=(block.num_src_nodes(), block.num_dst_nodes()))
                    X = A.T @ X
            else:
                for i in range(len(fanouts)):
                    A = dglsp.spmatrix(torch.stack(blocks.edges()),
                                       shape=(blocks.num_src_nodes(), blocks.num_dst_nodes()))
                    X = A.T @ X
                X = X[:output_nodes.shape[0]]
            num_traversed_edges.append(X)
        avg_traversed.append(torch.cat(num_traversed_edges))

avg_traversed = torch.stack(avg_traversed).float().mean(dim=0)
file_name = '{}_{}_{}_{}'.format(args.dataset, args.sampler, str(fanouts), args.batch_size) + \
            ('_edges' if args.edge else '_matrix')
torch.save(avg_traversed.cpu(), PROCESS_DIR+"{}.pt".format(file_name))
