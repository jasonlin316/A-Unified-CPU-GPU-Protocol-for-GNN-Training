import argparse
import math
import os
from contextlib import nullcontext

import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch import SAGEConv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler, ShaDowKHopSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch.profiler import profile, record_function, ProfilerActivity
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel

from data import get_data, UnevenDDPTensorizedDataset
from layer import SAGEConv2
from load_mag_to_shm import fetch_mag_from_shm
from load_papers_to_shm import fetch_papers_from_shm
from manager import ResourceManager
from utils import merge_trace_files

TRACE_NAME = 'mixture_product_{}.json'
OUTPUT_TRACE_NAME = "profile/combine.json"


class GNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, model_name='sage'):
        super().__init__()
        self.layers = nn.ModuleList()

        # GraphSAGE-mean
        if model_name.lower() == 'sage':
            self.layers.append(SAGEConv(in_size, hid_size, "mean"))
            for i in range(num_layers - 2):
                self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        # GCN
        elif model_name.lower() == 'gcn':
            kwargs = {'norm': 'both', 'weight': True, 'bias': True, 'allow_zero_in_degree': True}
            self.layers.append(dglnn.GraphConv(in_size, hid_size, **kwargs))
            for i in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, **kwargs))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, **kwargs))
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        if hasattr(blocks, '__len__'):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    # h = self.dropout(h)
        else:
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    # h = self.dropout(h)
        return h


def is_cpu_proc(num_cpu_proc, rank=None):
    if rank is None:
        rank = dist.get_rank()
    return rank < num_cpu_proc


def get_subbatch_size(args, rank=None, cpu_gpu_ratio=None) -> int:
    if rank is None:
        rank = dist.get_rank()
    if cpu_gpu_ratio is None:
        cpu_gpu_ratio = args.cpu_gpu_ratio
    world_size = dist.get_world_size()
    cpu_batch_size = int(args.batch_size * cpu_gpu_ratio)
    if is_cpu_proc(args.cpu_process, rank):
        return cpu_batch_size // args.cpu_process + \
            (cpu_batch_size % args.cpu_process if rank == args.cpu_process - 1 else 0)
    else:
        return (args.batch_size - cpu_batch_size) // args.gpu_process + \
            ((args.batch_size - cpu_batch_size) % args.gpu_process if rank == world_size - 1 else 0)


def device_mapping(num_cpu_proc):
    assert not is_cpu_proc(num_cpu_proc), "For GPU Comp process only"
    return dist.get_rank() - num_cpu_proc


def get_device(args):
    device = torch.device("cpu" if is_cpu_proc(args.cpu_process)
                          else "cuda:{}".format(device_mapping(args.cpu_process)))
    if not is_cpu_proc(args.cpu_process):
        torch.cuda.set_device(device)
    return device


def _train(loader, model, opt, **kwargs):
    model.train()
    total_loss = 0
    if kwargs['rank'] == 0:
        pbar = tqdm(total=kwargs['train_size'])
        epoch = kwargs['epoch']
        pbar.set_description(f'Epoch {epoch:02d}')

    process = kwargs['process']
    device = torch.device("cpu" if is_cpu_proc(process)
                          else "cuda:{}".format(device_mapping(process)))

    dist.barrier()
    tik_init = time.time()

    for it, (input_nodes, output_nodes, blocks) in enumerate(loader):

        # loader_iter = iter(loader)
        # for it in range(kwargs['num_batches']):
        #     if kwargs['CPU_SUBGRAPH_CACHE'][it] is not None and is_cpu_proc(process):
        #         input_nodes, output_nodes, blocks = kwargs['CPU_SUBGRAPH_CACHE'][it]
        #     else:
        #         input_nodes, output_nodes, blocks = next(loader_iter)
        #         if is_cpu_proc(process):
        #             kwargs['CPU_SUBGRAPH_CACHE'][it] = (input_nodes, output_nodes, blocks)

        # print(dist.get_rank(), 1, output_nodes.shape[0])

        if it == 1:
            dist.barrier()
            loader_init_time = time.time() - tik_init
            # if kwargs['rank'] == 0:
            #     print(f'Loader Init: {loader_init_time:.4f}s')
        if it == kwargs['num_batches'] - 1:
            dist.barrier()
            tik_end = time.time()

        # print(dist.get_rank(), output_nodes.shape[0])
        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print()

        # if it + 1 == 50: break
        if hasattr(blocks, '__len__'):
            x = blocks[0].srcdata["feat"].to(torch.float32)
            y = blocks[-1].dstdata["label"]
        else:
            x = blocks.srcdata["feat"].to(torch.float32)
            y = blocks.dstdata["label"]
        # y_hat = model(blocks, x)
        if kwargs['device'] == "cpu":  # for papers100M
            y = y.type(torch.LongTensor)
            y_hat = model(blocks, x)
        else:
            y = y.type(torch.LongTensor).to(device)
            y_hat = model(blocks, x).to(device)
        loss = F.cross_entropy(y_hat[:output_nodes.shape[0]],
                               y[:output_nodes.shape[0]])

        # print(dist.get_rank(), 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # print(dist.get_rank(), 3)

        del input_nodes, output_nodes, blocks
        torch.cuda.empty_cache()

        total_loss += loss.item()  # avoid cuda memory accumulation
        if kwargs['rank'] == 0:
            pbar.update(kwargs['batch_size'])

        if kwargs['prof']:
            kwargs['prof'].step()

    dist.barrier()
    loader_close_time = time.time() - tik_end

    if kwargs['rank'] == 0:
        pbar.close()
        print(f'Loader Init Time: {loader_init_time:.4f}s')
        print(f'Loader Close Time: {loader_close_time:.4f}s')
    return total_loss, loader_init_time, loader_close_time


def hybrid_train(args, config, func, params):
    # for log only
    rank = params['rank']
    epoch = params['epoch']
    num_batches = params['num_batches']
    loader: DataLoader = params['loader']

    # update cpu_gpu_ratio
    # sub_batch_sizes = [get_subbatch_size(args, r, config['cpu_gpu_ratio'])
    #                    for r in range(dist.get_world_size())]
    # loader.indices.update_batch_size(sub_batch_sizes)
    # if rank == 0:
    #     print(f'\nEpoch {epoch}, CPU/GPU workload ratio {config["cpu_gpu_ratio"]:.3f}')
    #     print('SubBatch sizes:', sub_batch_sizes, '\n')

    # start training
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # schedule=torch.profiler.schedule(
            #     skip_first=8,
            #     wait=4,
            #     warmup=1,
            #     active=3,
            #     repeat=5
            # ),
    ) if args.log else nullcontext() as prof:
        params['prof'] = prof
        _tik = time.time()
        cpu_runtime = gpu_runtime = 0
        if is_cpu_proc(args.cpu_process):
            with loader.enable_cpu_affinity(loader_cores=config['load_core'],
                                            compute_cores=config['comp_core']):
                loss, loader_init_time, loader_close_time = func(**params)
                cpu_runtime = time.time() - _tik
        else:
            loss, loader_init_time, loader_close_time = func(**params)
            gpu_runtime = time.time() - _tik
        if rank == 0:
            print(f'\nTraining loss: {loss / num_batches:.4f}')
            total_epoch_time = time.time() - _tik
            print(f'Total Epoch Time: {total_epoch_time:.3f}s')
            print(f'Epoch Time w/o loader overhead:'
                  f' {total_epoch_time - loader_init_time - loader_close_time:.3f}s\n')
    if epoch == 2 and prof:
        prof.export_chrome_trace(TRACE_NAME.format(rank))
    return prof, cpu_runtime, gpu_runtime


def train(rank, world_size, args):
    num_classes, train_idx, g = get_data(args.dataset, args.data_path)

    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    device = get_device(args)
    model = GNN(g.ndata["feat"].size(-1), 128, num_classes, args.layer, model_name=args.model)
    model = model.to(device)
    model = DistributedDataParallel(model)

    # loader config
    drop_last, shuffle = True, False
    fanouts = [15, 10, 5]
    wl_batch = 1024  # 196615
    # fanouts = [30, 20, 10]
    # fanouts = [5, 10, 15]

    sub_batch_sizes = [get_subbatch_size(args, r) for r in range(world_size)]

    # workload = torch.load('processed/ogbn-products_neighbor_[30, 20, 10].pt')
    workload = torch.load('processed/{}_neighbor_{}_{}_matrix.pt'.format(
        args.dataset, fanouts, wl_batch))
    # if args.cpu_gpu_ratio > 0:
    #     train_idx = train_idx[workload.sort().indices]
    # num_batches = len(train_idx) // args.batch_size
    # bsizes_psum = [sum(sub_batch_sizes[:j + 1]) for j in range(len(sub_batch_sizes))]
    # bsizes_psum.append(0)
    # if rank < args.cpu_process:
    #     train_indices = train_idx[:num_batches * bsizes_psum[args.cpu_process - 1]]
    #     train_indices = train_indices[rank::args.cpu_process]
    # else:
    #     train_indices = train_idx[
    #                     num_batches * bsizes_psum[rank - 1]: num_batches * bsizes_psum[rank]]
    # print(rank, train_indices, len(train_indices))

    train_indices = UnevenDDPTensorizedDataset(
        train_idx.to(device),
        args.batch_size,
        sub_batch_sizes,
        drop_last,
        shuffle,
        workload,
        args
    )
    if args.sampler.lower() == 'neighbor':
        sampler = NeighborSampler(
            fanouts,
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
        assert len(sampler.fanouts) == args.layer
    elif args.sampler.lower() == 'shadow':
        sampler = ShaDowKHopSampler(  # CPU sampling is 2x faster than GPU sampling
            fanouts,
            output_device=device,  # comment out in CPU sampling version
            prefetch_node_feats=["feat"],
        )
    else:
        raise NotImplementedError
    
    # training loop
    manager = ResourceManager(args, is_cpu_proc(args.cpu_process))
    params = {
        # training
        'model': model,
        'opt': torch.optim.Adam(model.parameters(), lr=1e-3),
        # logging
        'rank': rank,
        'train_size': len(train_indices),
        # 'train_size': args.batch_size * num_batches,
        'num_batches': train_indices.num_batches,
        # 'num_batches': num_batches,
        'batch_size': args.batch_size,
        'device': device,
        'process': args.cpu_process,
        'epoch': 0,
        # 'CPU_SUBGRAPH_CACHE': [None for _ in range(num_batches)],  # init cache
    }

    for epoch in range(10):
        conf = manager.config()
        if rank == 0:
            print()
            print(conf)
            print()
        train_loader = DataLoader(
            g,
            train_indices,
            sampler,
            # batch_size=sub_batch_sizes[rank],
            device=device,
            use_uva=device.type == 'cuda',
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=len(conf['load_core']),
        )

        params['epoch'] = epoch
        params['loader'] = train_loader

        prof = hybrid_train(args, manager.config(), _train, params)
        # manager.update(prof)  # comment on this line to disable Resource Manager

        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/gangda')
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products", "mag240M"])
    parser.add_argument("--cpu_process",
                        type=int,
                        default=2,
                        choices=[0, 1, 2, 4])
    parser.add_argument("--gpu_process",
                        type=int,
                        default=1)
    parser.add_argument("--cpu_gpu_ratio",
                        type=float,
                        default=0.15)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024 * 4)
    parser.add_argument('--sampler',
                        type=str,
                        default='neighbor',
                        choices=["neighbor", "shadow"])
    parser.add_argument('--model',
                        type=str,
                        default='sage',
                        choices=["sage", "gcn"])
    parser.add_argument('--layer',
                        type=int,
                        default=3)
    parser.add_argument('--batch_type',
                        type=str,
                        default='none',
                        choices=['none', 'static', 'dynamic', 'dynamic_hard', 'skip'])
    parser.add_argument('--log', action='store_true')
    arguments = parser.parse_args()

    # download dataset if not
    # DglNodePropPredDataset(arguments.dataset, arguments.data_path)

    # Assure Consistency
    if arguments.cpu_gpu_ratio == 0 or arguments.cpu_process == 0:
        arguments.cpu_gpu_ratio = 0
        arguments.cpu_process = 0
    if arguments.cpu_gpu_ratio == 1 or arguments.gpu_process == 0:
        arguments.cpu_gpu_ratio = 1
        arguments.gpu_process = 0
    nprocs = arguments.cpu_process + arguments.gpu_process
    assert nprocs > 0
    print(f'\nUse {arguments.cpu_process} CPU Comp processes and {arguments.gpu_process} GPUs\n'
          f'The batch size is {arguments.batch_size} with {arguments.cpu_gpu_ratio} cpu/gpu workload ratio\n'
          f'Sampler: {arguments.sampler}, Model: {arguments.model}, Layer: {arguments.layer}\n')

    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29503'

    mp.set_start_method('fork')
    processes = []
    for i in range(nprocs):
        p = dmp.Process(target=train, args=(i, nprocs, arguments))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    input_files = [TRACE_NAME.format(i) for i in range(nprocs)]
    merge_trace_files(input_files, OUTPUT_TRACE_NAME)
    for i in range(nprocs):
        os.remove(TRACE_NAME.format(i))

    print("Program finished")
