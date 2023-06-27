import argparse
import math
import os

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.dataloading.dataloader import _divide_by_worker, _TensorizedDatasetIter
from dgl.multiprocessing import call_once_and_share

from tqdm import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """
        Conduct layer-wise inference to get all the node embeddings.
        TODO: Should be compatible with UVA
        https://github.com/dmlc/dgl/blob/master/examples/pytorch/multigpu/multi_gpu_node_classification.py
        """
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=3
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class UnevenDDPTensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    This class additionally saves the index tensor in shared memory and therefore
    avoids duplicating the same index tensor during shuffling.
    """
    def __init__(self, indices, total_batch_size, sub_batch_sizes, drop_last, shuffle):
        self.rank = dist.get_rank()
        self.seed = 0
        self.epoch = 0
        self._mapping_keys = None
        self.drop_last = drop_last
        self._shuffle = shuffle

        self.prefix_sum_batch_size = sum(sub_batch_sizes[:self.rank])
        self.batch_size = sub_batch_sizes[self.rank]

        len_indices = len(indices)
        if self.drop_last and len_indices % total_batch_size != 0:
            self.num_batches = math.ceil((len_indices - total_batch_size) / total_batch_size)
        else:
            self.num_batches = math.ceil(len_indices / total_batch_size)
        self.total_size = self.num_batches * total_batch_size
        # If drop_last is False, we create a shared memory array larger than the number
        # of indices since we will need to pad it after shuffling to make it evenly
        # divisible before every epoch.  If drop_last is True, we create an array
        # with the same size as the indices so we can trim it later.
        self.shared_mem_size = self.total_size if not self.drop_last else len_indices
        self.num_indices = len_indices

        self._id_tensor = indices
        # self._device = self._id_tensor.device
        self.device = self._id_tensor.device

        self._indices = call_once_and_share(
            self._create_shared_indices, (self.shared_mem_size,), torch.int64)

    def _create_shared_indices(self):
        indices = torch.empty(self.shared_mem_size, dtype=torch.int64)
        num_ids = self._id_tensor.shape[0]
        torch.arange(num_ids, out=indices[:num_ids])
        torch.arange(self.shared_mem_size - num_ids, out=indices[num_ids:])
        return indices

    def shuffle(self):
        """Shuffles the dataset."""
        # Only rank 0 does the actual shuffling.  The other ranks wait for it.
        if self.rank == 0:
            np.random.shuffle(self._indices[:self.num_indices].numpy())
            if not self.drop_last:
                # pad extra
                self._indices[self.num_indices:] = \
                    self._indices[:self.total_size - self.num_indices]
        dist.barrier()

    def __iter__(self):
        start = self.prefix_sum_batch_size * self.num_batches
        end = start + self.batch_size * self.num_batches
        indices = _divide_by_worker(self._indices[start:end], self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices]
        return _TensorizedDatasetIter(
            id_tensor, self.batch_size, self.drop_last, self._mapping_keys, self._shuffle)

    def __len__(self):
        return self.total_size


def evaluate(model, graph, dataloader, load_core, comp_core):
    model.eval()
    ys = []
    y_hats = []
    with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core):
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata["feat"]
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, x))
                accuracy = MulticlassAccuracy(num_classes=47)
    return accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label)


def is_cpu_proc(num_cpu_proc, rank=None):
    if rank is None:
        rank = dist.get_rank()
    return rank < num_cpu_proc


def device_mapping(num_cpu_proc):
    assert not is_cpu_proc(num_cpu_proc), "For GPU Comp process only"
    return dist.get_rank() - num_cpu_proc


def assign_cores(num_cpu_proc):
    assert is_cpu_proc(num_cpu_proc), "For CPU Comp process only"
    rank = dist.get_rank()
    load_core, comp_core = [], []
    n = psutil.cpu_count(logical=False)
    if num_cpu_proc == 1:
        load_core = list(range(0, 4))
        comp_core = list(range(4, n))
    elif num_cpu_proc == 2:
        if rank == 0:
            load_core = list(range(0, 8))
            comp_core = list(range(8, n // 2))
        else:
            load_core = list(range(n // 2, n // 2 + 8))
            comp_core = list(range(n // 2 + 8, n))
    elif num_cpu_proc == 4:
        if rank == 0:
            load_core = list(range(0, 4))
            comp_core = list(range(4, n // 4))
        elif rank == 1:
            load_core = list(range(n // 4, n // 4 + 4))
            comp_core = list(range(n // 4 + 4, n // 2))
        elif rank == 2:
            load_core = list(range(n // 2, n // 2 + 4))
            comp_core = list(range(n // 2 + 4, n // 4 * 3))
        else:
            load_core = list(range(n // 4 * 3, n // 4 * 3 + 4))
            comp_core = list(range(n // 4 * 3 + 4, n))
    return load_core, comp_core


def get_subbatch_size(args, rank=None) -> int:
    if rank is None:
        rank = dist.get_rank()
    world_size = dist.get_world_size()
    cpu_batch_size = int(args.batch_size * args.cpu_gpu_ratio)
    if is_cpu_proc(args.cpu_process, rank):
        return cpu_batch_size // args.cpu_process + \
            (cpu_batch_size % args.cpu_process if rank == args.cpu_process - 1 else 0)
    else:
        return (args.batch_size - cpu_batch_size) // args.gpu_process + \
            ((args.batch_size - cpu_batch_size) % args.gpu_process if rank == world_size - 1 else 0)


def _train(loader, model, opt, **kwargs):
    total_loss = 0
    if kwargs['rank'] == 0:
        pbar = tqdm(total=kwargs['train_size'])
        epoch = kwargs['epoch']
        pbar.set_description(f'Epoch {epoch:02d}')
    for it, (input_nodes, output_nodes, blocks) in enumerate(loader):
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if kwargs['rank'] == 0:
            pbar.update(kwargs['batch_size'])
            # pbar.update(output_nodes.shape[0])
    if kwargs['rank'] == 0:
        pbar.close()
    return total_loss


def _train_cpu(load_core, comp_core, **kwargs):
    with kwargs['loader'].enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core):
        loss = _train(**kwargs)
    return loss


def train(rank, world_size, args, g, data):
    num_classes, train_idx, val_idx, test_idx = data
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    device = torch.device("cpu" if is_cpu_proc(args.cpu_process)
                          else "cuda:{}".format(device_mapping(args.cpu_process)))

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 128, num_classes).to(device)
    model = DistributedDataParallel(model)

    # create loader
    drop_last, shuffle = True, True
    sub_batch_sizes = [get_subbatch_size(args, r) for r in range(world_size)]
    if rank == 0: print('SubBatch sizes:', sub_batch_sizes)
    train_indices = UnevenDDPTensorizedDataset(
        train_idx.to(device),
        args.batch_size,
        sub_batch_sizes,
        drop_last,
        shuffle
    )
    sampler = NeighborSampler(
        [15, 10, 5],
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    train_dataloader = DataLoader(
        g,
        train_indices,  # train_idx.to(device)
        sampler,
        device=device,
        batch_size=get_subbatch_size(args),
        use_ddp=True,
        use_uva=not is_cpu_proc(args.cpu_process),
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=4 if is_cpu_proc(args.cpu_process) else 0,
    )

    # training loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    params = {
        # training
        'loader': train_dataloader,
        'model': model,
        'opt': opt,
        # logging
        'rank': rank,
        'train_size': len(train_indices),
        'batch_size': args.batch_size,
    }
    for epoch in range(2):
        params['epoch'] = epoch
        model.train()
        with profile(
                activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA
                ],
                record_shapes=True
        ) as prof:
            if is_cpu_proc(args.cpu_process):
                if params.get('load_core', None) is None or params.get('comp_core', None):
                    params['load_core'], params['comp_core'] = assign_cores(args.cpu_process)
                loss = _train_cpu(**params)
            else:
                loss = _train(**params)
        if epoch == 1 and rank == args.cpu_process:
            prof.export_chrome_trace('mixture_product.json')
        dist.barrier()

        # TODO: val or test
        if rank == 0:
            print(f'Training loss: {loss/train_indices.num_batches:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/gangda/dgl')
    parser.add_argument("--cpu_process",
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 4])
    parser.add_argument("--gpu_process",
                        type=int,
                        default=1,
                        choices=[0, 1])
    parser.add_argument("--cpu_gpu_ratio",
                        type=float,
                        default=0.3)
    parser.add_argument("--batch_size",
                        type=int,
                        default=4096)
    arguments = parser.parse_args()

    # set num processes
    comp_proc_size = arguments.cpu_process + arguments.gpu_process
    assert comp_proc_size > 0
    if arguments.cpu_process == 0:
        arguments.cpu_gpu_ratio = 0
    if arguments.gpu_process == 0:
        arguments.cpu_gpu_ratio = 1
    print(f'Use {arguments.cpu_process} CPU Comp processes and {arguments.gpu_process} GPUs\n'
          f'The batch size is {arguments.batch_size} with {arguments.cpu_gpu_ratio} cpu/gpu workload ratio')

    # load and preprocess dataset
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", arguments.data_path))
    g = dataset[0]
    # avoid creating certain graph formats in each sub-process to save memory
    g.create_formats_()
    data = (
        dataset.num_classes,
        dataset.train_idx,
        dataset.val_idx,
        dataset.test_idx,
    )
    print("Data loading finished")

    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.set_start_method('fork')
    processes = []
    for i in range(comp_proc_size):
        p = dmp.Process(target=train, args=(i, comp_proc_size, arguments, g, data))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Program finished")
