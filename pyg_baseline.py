import copy
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3"

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import sklearn
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all: Tensor, device: torch.device,
                  subgraph_loader: NeighborLoader) -> Tensor:

        pbar = tqdm(total=len(subgraph_loader) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.node_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                x = x[:batch.batch_size]
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x.cpu())
                pbar.update(1)
            x_all = torch.cat(xs, dim=0)

        pbar.close()
        return x_all


def run(rank, world_size, dataset, batch_size, cpu_gpu_ratio):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    device = torch.device('cuda:{}'.format(rank) if rank != 0 else 'cpu')

    data = dataset[0]
    data = data.to(device, 'x', 'y')  # Move to device for faster feature fetch.

    # Split training indices into `world_size` many chunks:
    # train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = dataset.get_idx_split()['train']
    # train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    pivot = int(train_idx.size(0) * cpu_gpu_ratio)
    if rank == 0:
        train_idx = train_idx[:pivot]
        batch_size = int(batch_size * cpu_gpu_ratio)
    else:
        train_idx = train_idx[pivot:]
        batch_size = batch_size - int(batch_size * cpu_gpu_ratio)

    print(rank, train_idx.size(0), batch_size)

    kwargs = dict(batch_size=batch_size, num_workers=4, persistent_workers=True)
    train_loader = NeighborLoader(data, input_nodes=train_idx,
                                  num_neighbors=[25, 10], shuffle=True,
                                  drop_last=True, **kwargs)

    if rank == 0:  # Create single-hop evaluation neighbor loader:
        subgraph_loader = NeighborLoader(copy.copy(data), num_neighbors=[-1],
                                         shuffle=False, **kwargs)
        # No need to maintain these features during evaluation:
        del subgraph_loader.data.x, subgraph_loader.data.y
        # Add global node index information:
        subgraph_loader.data.node_id = torch.arange(data.num_nodes)

    torch.manual_seed(12345)

    model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    if rank == 0:
        model = DistributedDataParallel(model)
        # model = DistributedDataParallel(model, device_ids=[rank])
    else:
        model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print('Group init finished', rank)

    for epoch in range(1, 21):
        model.train()
        if rank == 0:
            pbar = tqdm(total=int(len(train_loader.dataset)))
            pbar.set_description(f'Epoch {epoch:02d}')
        for batch in train_loader:

            tik = time.time()

            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size].squeeze())

            print(f'Model Forward: {time.time() - tik:.3f}, Rank: {rank}')
            tik = time.time()

            loss.backward()

            print(f'Backward with Gradient Sync: {time.time() - tik:.3f}, Rank: {rank}')
            tik = time.time()

            optimizer.step()

            print(f'Optimizer: {time.time() - tik:.3f}, Rank: {rank}')

            if rank == 0:
                pbar.update(batch.batch_size)
        if rank == 0:
            pbar.close()

        dist.barrier()

        if rank == 0:
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        if rank == 0 and epoch % 5 == 0:  # We evaluate on a single GPU for now
            model.eval()
            with torch.no_grad():
                out = model.module.inference(data.x, device, subgraph_loader)
            res = out.argmax(dim=-1) == data.y.to(out.device)
            acc1 = int(res[data.train_mask].sum()) / int(data.train_mask.sum())
            acc2 = int(res[data.val_mask].sum()) / int(data.val_mask.sum())
            acc3 = int(res[data.test_mask].sum()) / int(data.test_mask.sum())
            print(f'Train: {acc1:.4f}, Val: {acc2:.4f}, Test: {acc3:.4f}')

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    # dataset = Reddit('/data/gangda/pyg/Reddit2')
    dataset = PygNodePropPredDataset('ogbn-products', root='/data/gangda/ogb', )


    batch_size = 204
    cpu_gpu_ratio = 1
    world_size = 1
    # world_size = torch.cuda.device_count()

    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset, batch_size, cpu_gpu_ratio), nprocs=world_size, join=True)
