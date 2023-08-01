import argparse
import os

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

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
        """Conduct layer-wise inference to get all the node embeddings."""
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


def train(rank, size, args, device, g, dataset, model):
    # create sampler & dataloader
    dist.init_process_group('gloo', rank=rank, world_size=size)
    model = DistributedDataParallel(model)
    n = psutil.cpu_count(logical=False)

    load_core, comp_core = [], []
    if size == 1:
        load_core = list(range(0, 4))
        comp_core = list(range(4, n))

    elif size == 2:
        if rank == 0:
            load_core = list(range(0, 8))
            comp_core = list(range(8, n // 2))
        else:
            load_core = list(range(n // 2, n // 2 + 8))
            comp_core = list(range(n // 2 + 8, n))

    elif size == 4:
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

    train_idx = dataset.train_idx.to(device)

    sampler = NeighborSampler(
        [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )

    batch_size = 4096

    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=(batch_size // size),
        use_ddp=True,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(1):
        model.train()
        total_loss = 0
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with train_dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core):
                if rank == 0:
                    pbar = tqdm(total=int(train_idx.shape[0] - train_idx.shape[0] % batch_size))
                    pbar.set_description(f'Epoch {epoch:02d}')
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                    x = blocks[0].srcdata["feat"]
                    y = blocks[-1].dstdata["label"]
                    y_hat = model(blocks, x)
                    loss = F.cross_entropy(y_hat, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    if rank == 0:
                        pbar.update(batch_size)
                if rank == 0:
                    pbar.close()
                time.sleep(1)

        prof.export_chrome_trace('product_three_layer.json')
        # if rank == 0:
        #     prof.export_chrome_trace('product_three_layer_0.json')
        # elif rank == 1:
        #     prof.export_chrome_trace('product_three_layer_1.json')
        # elif rank == 2:
        #     prof.export_chrome_trace('product_three_layer_2.json')
        # else:
        #     prof.export_chrome_trace('product_three_layer_3.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='')
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
             "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--process",
        default="1",
        choices=["1", "2", "4"],
    )

    args = parser.parse_args()
    args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    size = 1
    if args.process == "2":
        size = 2
    elif args.process == "4":
        size = 4

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", args.data_path))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 128, out_size).to(device)

    # model training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    processes = []

    mp.set_start_method('fork')

    for rank in range(size):
        p = dmp.Process(target=train, args=(rank, size, args, device, g, dataset, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("program finished.")
