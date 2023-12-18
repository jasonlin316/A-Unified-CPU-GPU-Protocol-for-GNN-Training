import argparse
import math

import dgl
import numpy as np
import torch
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset
from ogb.lsc import MAG240MDataset
from ogb.nodeproppred import DglNodePropPredDataset

from load_mag_to_shm import fetch_mag_from_shm
from load_papers_to_shm import fetch_papers_from_shm

import torch.distributed as dist
from dgl.multiprocessing import call_once_and_share
from dgl.dataloading.dataloader import _divide_by_worker


def get_data(name, data_path):
    if name == 'mag240M':
        dataset = MAG240MDataset(root=data_path+'/HiPC')
        print('Start Loading Graph Structure')
        (g,), _ = dgl.load_graphs(data_path+'/HiPC/graph.dgl')
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
    else:
        dataset = YelpDataset(raw_dir=data_path)
        g = dataset[0]
        train_idx = g.ndata['train_mask'].nonzero().view(-1)
        return dataset.num_classes, train_idx, g
    """
    Note 1: This func avoid creating certain graph formats in each sub-process to save memory
    Note 2: This func will init CUDA. It is not possible to use CUDA in a child process 
            created by fork(), if CUDA has been initialized in the parent process. 
    """
    # g.create_formats_()

    return dataset.num_classes, dataset.train_idx, g


class UnevenDDPIndices(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    This class additionally saves the index tensor in shared memory and therefore
    avoids duplicating the same index tensor during shuffling.
    """

    def __init__(self, indices, total_batch_size, sub_batch_sizes, drop_last, shuffle,
                 indices_workload, args):
        self.rank = dist.get_rank()
        self.seed = 0
        self.epoch = 0
        self._mapping_keys = None
        self.drop_last = drop_last
        self._shuffle = shuffle

        self.total_batch_size = total_batch_size
        self.sub_batch_sizes = sub_batch_sizes
        self.workload = indices_workload
        self.s_wl = torch.cumsum(indices_workload.sort().values, dim=0)
        self.args = args

        # batch size
        self.prefix_sum_batch_size = [sum(sub_batch_sizes[:j + 1]) for j in range(len(sub_batch_sizes))]
        self.prefix_sum_batch_size.insert(0, 0)
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
        # if self.args.dataset == 'ogbn-products':
        #     # dynamic_threshold = 720
        #     threshold = self.s_wl[int(self.args.cpu_gpu_ratio*self.s_wl.shape[0])]
        #     skip_threshold = 405000
        # elif self.args.dataset == 'ogbn-papers100M':
        #     threshold = 0.001
        #     skip_threshold = 100
        # else:
        #     threshold = 1
        #     skip_threshold = 1

        if self.args.batch_type == 'none' or self.args.cpu_process == 0 or self.args.gpu_process == 0:
            batch_sizes = [self.batch_size] * self.num_batches
            start = self.prefix_sum_batch_size[self.rank] * self.num_batches
            end = start + self.batch_size * self.num_batches
            indices = self._indices[start:end]
            indices = _divide_by_worker(indices, self.batch_size, self.drop_last)
        else:
            skip_workload = self.args.cpu_process
            idx = int(self.args.cpu_gpu_ratio * self.s_wl.shape[0])
            dynamic_threshold = (self.s_wl[
                                     idx] / idx) * self.total_batch_size * self.args.cpu_gpu_ratio
            idx2 = int((self.args.cpu_gpu_ratio + self.args.skip_delta) * self.s_wl.shape[0])
            skip_threshold = (self.s_wl[
                                  idx2] / idx2) * self.total_batch_size * self.args.cpu_gpu_ratio

            indices, batch_sizes = [], []
            worker_info = torch.utils.data.get_worker_info()
            for b_id in range(self.num_batches):
                batch_index = self._indices[b_id * self.total_batch_size:
                                            (b_id + 1) * self.total_batch_size]

                w_val, w_idx = self.workload[batch_index].sort()
                batch_index = batch_index[w_idx]
                # w_val = self.workload[batch_index]

                if 'dynamic' in self.args.batch_type:
                    w_val = torch.cumsum(w_val, dim=0)
                    pivot = torch.searchsorted(w_val, dynamic_threshold, side='right')
                    if 'hard' in self.args.batch_type:
                        pivot = min(pivot, self.prefix_sum_batch_size[self.args.cpu_process])
                else:
                    pivot = self.prefix_sum_batch_size[self.args.cpu_process]
                    if self.args.batch_type == 'skip':
                        if w_val[:pivot].sum() > skip_threshold:
                            pivot = skip_workload
                if self.rank < self.args.cpu_process:
                    partial_batch_index = batch_index[:pivot]
                    idx = partial_batch_index[self.rank::self.args.cpu_process]
                else:
                    partial_batch_index = batch_index[pivot:]
                    idx = partial_batch_index[self.rank - self.args.cpu_process::self.args.gpu_process]

                if not worker_info or b_id % worker_info.num_workers == worker_info.id:
                    indices.append(idx)
                    batch_sizes.append(len(idx))
            indices = torch.cat(indices)

            print(self.rank, len(indices), sum(batch_sizes),
                  sum([bs == skip_workload for bs in batch_sizes]))

        id_tensor = self._id_tensor[indices]
        return DynamicTensorizedDatasetIter(
            id_tensor, batch_sizes, self.drop_last, self._mapping_keys, self._shuffle)

    def __len__(self):
        return self.total_size


@DeprecationWarning
def divide_by_worker(dataset, batch_sizes):
    num_samples = dataset.shape[0]
    worker_info = torch.utils.data.get_worker_info()
    if worker_info:
        # num_batches = (
        #     num_samples + (0 if drop_last else batch_size - 1)
        # ) // batch_size
        num_batches = len(batch_sizes)
        num_batches_per_worker = num_batches // worker_info.num_workers
        left_over = num_batches % worker_info.num_workers
        start = (num_batches_per_worker * worker_info.id) + min(
            left_over, worker_info.id
        )
        end = start + num_batches_per_worker + (worker_info.id < left_over)

        start_idx = sum(batch_sizes[:start])
        end_idx = min(sum(batch_sizes[:end]), num_samples)
        # start *= batch_size
        # end = min(end * batch_size, num_samples)
        dataset = dataset[start_idx:end_idx]
        batch_sizes = batch_sizes[start:end]
    return dataset, batch_sizes


class DynamicTensorizedDatasetIter(object):
    def __init__(self, dataset, batch_sizes, drop_last, mapping_keys, shuffle):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.drop_last = drop_last
        self.mapping_keys = mapping_keys
        self.index = 0
        self.shuffle = shuffle

        self.num_batches = len(batch_sizes)

    # For PyTorch Lightning compatibility
    def __iter__(self):
        return self

    def _next_indices(self):
        # if dist.get_rank() == 0:
        #     print('Worker ID', torch.utils.data.get_worker_info().id, self.index, self.dataset.shape[0])

        num_items = self.dataset.shape[0]
        # if self.index >= num_items:
        if len(self.batch_sizes) == 0:
            raise StopIteration
        batch_size = self.batch_sizes.pop(0)

        # if dist.get_rank() == 0:
        #     print('Worker ID', torch.utils.data.get_worker_info().id,
        #           self.num_batches-len(self.batch_sizes), batch_size)

        end_idx = self.index + batch_size
        # print(dist.get_rank(), 'next start', num_items, self.index, end_idx)
        if end_idx > num_items:
            if self.drop_last:
                raise StopIteration
            end_idx = num_items
        batch = self.dataset[self.index: end_idx]
        self.index += batch_size
        # if dist.get_rank() == 0:
        #     print('Worker ID', torch.utils.data.get_worker_info().id, 'next end')
        return batch

    def __next__(self):
        batch = self._next_indices()
        if self.mapping_keys is None:
            return batch.clone()

        # convert the type-ID pairs to dictionary
        type_ids = batch[:, 0]
        indices = batch[:, 1]
        _, type_ids_sortidx = torch.sort(type_ids, stable=True)
        type_ids = type_ids[type_ids_sortidx]
        indices = indices[type_ids_sortidx]
        type_id_uniq, type_id_count = torch.unique_consecutive(
            type_ids, return_counts=True
        )
        type_id_uniq = type_id_uniq.tolist()
        type_id_offset = type_id_count.cumsum(0).tolist()
        type_id_offset.insert(0, 0)
        id_dict = {
            self.mapping_keys[type_id_uniq[i]]: indices[
                type_id_offset[i]: type_id_offset[i + 1]
            ].clone()
            for i in range(len(type_id_uniq))
        }
        return id_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products')
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/gangda')
    args = parser.parse_args()
    get_data(args.dataset, args.data_path)
