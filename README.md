## Setup

1. Setup a Python environment and Install Deep Graph Library (DGL)
2. Install TCMalloc:  https://google.github.io/tcmalloc/quickstart.html  
Note: need to install Bazel first (also mentioned in the link above).   
Note 2: Alternative approach to install TCMalloc: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10117
3. After installation, preload tcmalloc when running your code. For example:   
``` LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.3 python main.py```


## Hybrid GNN Training
### Usage
  ```
  python main.py --cpu_process 4 --gpu_process 1 --cpu_gpu_ratio 0.3 --dataset ogbn-products --sampler shadow --model sage --layer 5
  ```
  Important Arguments: 
  - `--cpu_process`: Number of CPU computing processes used in training. Available choices [0, 1, 2, 4]
  - `--gpu_process`: Number of GPU computing processes (devices) used in training. Available choices [0, 1]
  - `--cpu_gpu_ratio`: Workload ratio between CPU and GPU, computed by
    `(# of batch nodes assigned to all CPU processes) / (# of batch nodes assigned to all GPU processes)`
  - `--dataset`: the training datasets. Available choices [ogbn-products, ogbn-papers100M, mag240M]
  - `--sampler`: the mini-batch sampling algorithm. Available choices [shadow, neighbor]
  - `--model`: GNN model. Available choices [gcn, sage]
  - `--layer`: number of GNN layers.
  
  Hint: arguments `--cpu_process` and `--gpu_process` can be set to 0 for baseline comparison.  
  Note: while we only test our library using three datasets, two samplers, and two types of GNN model, other setups should also work as our library is compatible with DGL. Please refer to the [DGL document](https://docs.dgl.ai) for a full list of available sampler, GNN models, etc.  
  Note 2: large memory space (512 GB or above) is highly recommended. 

### Code Explanation

To ensure load balancing, we need to be able to assign different batch sizes to the CPUs and the GPUs.  
We replace the [DDPTensorizedDataset](https://github.com/dmlc/dgl/blob/7b1639f1aacb006fa65ef8cef09c49f5219bd5c1/python/dgl/dataloading/dataloader.py#L252)
class in DGL with a new class [UnevenDDPTensorizedDataset](https://github.com/jasonlin316/HiPC23/blob/main/main.py#L95). 
This class splits the indices based on the `sub_batch_sizes` array while including all other features in the previous class.
For batch size assignment, the logic is written in function 
[get_subbatch_size()](https://github.com/jasonlin316/HiPC23/blob/main/main.py#L230).

The two-level resource manager is implemented in ```manager.py```.


## Specific Instruction for Training on MAG240M
### Load Feature Matrix to Shared Memory
MAG240M dataset has a large feature matrix (380G). To reduce the memory consumption and feature loading time,
we first load the required data into the shared memory, then conduct one or more trials at the same time.
  ```
  python load_mag_to_shm.py
  ```
Note that `load_mag_to_shm.py` requires a large amount of available memory (roughly 800G) when copying data
from disk to shared memory, and consumes a smaller amount of memory (roughly 400G) after movement completes.
Be sure to spare enough memory during the data movement.
