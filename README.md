## Instruction

Install TCMalloc:  https://google.github.io/tcmalloc/quickstart.html  
Note: need to install Bazel first (also mentioned in the link above).   
Note 2: Alternative approach to install TCMalloc: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10117

After installation, preload tcmalloc first when running your code. For example:   
``` LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.3 python main.py```


## Hybrid GNN Training
### Usage
  ```
  python main.py --cpu_process 4 --gpu_process 1 --cpu_gpu_ratio 0.3
  ```
  Important Arguments: 
  - `--cpu_process`: Number of CPU computing processes used in training. Available choices [0, 1, 2, 4]
  - `--gpu_process`: Number of GPU computing processes (devices) used in training. Available choices [0, 1]
  - `--cpu_gpu_ratio`: Workload ratio between CPU and GPU, computed by
    `(# of batch nodes assigned to all CPU processes) / (# of batch nodes assigned to all GPU processes)`
  
  Hint: arguments `--cpu_process` and `--gpu_process` can be set to 0 for baseline comparison.

### Code Explanation

For process assignment to target hardware, the logic is written in func 
[is_cpu_proc](https://github.com/jasonlin316/HiPC23/blob/main/main.py#L188), which is relied upon by many other funcs.
A process is considered a CPU process only if its rank is smaller than the number of CPU processes. 
Otherwise, it is considered a GPU process.
Modify this function (also [device_mapping](https://github.com/jasonlin316/HiPC23/blob/main/main.py#L194))
to easily customize the process-assignment rules.

For batch size assignment, the logic is written in func 
[get_subbatch_size()](https://github.com/jasonlin316/HiPC23/blob/main/main.py#L230).
Given that the Dataloader in DGL supports more features than PyG, the impl of uneven batch size is not that 
straightforward. I replace the [DDPTensorizedDataset](https://github.com/dmlc/dgl/blob/7b1639f1aacb006fa65ef8cef09c49f5219bd5c1/python/dgl/dataloading/dataloader.py#L252)
class in DGL with a new class [UnevenDDPTensorizedDataset](https://github.com/jasonlin316/HiPC23/blob/main/main.py#L95). 
This class splits the indices based on the `sub_batch_sizes` array while including all other features in the previous class.
Note that there is no need to modify this class when changing batch size assignment logic.


## Train on MAG240M
### Load Feature Matrix to shm
MAG240M dataset has a large feature matrix (380G). To reduce the memory consumption and feature loading time,
we first load the required data into shared memory, then conduct one or more trials at the same time.
  ```
  python load_mag_to_shm.py
  ```
Note that `load_mag_to_shm.py` requires a large amount of available memory (roughly 800G) when copying data
from disk to shared memory, and consumes a smaller amount of memory (roughly 400G) after movement completes.
Be sure to spare enough memory during the data movement.

### Training Instruction
  ```
  python main.py --cpu_process 2 --cpu_gpu_ratio 0.9 --dataset mag240M --sampler shadow --model sage --layer 5
  ```
For shadow sampler, reduce the neighbor budget can boost cpu speed relative to gpu。 