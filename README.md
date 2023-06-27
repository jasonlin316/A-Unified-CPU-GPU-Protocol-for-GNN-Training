## Instruction

Install TCMalloc:  https://google.github.io/tcmalloc/quickstart.html  
Note: need to install Bazel first (also mentioned in the link above).   

After installation, preload tcmalloc first when running your code. For example:   
``` LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.3 python training.py```

Some sample codes I ran using DDP can be found here: https://github.com/jasonlin316/DDP_GNN  


## Hybrid GNN Training
### Usage
  ```
  python mixture.py --cpu_process 4 --gpu_process 1 --cpu_gpu_ratio 0.3
  ```
  Important Arguments: 
  - `--cpu_process`: Number of CPU computing processes used in training. Available choices [0, 1, 2, 4]
  - `--gpu_process`: Number of GPU computing processes (devices) used in training. Available choices [0, 1]
  - `--cpu_gpu_ratio`: Workload ratio between CPU and GPU, computed by
    `(# of batch nodes assigned to all CPU processes) / (# of batch nodes assigned to all GPU processes)`
  
  Hint: arguments `--cpu_process` and `--gpu_process` can be set to 0 for baseline comparison.

### Code Explanation

For process assignment to target hardware, the logic is written in func 
[is_cpu_proc](https://github.com/jasonlin316/HiPC23/blob/main/mixture.py#L188), which is relied upon by many other funcs.
A process is considered a CPU process only if its rank is smaller than the number of CPU processes. 
Otherwise, it is considered a GPU process.
Modify this function (also [device_mapping](https://github.com/jasonlin316/HiPC23/blob/main/mixture.py#L194))
to easily customize the process-assignment rules.

For batch size assignment, the logic is written in func 
[get_subbatch_size()](https://github.com/jasonlin316/HiPC23/blob/main/mixture.py#L230).
Given that the Dataloader in DGL supports more features than PyG, the impl of uneven batch size is not that 
straightforward. I replace the [DDPTensorizedDataset](https://github.com/dmlc/dgl/blob/7b1639f1aacb006fa65ef8cef09c49f5219bd5c1/python/dgl/dataloading/dataloader.py#L252)
class in DGL with a new class [UnevenDDPTensorizedDataset](https://github.com/jasonlin316/HiPC23/blob/main/mixture.py#L95). 
This class splits the indices based on the `sub_batch_sizes` array while including all other features in the previous class.
Note that there is no need to modify this class when changing batch size assignment logic.
