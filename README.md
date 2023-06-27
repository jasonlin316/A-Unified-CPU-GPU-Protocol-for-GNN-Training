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