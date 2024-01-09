## Setup

1. Setup a Python environment (>=3.11). Install PyTorch (>=2.0.1) and Deep Graph Library (>=1.1).
2. Install TCMalloc:  https://google.github.io/tcmalloc/quickstart.html  
Note: need to install Bazel first (also mentioned in the link above).   
Note 2: Alternative approach to install TCMalloc: https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/10117
3. After installation, preload tcmalloc when running your code. For example:   
``` LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.3 python main.py```


## Usage
### 1. Download dataset:  
```
python dataset.py --dataset ogbn-products --data_path your_data_path
```
### 2. (Optional) Load feature matrix into shared memory:  
MAG240M datasets have large feature matrices (380G). To reduce the memory consumption and feature loading time, we first load the required data into the shared memory, then conduct one or more trials at the same time.  
```
python load_mag_to_shm.py --data_path your_data_path
```   
Note that `load_mag_to_shm.py` requires a large amount of available memory (roughly 800G) when copying data from disk to shared memory, and consumes a smaller amount of memory (roughly 400G) after movement completes. Be sure to spare enough memory during the data movement.
### 3. Preprocess training node workload:   
``` 
python workload.py --sampler neighbor --dataset ogbn-products --data_path your_data_path 
```
### 4. Training GNNs on CPUs and GPUs:  
```
python main.py --dataset ogbn-products --data_path your_data_path
               --cpu_process 2 --gpu_process 1
               --batch_type dynamic --cached_ratio 0.2
               --sampler neighbor --model sage
```  
  Important Arguments: 
  - `--dataset`: the training datasets. Available choices [reddit, ogbn-products, mag240M]
  - `--cpu_process`: Number of CPU computing processes used in training. Available choices [0, 1, 2, 4]
  - `--gpu_process`: Number of GPU computing processes (devices) used in training. Available choices [0, 1]
  - `--batch_type`: Strategy of workload assignment. Available choices ['none', 'static', 'dynamic', 'dynamic_hard']
  - `--cached_ratio`: Ratios of node features cached in GPU.
  - `--sampler`: the mini-batch sampling algorithm. Available choices [shadow, neighbor]
  - `--model`: GNN model. Available choices [gcn, sage]
  
  Hint: arguments `--cpu_process` and `--gpu_process` can be set to 0 for baseline comparison.  
  Note: while we only test our library using three datasets, two samplers, and two types of GNN model, other setups should also work as our library is compatible with DGL. Please refer to the [DGL document](https://docs.dgl.ai) for a full list of available sampler, GNN models, etc.  
  Note 2: large memory space (512 GB or above) is highly recommended.
