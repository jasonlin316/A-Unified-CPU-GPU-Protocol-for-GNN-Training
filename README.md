Install TCMalloc:  https://google.github.io/tcmalloc/quickstart.html  
Note: need to install Bazel first (also mentioned in the link above).   

After installation, preload tcmalloc first when running your code. For example:   
``` LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.3 python training.py```

Some sample codes I ran using DDP can be found here: https://github.com/jasonlin316/DDP_GNN  
