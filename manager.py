import psutil
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


class DynamicLoadBalancer:

    def __init__(self, args, is_cpu, threshold_1=0.3, threshold_2=0.01):

        self.args = args
        self.is_cpu = is_cpu
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2

        self.cpu_gpu_ratio = None
        self.num_sample_cores = None
        self.time_gpu = None
        self.time_cpu = None
        self.time_compute = None
        self.time_sample = None

        self.reset()

    def reset(self):
        self.time_sample = []  # μs
        self.time_compute = []  # μs
        self.time_cpu = []  # s
        self.time_gpu = []  # s
        if self.args.cpu_process > 0:
            self.num_sample_cores = 4 // self.args.cpu_process
            # self.num_sample_cores = 2
        self.cpu_gpu_ratio = self.args.cpu_gpu_ratio

    def config(self):
        load_core, comp_core = self.assign_cores()
        return {
            'load_core': load_core,
            'comp_core': comp_core,
            'cpu_gpu_ratio': self.cpu_gpu_ratio,
        }

    def update(self, profile):
        prof, cpu_runtime, gpu_runtime = profile
        time_sample = time_compute = 0
        if self.is_cpu:
            event_list = prof.key_averages()
            for event in event_list:
                if 'DataLoaderIter.__next__' in event.key:
                    time_sample = event.cpu_time_total
                if 'DistributedDataParallel.forward' in event.key:
                    time_compute = event.cpu_time_total
            assert time_sample != 0 and time_compute != 0

        runtime = torch.tensor([time_sample, time_compute, cpu_runtime, gpu_runtime], dtype=torch.float32)
        dist.all_reduce(runtime, op=ReduceOp.SUM)
        runtime = runtime.tolist()
        if dist.get_rank() == 0:
            print()
            print(runtime[0]/runtime[1], runtime[2]/runtime[3])
            print()
        # return

        self.time_sample.append(runtime[0] / self.args.cpu_process)
        self.time_compute.append(runtime[1] / self.args.cpu_process)
        self.time_cpu.append(runtime[2] / self.args.cpu_process)
        self.time_gpu.append(runtime[3] / self.args.gpu_process)

        # update self.num_sample_cores & self.cpu_gpu_ratio
        time_sample, time_compute, time_cpu, time_gpu = self.estimate_time()
        if dist.get_rank() == 0:
            print('TH1', abs(time_sample - time_compute) / min(time_sample, time_compute))
            print('TH2', abs(time_cpu - time_gpu) / min(time_cpu, time_gpu))

        if abs(time_sample - time_compute) / min(time_sample, time_compute) \
                > self.threshold_1:
            if time_sample > time_compute:
                self.num_sample_cores = min(8, self.num_sample_cores * 2)
            else:
                self.num_sample_cores = max(2, self.num_sample_cores // 2)
        else:
            if abs(time_cpu - time_gpu) / max(time_cpu, time_gpu) \
                    > self.threshold_2:
                ratio = time_gpu / (time_cpu + time_gpu)
                self.cpu_gpu_ratio = min(max(0.05, ratio), 0.95)

    def estimate_time(self):
        span = 5
        time_sample = sum(self.time_sample[-span:]) / len(self.time_sample)
        time_compute = sum(self.time_compute[-span:]) / len(self.time_compute)
        time_cpu = sum(self.time_cpu[-span:]) / len(self.time_cpu)
        time_gpu = sum(self.time_gpu[-span:]) / len(self.time_gpu)
        return time_sample, time_compute, time_cpu, time_gpu

    def assign_cores(self):
        if not self.is_cpu:
            return [], []

        num_cpu_proc = self.args.cpu_process
        num_sample_cores = self.num_sample_cores
        n = psutil.cpu_count(logical=False)
        rank = dist.get_rank()

        cores_per_proc = n // num_cpu_proc

        # num_cpu_proc is 2, workload 0.15
        # products
        # cores_per_proc = 12
        # num_sample_cores = 2
        # papers
        # cores_per_proc = 5
        # num_sample_cores = 4

        # num_cpu_proc is 4, workload 0.25
        # products
        # cores_per_proc = 14
        # num_sample_cores = 1

        load_core = list(range(cores_per_proc*rank, cores_per_proc*rank+num_sample_cores))
        comp_core = list(range(cores_per_proc*rank+num_sample_cores, cores_per_proc*(rank+1)))

        return load_core, comp_core
