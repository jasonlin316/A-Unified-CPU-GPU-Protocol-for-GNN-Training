import psutil
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp


class DynamicLoadBalancer:

    def __init__(self, args, is_cpu, threshold_1=0.3, threshold_2=0.01):

        self.args = args
        self.is_cpu = is_cpu
        self.threshold_1 = threshold_1  # deprecated
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

    def update(self, prof):
        profile_time = 0
        cpu_sync_time = 0
        gpu_sync_time = 0
        cuda_stream_sync_time = 0

        event_list = prof.key_averages()
        for event in event_list:
            if 'ProfilerStep' in event.key:
                profile_time = event.cpu_time_total
            if 'gloo:all_reduce' in event.key:
                if self.is_cpu:
                    cpu_sync_time = event.cpu_time_total
                else:
                    gpu_sync_time = event.cpu_time_total
            if 'cudaStreamSynchronize' in event.key:
                cuda_stream_sync_time = event.cpu_time_total
        if not self.is_cpu:
            gpu_sync_time -= cuda_stream_sync_time  # actual GPU sync time

        runtime = torch.tensor([cpu_sync_time, gpu_sync_time], dtype=torch.float32)
        dist.all_reduce(runtime, op=ReduceOp.SUM)
        cpu_sync_time, gpu_sync_time = runtime.tolist()
        cpu_sync_time /= self.args.cpu_process
        gpu_sync_time /= self.args.gpu_process
        self.time_cpu.append((profile_time - cpu_sync_time) / 1e3)
        self.time_gpu.append((profile_time - gpu_sync_time) / 1e3)

        time_cpu, time_gpu = self.estimate_time()
        if dist.get_rank() == 0:
            print('CPU runtime: {:.0f}, GPU runtime: {:.0f}'.format(time_cpu, time_gpu))
            print('TH2:{:.4f}'.format(abs(time_cpu - time_gpu) / min(time_cpu, time_gpu)))

        if abs(time_cpu - time_gpu) / max(time_cpu, time_gpu) \
                > self.threshold_2:
            gpu_estimate_time = time_gpu / (1-self.cpu_gpu_ratio)
            cpu_estimate_time = time_cpu / self.cpu_gpu_ratio
            ratio = gpu_estimate_time / (cpu_estimate_time + gpu_estimate_time)
            self.cpu_gpu_ratio = min(max(0.05, round(ratio, 3)), 0.95)

        # discard: two-fold scheduling
        # if abs(time_sample - time_compute) / min(time_sample, time_compute) \
        #         > self.threshold_1:
        #     if time_sample > time_compute:
        #         self.num_sample_cores = min(8, self.num_sample_cores * 2)
        #     else:
        #         self.num_sample_cores = max(2, self.num_sample_cores // 2)
        # else:
        #     if abs(time_cpu - time_gpu) / max(time_cpu, time_gpu) \
        #             > self.threshold_2:
        #         ratio = time_gpu / (time_cpu + time_gpu)
        #         self.cpu_gpu_ratio = min(max(0.05, ratio), 0.95)

    def estimate_time(self):
        span = 1
        time_cpu = sum(self.time_cpu[-span:]) / span
        time_gpu = sum(self.time_gpu[-span:]) / span
        return time_cpu, time_gpu

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
