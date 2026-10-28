"""All_reduce 256 MB між GPU через NCCL з PyTorch. Запуск: torchrun --nproc_per_node=2 ar.py"""
import time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r)
x = torch.ones(64 * 1024 * 1024, device="cuda")  # 256 MB
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 20
if r == 0: print(f"TORCH NCCL OK: all_reduce 256 MB = {dt*1e3:.1f} ms, busbw ~ {x.numel()*4/dt/1e9:.1f} GB/s")
dist.destroy_process_group()
