import torch
def pick():
    for i in range(torch.cuda.device_count()):
        n=torch.cuda.get_device_name(i)
        if "Ada" in n or "6000" in n: torch.cuda.set_device(i); return i,n
    torch.cuda.set_device(0); return 0, torch.cuda.get_device_name(0)
i,n=pick(); open("/tmp/dev_torch.txt","w").write(f"dev {i} {n}")
x=torch.randn(10_000_000, dtype=torch.float64, device='cuda')
@torch.compile
def abs_sum(x): return torch.abs(x).sum()
for _ in range(3):
    abs_sum(x); torch.cuda.synchronize()          # warm-up (compile)
torch.cuda.profiler.start()
r=abs_sum(x); torch.cuda.synchronize()
torch.cuda.profiler.stop()
print("torch sum=", float(r))
