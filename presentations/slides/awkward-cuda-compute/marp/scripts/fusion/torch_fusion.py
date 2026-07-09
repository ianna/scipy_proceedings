import sys, torch, statistics as st
dev="cuda"
N=10_000_000
x=((torch.arange(N,device=dev,dtype=torch.float64)%7)-3.0)
y=((torch.arange(N,device=dev,dtype=torch.float64)%5)-2.0)

def abs_eager():  return torch.abs(x).sum()
@torch.compile
def abs_comp(x):  return torch.abs(x).sum()
def expxy_eager():return torch.exp(x*y).sum()
@torch.compile
def expxy_comp(x,y): return torch.exp(x*y).sum()

variants={"abs_eager":abs_eager,
          "abs_compile":lambda:abs_comp(x),
          "expxy_eager":expxy_eager,
          "expxy_compile":lambda:expxy_comp(x,y)}
v=sys.argv[1]; fn=variants[v]
for _ in range(8): fn()          # warm up (compile)
torch.cuda.synchronize()
# timing
ts=[]; s=torch.cuda.Event(True); e=torch.cuda.Event(True)
for _ in range(50):
    s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e)*1e3)
print(f"{v}: {st.median(ts):.1f} us")
# kernel capture
torch.cuda.cudart().cudaProfilerStart(); fn(); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
