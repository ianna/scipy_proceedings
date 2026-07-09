import cupy as cp
from _pickada import pick_ada
i,name = pick_ada("eager"); print("using dev", i, name)
n = 10_000_000
x = cp.random.randn(n, dtype=cp.float64); cp.cuda.Stream.null.synchronize()
cp.cuda.profiler.start()
y = cp.abs(x); s = cp.sum(y)
cp.cuda.Stream.null.synchronize(); cp.cuda.profiler.stop()
print("eager sum=", float(s))
