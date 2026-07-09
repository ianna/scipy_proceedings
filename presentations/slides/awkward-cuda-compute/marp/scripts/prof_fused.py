import cupy as cp, numpy as np
from _pickada import pick_ada
from cuda.compute import reduce_into, TransformIterator, OpKind
i,name = pick_ada("fused"); print("using dev", i, name)
n = 10_000_000
x = cp.random.randn(n, dtype=cp.float64)
out = cp.empty(1, dtype=cp.float64)
absx = TransformIterator(x, lambda v: abs(v))
kw = dict(d_in=absx, d_out=out, num_items=n, op=OpKind.PLUS, h_init=np.array([0.0]))
reduce_into(**kw); cp.cuda.Stream.null.synchronize()   # warm-up JIT
cp.cuda.profiler.start()
reduce_into(**kw); cp.cuda.Stream.null.synchronize()
cp.cuda.profiler.stop()
print("fused sum=", float(out[0]))
