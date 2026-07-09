import sys, math, time, numpy as np, cupy as cp, statistics as st
from cuda.compute import (TransformIterator, ZipIterator, reduce_into,
                          OpKind, Determinism)

N = 10_000_000
xc = (cp.arange(N, dtype=cp.float64) % 7) - 3.0
yc = (cp.arange(N, dtype=cp.float64) % 5) - 2.0
out = cp.empty(1, dtype=cp.float64); h0 = np.zeros(1)
G = Determinism.NOT_GUARANTEED
sync = cp.cuda.Stream.null.synchronize

def bench(fn, warmup=20, reps=200):
    for _ in range(warmup): fn()
    sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t) * 1e6)   # us, walltime
    return st.median(ts)

# --- iterators constructed ONCE, reused every call ---
absx_once  = TransformIterator(xc, lambda v: abs(v))
expxy_once = TransformIterator(ZipIterator(xc, yc), lambda p: math.exp(p[0] * p[1]))

def red(it, det=None):
    kw = {} if det is None else {"determinism": det}
    reduce_into(d_in=it, d_out=out, num_items=N, op=OpKind.PLUS, h_init=h0, **kw)

cases = [
    ("abs-sum   reused iterator",  lambda: red(absx_once)),
    ("abs-sum   fresh iterator",   lambda: red(TransformIterator(xc, lambda v: abs(v)))),
    ("exp*sum   reused iterator",  lambda: red(expxy_once, G)),
    ("exp*sum   fresh iterator",   lambda: red(TransformIterator(ZipIterator(xc, yc),
                                                lambda p: math.exp(p[0]*p[1])), G)),
]
print(f"{'case':28s} {'median us':>10s}   (walltime, warmup 20, median of 200, JIT excluded)")
for label, fn in cases:
    print(f"{label:28s} {bench(fn):10.1f}")
