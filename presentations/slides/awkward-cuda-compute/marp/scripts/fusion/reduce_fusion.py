import sys, math, numpy as np, cupy as cp
from cuda.compute import (TransformIterator, ZipIterator, reduce_into, OpKind, Determinism)
N=10_000_000
x=(cp.arange(N,dtype=cp.float64)%7)-3.0; y=(cp.arange(N,dtype=cp.float64)%5)-2.0
out=cp.empty(1,dtype=cp.float64); h0=np.zeros(1)
absx =lambda: TransformIterator(x, lambda v: abs(v))
expxy=lambda: TransformIterator(ZipIterator(x,y), lambda p: math.exp(p[0]*p[1]))
def red(it,det=None):
    kw={} if det is None else {"determinism":det}
    reduce_into(d_in=it,d_out=out,num_items=N,op=OpKind.PLUS,h_init=h0,**kw)
v=sys.argv[1]
fn={"absum_default":lambda:red(absx()),
    "absum_notguar":lambda:red(absx(),Determinism.NOT_GUARANTEED),
    "expxy_default":lambda:red(expxy()),
    "expxy_notguar":lambda:red(expxy(),Determinism.NOT_GUARANTEED)}[v]
for _ in range(3): fn()
cp.cuda.Stream.null.synchronize()
cp.cuda.profiler.start(); fn(); cp.cuda.Stream.null.synchronize(); cp.cuda.profiler.stop()
