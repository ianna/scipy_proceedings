# Kernel-fusion micro-benchmarks

IMPORTANT: size the working set past the GPU L2 cache or you measure cache, not
DRAM. The RTX 6000 Ada (AD102) has a 96 MB L2; 10M float64 = 80 MB fits entirely
in it and gives bogus >1 TB/s "bandwidth". Use 100M float64 = 800 MB (DRAM-bound).

bench_fusion_timing.py    walltime, warmup 20, median of 50, iterators built once.
                          Run cc and torch as SEPARATE processes (a shared CUDA
                          context makes the cuda.compute reduce build fail, err 999):
                            python bench_fusion_timing.py cc
                            python bench_fusion_timing.py torch
bench_reused_iterator.py  shows reused vs fresh iterator (fresh adds ~20 us/call).
reduce_fusion.py / torch_fusion.py   kernel COUNTS via nsys --capture-range=cudaProfilerApi.

Reproduced (RTX 6000 Ada, 100M float64, DRAM-bound ~870 GB/s):
  abs-sum:   torch eager 2 kern 2.9 ms | torch.compile 2 kern 0.93 ms | cuda.compute 2 kern 0.93 ms
  Both fused paths hit the bandwidth floor (800 MB / 870 GB/s ~ 0.92 ms); eager is
  3x slower because it writes |x| and reads it back. torch.compile and cuda.compute
  are comparable on dense arrays -- cuda.compute's edge is ragged/custom data (dimuon).

Earlier "74/112 us, cuda.compute faster" numbers were an L2-cache artifact (80 MB).
