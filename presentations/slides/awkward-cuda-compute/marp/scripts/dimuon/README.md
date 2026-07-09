# Dimuon invariant-mass kernel-fusion benchmark
Reproduces https://github.com/shwina/awkward-cccl (dimuon example) with nsys.

Run in a working cuda.compute env (PYTHONPATH must include this dir for ak_helpers):

  # BEFORE: Awkward GPU array ops  (many small kernels)
  nsys profile -t cuda,nvtx --capture-range=nvtx --nvtx-capture="mass_calculation" \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 -o mass_awkward --force-overwrite true \
    python mass_awkward.py

  # AFTER: cuda.compute binary_transform  (one fused transform_kernel)
  nsys profile -t cuda,nvtx --capture-range=nvtx --nvtx-capture="mass_calculation" \
    --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 -o mass_cuda_compute --force-overwrite true \
    python mass_cuda_compute.py

Count kernels (exclude memory ops = rows without a grid):
  nsys stats --report cuda_gpu_trace --format csv mass_awkward.nsys-rep

Reproduced here (RTX 6000 Ada, awkward 2.10):
  BEFORE  ~88 compute kernels + ~212 memory ops  (~300 GPU operations)
  AFTER   1 fused transform_kernel   (from the notebook; binary_transform build
          currently blocked in this checkout by a stale cuda.compute cp312 binding)

Patches vs the upstream notebook (current cuda.compute API):
  - binary_transform(...) is keyword-only: d_in1=, d_in2=, d_out=, op=, num_items=
  - the op receives the ZipIterator's tuple value: use m[0..3], not m.pt/.eta/...
