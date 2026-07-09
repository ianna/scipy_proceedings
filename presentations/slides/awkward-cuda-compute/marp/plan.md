1. What is cuda.compute?
   - One of several libraries in cuda.python (a diagram of the cuda-python ecosystem)
   - Let's you <u> build algorithms </u>
   - Lower level than array/tensor frameworks
   - Inspired by C++ (generic algorithms, iterators)
   - Everything is JIT

2. An example - argmin
   - What is argmin?
   - CUDA C++ implementation (the older, static implementation - complex and verbose)
   - With cuda.compute (simple, pure Python, uses iterators)
   
3. cuda-cccl features/overview
   - Algorithms
   - Iterators
   - Struct types
   - JIT and ahead-of-time (serialization - see docs beoing added in https://github.com/NVIDIA/cccl/pull/9732).
   
--

4. Kernel fusion
   - Simple example, abs sum, nsight profile showing distinct kernels
   - Implicit fusion (torch.compile). and when it doesn't work (see the older presentation). Nsight profile showing a single kernel.
   - User-controlled fusion (cuda.compute). Nsight profile showing a single kernel.

   
5. cuda.compute + Awkward Arrays

  - What the state of Awkward was previously
  - Some examples of ported algorithms (pick interesting ones)
  - How awkward + CUDA works today
  - Where is this going? (lazy array, e.g., https://github.com/scikit-hep/awkward/issues/4141).
  - In the meantime, explicit cuda.compute to fuse operations, such as the "physics analysis" diagram from the previous presentation.


   - 
