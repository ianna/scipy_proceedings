---
marp: true
theme: default
paginate: true
size: 16:9
header: 'GPU-Accelerated Awkward Arrays with CUDA Python'
style: |
  section { font-size: 24px; color: #17303a; padding: 44px 60px 52px 60px; display: flex; flex-direction: column; justify-content: flex-start !important; }
  section.sec { justify-content: center !important; }
  h2 { color: #123f4d; border-bottom: 3px solid #76B900; padding-bottom: 6px; margin: 0 0 6px 0; }
  strong { color: #123f4d; }
  code { background: #eef3f5; padding: 1px 5px; border-radius: 4px; }
  .cols { display: grid; grid-template-columns: 0.9fr 1.1fr; gap: 34px; flex: 1 1 auto; align-items: stretch; }
  .cols > div:last-child { display: flex; flex-direction: column; justify-content: center; }
  .cols ul { margin-top: 16px; }
  .cols li { margin: 12px 0; }
  .cur  { color: #123f4d; font-weight: 700; }
  .past, .past strong { color: #9aa8ad; font-weight: 400; }
  pre { font-size: 15px; margin: 0; }
  .note { font-size: 16px; color: #45555b; margin: 3px 0 12px 4px; }
  .note b { color: #123f4d; }
  .slow { color: #b23b2e; font-weight: 700; }
  .fast { color: #4c8c00; font-weight: 700; }
  .aot-h { font-size: 17px; font-weight: 700; color: #4c8c00; margin: 0 0 8px 2px; }
  .aot-gap { margin-top: 22px; border-top: 1px solid #dfe6e8; padding-top: 16px; }
  header { color: #9fb3bb; font-size: 15px; }
  footer { color: #9fb3bb; font-size: 15px; }
  section.sec { background: #0b2a34; justify-content: center; text-align: center; }
  section.sec h1 { color: #76B900; font-size: 44px; }
  section.sec p { color: #7fa6a0; font-size: 22px; }
  section.sec header, section.sec footer, section.sec:after { display: none; }
---

<!-- _class: sec -->

# GPU-Accelerated Awkward Arrays with CUDA Python

Ianna Osborne (Princeton) · **Ashwin Srinath** (NVIDIA) · SciPy 2026


---

## What is `cuda.compute`?

<div class="cols">
<div>

- <span class="cur">One of several libraries in **CUDA Python**</span>

</div>
<div>

![w:760](figs/ecosystem.png)

</div>
</div>


---

## What is `cuda.compute`?

<div class="cols">
<div>

- <span class="past">One of several libraries in **CUDA Python**</span>
- <span class="cur">Lets you build **custom algorithms**</span>

</div>
<div>

```python
import cupy as cp
from cuda.compute import merge_sort

data_in  = cp.array([13, 41, 22, 8, 95, 34])
data_out = cp.empty_like(data_in)

# sort by the last digit: a custom comparator, in Python
merge_sort(d_in_keys=data_in, d_out_keys=data_out,
           num_items=data_in.size,
           op=lambda a, b: a % 10 < b % 10)

# data_out -> [41, 22, 13, 34, 95, 8]
```

</div>
</div>


---

## What is `cuda.compute`?

<div class="cols">
<div>

- <span class="past">One of several libraries in **CUDA Python**</span>
- <span class="past">Lets you build **custom algorithms**</span>
- <span class="cur">Sits **below** the array / tensor frameworks</span>

</div>
<div>

![w:780](figs/spectrum.png)

</div>
</div>


---

## What is `cuda.compute`?

<div class="cols">
<div>

- <span class="past">One of several libraries in **CUDA Python**</span>
- <span class="past">Lets you build **custom algorithms**</span>
- <span class="past">Sits **below** the array / tensor frameworks</span>
- <span class="cur">Inspired by **C++**: generic algorithms and iterators</span>

</div>
<div>

<div class="aot-h">CUDA C++ / Thrust</div>

```cpp
auto squares = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0),
    [] __device__ (int i) { return i * i; });

int total = thrust::reduce(squares, squares + n);
```

<div class="aot-h aot-gap">cuda.compute (Python)</div>

```python
squares = TransformIterator(CountingIterator(0),
                            lambda i: i * i)
reduce_into(d_in=squares, d_out=out, num_items=n,
            op=OpKind.PLUS, h_init=np.zeros(1))
```

</div>
</div>


---

## What is `cuda.compute`?

<div class="cols">
<div>

- <span class="past">One of several libraries in **CUDA Python**</span>
- <span class="past">Lets you build **custom algorithms**</span>
- <span class="past">Sits **below** the array / tensor frameworks</span>
- <span class="past">Inspired by **C++**: generic algorithms and iterators</span>
- <span class="cur">Everything is compiled **just in time**</span>

</div>
<div>

![w:700](figs/jit_compile.png)

</div>
</div>


---

## `cuda.compute` features

<div class="cols">
<div>

- <span class="cur">**Algorithms**: composable parallel building blocks (CUB and Thrust)</span>

</div>
<div>

![w:680](figs/algorithms.png)

</div>
</div>


---

## `cuda.compute` features

<div class="cols">
<div>

- <span class="past">**Algorithms**: composable parallel building blocks (CUB and Thrust)</span>
- <span class="cur">**Iterators**: lazy sequences that fuse a step in</span>

</div>
<div>

![w:680](figs/iterators_seq.png)

</div>
</div>


---

## `cuda.compute` features

<div class="cols">
<div>

- <span class="past">**Algorithms**: composable parallel building blocks (CUB and Thrust)</span>
- <span class="past">**Iterators**: lazy sequences that fuse a step in</span>
- <span class="cur">**Computation with arbitrary data types**</span>

</div>
<div>

```python
from cuda.compute import gpu_struct, reduce_into
import numpy as np

@gpu_struct
class Pixel:
    r: np.int32
    g: np.int32
    b: np.int32

def max_green(x, y):
    return x if x.g > y.g else y

# reduce a buffer of RGB pixels to the greenest one
reduce_into(d_in=d_rgb, d_out=out, num_items=n,
            op=max_green, h_init=Pixel(0, 0, 0))
```

</div>
</div>


---

## `cuda.compute` features

<div class="cols">
<div>

- <span class="past">**Algorithms**: composable parallel building blocks (CUB and Thrust)</span>
- <span class="past">**Iterators**: lazy sequences that fuse a step in</span>
- <span class="past">**Computation with arbitrary data types**</span>
- <span class="cur">**JIT by default. Supports ahead-of-time compilation workflows.**</span>

</div>
<div>

<div class="aot-h">JIT: compile on first use</div>

```python
from cuda.compute import unary_transform

def square(v): return v * v

unary_transform(d_in=x, d_out=y, op=square, num_items=n)
unary_transform(d_in=x, d_out=y, op=square, num_items=n)   # again
```

<div class="note">1st call <span class="slow">1077 ms</span>, JIT compiles &nbsp;·&nbsp; 2nd call <span class="fast">0.27 ms</span>, cached</div>

<div class="aot-h aot-gap">Ahead of time: compile once, reuse</div>

```python
from cuda.compute import make_unary_transform, serialize

alg = make_unary_transform(d_in=x, d_out=y, op=square)
open("square.cccl", "wb").write(serialize(alg))
```

<div class="note"><b>serialize</b> the compiled kernel → a <b>142 KB</b> blob on disk</div>

```python
from cuda.compute import deserialize     # a later session

alg = deserialize(open("square.cccl", "rb").read())
alg(d_in=x, d_out=y, op=square, num_items=n)
```

<div class="note"><b>deserialize</b> → 1st call <span class="fast">3.8 ms</span> (<b>no JIT</b>) &nbsp;·&nbsp; 2nd call <span class="fast">0.21 ms</span>, cached</div>

</div>
</div>


---

## Awkward Array algorithms built with `cuda.compute`

<div class="cols">
<div>

- <span class="cur">**Awkward array layout**: ragged data stored as flat content and offsets buffers, not a rectangular block</span>

</div>
<div>

![w:720](figs/layout.png)

</div>
</div>


---

## Awkward Array algorithms built with `cuda.compute`

<div class="cols">
<div>

- <span class="past">**Awkward array layout**: ragged data stored as flat content and offsets buffers, not a rectangular block</span>
- <span class="cur">**Example**: `ak.argmin`</span>

</div>
<div>

```python
data = [[3.1, 1.4], [], [9.0, 0.5, 2.2]]

# index of the minimum within each sublist
ak.argmin(data, axis=1)   # [1, None, 1]
```

</div>
</div>


---

## Awkward Array algorithms built with `cuda.compute`

<div class="cols">
<div>

- <span class="past">**Awkward array layout**: ragged data stored as flat content and offsets buffers, not a rectangular block</span>
- <span class="past">**Example**: `ak.argmin`</span>
- <span class="cur">**Previous Awkward implementation**: a handwritten CUDA C++ kernel, three passes and about 150 lines</span>

</div>
<div>

```cpp
// awkward_reduce_argmin_b  (1 of 3 kernels)
// per block reduce by parent, then combine block
// winners across blocks with an atomic CAS retry loop:
uint64_t cur = atomic_toptr[parent];
while (true) {
  if (cur == EMPTY) {
    if (atomicCAS(&atomic_toptr[parent], EMPTY, cand)
          == EMPTY) break;          // installed
  } else if (fromptr[cand] < fromptr[(int64_t)cur]) {
    uint64_t prev =
        atomicCAS(&atomic_toptr[parent], cur, cand);
    if (prev == cur) break;         // replaced
    cur = prev;                     // lost race, retry
  } else break;
}
```

</div>
</div>


---

## Awkward Array algorithms built with `cuda.compute`

<div class="cols">
<div>

- <span class="past">**Awkward array layout**: ragged data stored as flat content and offsets buffers, not a rectangular block</span>
- <span class="past">**Example**: `ak.argmin`</span>
- <span class="past">**Previous Awkward implementation**: a handwritten CUDA C++ kernel, three passes and about 150 lines</span>
- <span class="cur">**Using cuda.compute**: the same reduction in a few lines of Python</span>

</div>
<div>

```python
# Awkward's cuda.compute ak.argmin, in full:
def segment_argmin(seg_id):
    lo, hi = starts[seg_id], stops[seg_id]
    if lo == hi:
        return -1
    return np.argmin(content[lo:hi]) + lo

unary_transform(d_in=CountingIterator(0),
                d_out=result, op=segment_argmin,
                num_items=num_sublists)
```

</div>
</div>


---

## Awkward Array algorithms built with `cuda.compute`

<div class="cols">
<div>

- <span class="past">**Awkward array layout**: ragged data stored as flat content and offsets buffers, not a rectangular block</span>
- <span class="past">**Example**: `ak.argmin`</span>
- <span class="past">**Previous Awkward implementation**: a handwritten CUDA C++ kernel, three passes and about 150 lines</span>
- <span class="past">**Using cuda.compute**: the same reduction in a few lines of Python</span>
- <span class="cur">**Performance results**: identical output, and faster than the handwritten kernel</span>

</div>
<div>

![w:720](figs/bench_ak_argmin.png)

</div>
</div>


---

## Kernel fusion

<div class="cols">
<div>

- <span class="cur">**Kernel fusion** is key to optimizing memory traffic and reducing launch overhead</span>

</div>
<div>

![w:820](figs/kernel_passes.png)

</div>
</div>


---

## Kernel fusion

<div class="cols">
<div>

- <span class="past">**Kernel fusion** is key to optimizing memory traffic and reducing launch overhead</span>
- <span class="cur">**Implicit vs explicit fusion**: `|x|` then `sum`, without materializing the intermediate (100M values)</span>

</div>
<div>

<div class="aot-h">eager: materialize the intermediate</div>

```python
y = torch.abs(x).sum()
```

<div class="note"><span class="slow">2 kernels · 2.9 ms</span> &nbsp; writes `|x|` to memory, reads it back: 3x the traffic</div>

<div class="aot-h aot-gap">implicit fusion (torch.compile)</div>

```python
@torch.compile
def abs_sum(x): return torch.abs(x).sum()
```

<div class="note"><span class="fast">2 kernels · 0.93 ms</span> &nbsp; a compiler decides what fuses</div>

<div class="aot-h aot-gap">explicit fusion (cuda.compute)</div>

```python
absx = TransformIterator(x, lambda v: abs(v))
reduce_into(d_in=absx, d_out=out, op=OpKind.PLUS, ...)
```

<div class="note"><span class="fast">2 kernels · 0.93 ms</span> &nbsp; you compose the fusion, on any data</div>

</div>
</div>


---

## Kernel fusion

<div class="cols">
<div>

- <span class="past">**Kernel fusion** is key to optimizing memory traffic and reducing launch overhead</span>
- <span class="past">**Implicit vs explicit fusion**: `|x|` then `sum`, without materializing the intermediate (100M values)</span>
- <span class="cur">**A real workflow**: dimuon invariant mass, one line of Awkward</span>

</div>
<div>

```python
mu1, mu2 = ak.unzip(ak.combinations(muons, 2))

mass = np.sqrt(
    2 * mu1.pt * mu2.pt
    * (np.cosh(mu1.eta - mu2.eta)
     - np.cos (mu1.phi - mu2.phi)))
```

<div class="note">array at a time: on the GPU every `ufunc` becomes its own kernel, spilling each intermediate to memory</div>

</div>
</div>


---

## Kernel fusion

<div class="cols">
<div>

- <span class="past">**Kernel fusion** is key to optimizing memory traffic and reducing launch overhead</span>
- <span class="past">**Implicit vs explicit fusion**: `|x|` then `sum`, without materializing the intermediate (100M values)</span>
- <span class="past">**A real workflow**: dimuon invariant mass, one line of Awkward</span>
- <span class="cur">**Fused with cuda.compute**: the whole formula in one `binary_transform`</span>

</div>
<div>

```python
@gpu_struct
class Muon:
    pt: float32; eta: float32
    phi: float32; charge: int32

def mass(m1, m2):
    return (2 * m1.pt * m2.pt
        * (cosh(m1.eta - m2.eta)
         - cos (m1.phi - m2.phi))) ** 0.5

binary_transform(d_in1=muons1, d_in2=muons2,
                 d_out=out, op=mass, num_items=n)
```

<div class="note">one fused kernel over zipped, permuted iterators: no intermediates in memory</div>

</div>
</div>


---

## Kernel fusion

<div class="cols">
<div>

- <span class="past">**Kernel fusion** is key to optimizing memory traffic and reducing launch overhead</span>
- <span class="past">**Implicit vs explicit fusion**: `|x|` then `sum`, without materializing the intermediate (100M values)</span>
- <span class="past">**A real workflow**: dimuon invariant mass, one line of Awkward</span>
- <span class="past">**Fused with cuda.compute**: the whole formula in one `binary_transform`</span>
- <span class="cur">**The measured GPU timeline**: 88 kernels become 1, **2.9x faster**</span>

</div>
<div>

![w:940](figs/dimuon_timeline.png)

</div>
</div>


---

## Awkward Array: present and future

<div class="cols">
<div>

- <span class="cur">**85% of CUDA kernels** ported from CUDA C++ to pure Python</span>

</div>
<div>

![w:830](figs/port_ratio.png)

</div>
</div>


---

## Awkward Array: present and future

<div class="cols">
<div>

- <span class="past">**85% of CUDA kernels** ported from CUDA C++ to pure Python</span>
- <span class="cur">**From `parents` to `offsets`**: the ragged layout maps straight onto segmented algorithms</span>

</div>
<div>

![w:520](figs/layout.png)

```python
# a ragged array is a flat `content` + `offsets`;
# sublist k spans offsets[k] : offsets[k+1]
segmented_reduce(
    d_in=content, d_out=out,
    num_segments=len(offsets) - 1,
    start_offsets_in=offsets[:-1],
    end_offsets_in=offsets[1:],
    op=OpKind.MINIMUM, h_init=identity)
```

</div>
</div>


---

## Awkward Array: present and future

<div class="cols">
<div>

- <span class="past">**85% of CUDA kernels** ported from CUDA C++ to pure Python</span>
- <span class="past">**From `parents` to `offsets`**: the ragged layout maps straight onto segmented algorithms</span>
- <span class="cur">**Awkward on CUDA is now even faster**</span>

</div>
<div>

![w:880](figs/adl_speedup_panel.png)

<div class="note">since the move to `cuda.compute`, Awkward matches or beats the handwritten kernels on the ADL benchmark queries: up to <b>1485x</b> on the GPU compute stage, and queries the old backend could not finish now run in seconds</div>

</div>
</div>


---

## Awkward Array: present and future

<div class="cols">
<div>

- <span class="past">**85% of CUDA kernels** ported from CUDA C++ to pure Python</span>
- <span class="past">**From `parents` to `offsets`**: the ragged layout maps straight onto segmented algorithms</span>
- <span class="past">**Awkward on CUDA is now even faster**</span>
- <span class="cur">**What is next: lazy execution**</span>

</div>
<div>

```python
la = lazy(array)             # wrap: nothing runs yet

# a chain of elementwise ops, written normally
expr = la
for _ in range(16):
    expr = expr * 1.001 + 0.5

expr.compute(fuse=True)      # whole chain -> ONE kernel
```

<div class="note">the lazy layer fuses a whole chain of operations into a single kernel automatically: a 32 op chain becomes one launch, <b>up to ~90x faster</b> than running each op eagerly, and the deeper the chain the bigger the win</div>

</div>
</div>
