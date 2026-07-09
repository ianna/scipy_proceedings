#!/usr/bin/env python3
"""
Measure cuda.compute JIT vs ahead-of-time (serialize/deserialize) overhead,
for the "JIT and ahead of time" slide.

Run in your working cuda.compute env:

    python measure_aot.py

It prints the numbers the slide needs:
  * 1st unary_transform call  -> JIT compiles the operator (slow)
  * 2nd call, same process    -> cached (fast)
  * a FRESH process: deserialize a saved kernel + 1st call -> no JIT (fast)

Notes:
  - Picks the largest-memory visible GPU. Override with CUDA_VISIBLE_DEVICES.
  - Writes a temporary blob to $TMPDIR/square_transform.cccl.
"""
import sys, os, time, subprocess

N = 10_000_000
BLOB = os.path.join(os.environ.get("TMPDIR", "/tmp"), "square_transform.cccl")


def square(v):                 # module-level so it survives (de)serialization
    return v * v


def pick_device():
    import cupy as cp
    best, best_mem = 0, -1
    for i in range(cp.cuda.runtime.getDeviceCount()):
        mem = cp.cuda.runtime.getDeviceProperties(i)["totalGlobalMem"]
        if mem > best_mem:
            best, best_mem = i, mem
    cp.cuda.Device(best).use()
    return cp.cuda.runtime.getDeviceProperties(best)["name"].decode()


def sync():
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


def load_and_run():
    """Fresh-process ahead-of-time path: deserialize + first call."""
    import cupy as cp
    dev = pick_device()
    from cuda.compute import deserialize
    x = cp.arange(N, dtype=cp.float64)
    y = cp.empty(N, dtype=cp.float64)

    t0 = time.perf_counter()
    alg = deserialize(open(BLOB, "rb").read())
    t_des = (time.perf_counter() - t0) * 1e3

    sync()
    t = time.perf_counter()
    alg(d_in=x, d_out=y, op=square, num_items=N)
    sync()
    t_call = (time.perf_counter() - t) * 1e3

    sync()
    t2 = time.perf_counter()
    alg(d_in=x, d_out=y, op=square, num_items=N)
    sync()
    t_call2 = (time.perf_counter() - t2) * 1e3

    print(f"AOT_DESERIALIZE_MS={t_des:.3f}")
    print(f"AOT_FIRST_CALL_MS={t_call:.3f}")
    print(f"AOT_SECOND_CALL_MS={t_call2:.3f}")
    print(f"AOT_DEVICE={dev}")
    print(f"AOT_OK={bool(cp.allclose(y, x * x))}")


def main():
    import cupy as cp
    dev = pick_device()
    from cuda.compute import unary_transform, make_unary_transform, serialize
    try:
        from cuda.compute import clear_all_caches
        clear_all_caches()
    except Exception:
        pass

    x = cp.arange(N, dtype=cp.float64)
    y = cp.empty(N, dtype=cp.float64)

    def timed(fn):
        sync(); t = time.perf_counter(); fn(); sync()
        return (time.perf_counter() - t) * 1e3

    try:
        t_jit = timed(lambda: unary_transform(d_in=x, d_out=y, op=square, num_items=N))
        t_cached = timed(lambda: unary_transform(d_in=x, d_out=y, op=square, num_items=N))
        ok = bool(cp.allclose(y, x * x))
        alg = make_unary_transform(d_in=x, d_out=y, op=square)
        blob = serialize(alg)
        open(BLOB, "wb").write(blob)
    except Exception:
        import traceback
        print("!! unary_transform path failed in this env:\n")
        traceback.print_exc()
        sys.exit(1)

    # fresh process for the ahead-of-time measurement
    res = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--load"],
        capture_output=True, text=True, env=os.environ,
    )
    aot = {}
    for line in res.stdout.splitlines():
        if line[:4].isupper() and "=" in line:
            k, v = line.split("=", 1)
            aot[k] = v

    print("\n" + "=" * 60)
    print("  cuda.compute: JIT vs ahead-of-time  (unary_transform, square)")
    print("=" * 60)
    print(f"  device                        : {dev}")
    print(f"  n                             : {N:,}")
    print(f"  correct result                : {ok}")
    print("-" * 60)
    print(f"  1st call   (JIT compiles op)  : {t_jit:11.3f} ms")
    print(f"  2nd call   (cached, process)  : {t_cached:11.3f} ms")
    print(f"  serialized blob               : {len(blob) / 1024:11.1f} KB")
    print("  --- fresh process (ahead of time) ---")
    print(f"  deserialize()                 : {aot.get('AOT_DESERIALIZE_MS', '?'):>11} ms")
    print(f"  1st call   (loaded, no JIT)   : {aot.get('AOT_FIRST_CALL_MS', '?'):>11} ms")
    print(f"  2nd call   (cached, in process): {aot.get('AOT_SECOND_CALL_MS', '?'):>10} ms")
    print("=" * 60)
    if res.returncode != 0 or aot.get("AOT_OK") != "True":
        print("\n[fresh-process stderr/stdout]\n", res.stdout, res.stderr)


if __name__ == "__main__":
    if "--load" in sys.argv:
        load_and_run()
    else:
        main()
