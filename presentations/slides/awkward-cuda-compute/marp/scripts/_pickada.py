import cupy as cp
def pick_ada(tag=""):
    chosen=(0,"?")
    for i in range(cp.cuda.runtime.getDeviceCount()):
        name = cp.cuda.runtime.getDeviceProperties(i)['name'].decode()
        if "Ada" in name or "RTX 6000" in name:
            chosen=(i,name); break
    cp.cuda.Device(chosen[0]).use()
    open(f"/tmp/dev_{tag}.txt","w").write(f"dev {chosen[0]} {chosen[1]}")
    return chosen
