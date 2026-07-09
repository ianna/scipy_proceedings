import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge

plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
GREEN="#4c8c00"; LGREEN="#a5d152"; GREY="#9aa7ad"; INK="#17303a"
OUT="/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/figs"

# ---- Data (paper coverage table + issue #3793 categories) ----
total=130; migrated=80; remaining=total-migrated
pct=migrated/total*100

# ===== Option A: donut =====
fig,ax=plt.subplots(figsize=(5.4,5.0)); ax.axis("equal")
ax.pie([migrated,remaining], startangle=90, counterclock=False,
       colors=[GREEN,"#dfe6e9"], wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
ax.text(0,0.12,f"{pct:.0f}%",ha="center",va="center",fontsize=46,fontweight="bold",color=INK)
ax.text(0,-0.30,"of GPU kernels\non cuda.compute",ha="center",va="center",fontsize=13,color="#555")
ax.set_title("Migration to cuda.compute\n~80 of ~130 kernels",fontsize=14,fontweight="bold",color=INK,pad=12)
# legend
ax.text(0,-1.32,f"✔ reductions & sort: 100%   ·   remaining ~{remaining}: structural ops",
        ha="center",fontsize=10.5,color="#555")
fig.savefig(f"{OUT}/migration_donut.png"); plt.close(fig); print("wrote migration_donut.png")

# ===== Option B: stacked category bars =====
# issue #3793 priority tiers (approx counts) -> migrated vs remaining per tier
cats   =["Reductions & sort\n(high priority)","List / record\nstructural (medium)","Union & misc\n(low priority)"]
migr   =[23, 48, 9]
remain =[0, 21, 6]
y=np.arange(len(cats))[::-1]
fig,ax=plt.subplots(figsize=(8.6,3.6))
ax.barh(y,migr,color=GREEN,label="on cuda.compute",edgecolor="white")
ax.barh(y,remain,left=migr,color="#dfe6e9",label="still hand-written",edgecolor="white")
for yi,m,r in zip(y,migr,remain):
    ax.text(m/2,yi,str(m),ha="center",va="center",color="white",fontweight="bold",fontsize=11)
    if r: ax.text(m+r/2,yi,str(r),ha="center",va="center",color="#555",fontweight="bold",fontsize=11)
ax.set_yticks(y); ax.set_yticklabels(cats,fontsize=11)
ax.set_xlabel("GPU kernels")
ax.set_title("Kernel migration by category  ·  ~80 / ~130 ported (issue #3793)",
             fontsize=12.5,fontweight="bold",color=INK)
ax.legend(loc="lower right",frameon=False,fontsize=10)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(left=False)
fig.savefig(f"{OUT}/migration_bars.png"); plt.close(fig); print("wrote migration_bars.png")
