import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","savefig.dpi":200,"savefig.bbox":"tight"})
INK="#17303a"; GREEN="#4c8c00"; LGREEN="#eaf5da"; RED="#b23b2e"; LRED="#fbe9e6"
GREY="#9aa7ad"; LGREY="#eef1f2"
OUT="/home/coder/scipy_proceedings/presentations/slides/awkward-cuda-compute/marp/figs"

def rbox(ax,x,y,w,h,txt,face,edge,sub=None,fs=12,tc=INK,bold=True):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=face,ec=edge,lw=2))
    ax.text(x+w/2,y+h/2+(0.12 if sub else 0),txt,ha="center",va="center",
            fontsize=fs,fontweight="bold" if bold else "normal",color=tc)
    if sub: ax.text(x+w/2,y+h/2-0.26,sub,ha="center",va="center",fontsize=9.5,color="#5a6b70")

def arr(ax,p1,p2,color=INK,lw=2.2):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=16,lw=lw,color=color))

fig,(axL,axR)=plt.subplots(1,2,figsize=(12.4,4.9))
for ax in (axL,axR): ax.set_xlim(0,10); ax.set_ylim(0,7); ax.axis("off")

# ---------- LEFT: unfused ----------
axL.set_title("Eager: one kernel per step",fontsize=14,fontweight="bold",color=RED,pad=8)
for i,(x,lbl) in enumerate([(0.4,"map"),(3.5,"zip"),(6.6,"reduce")]):
    rbox(axL,x,4.6,3.0,1.2,f"kernel {i+1}",LGREY,GREY,sub=lbl,fs=12)
rbox(axL,0.4,0.5,9.2,1.2,"GPU global memory  (DRAM)",LRED,RED,fs=12.5,tc=RED)
# write/read arrows
for x in (1.9,5.0,8.1):
    arr(axL,(x,4.55),(x,1.75),color=RED,lw=1.9)
    arr(axL,(x+0.5,1.75),(x+0.5,4.55),color=RED,lw=1.9)
axL.text(5.0,3.15,"every intermediate\nspilled & re-read",ha="center",va="center",
         fontsize=10.5,color=RED,style="italic")
axL.text(5.0,6.15,"3 launches  ·  6 DRAM trips",ha="center",fontsize=11,color="#5a6b70")

# ---------- RIGHT: fused ----------
axR.set_title("Fused: one kernel, iterators",fontsize=14,fontweight="bold",color=GREEN,pad=8)
axR.add_patch(FancyBboxPatch((1.6,3.3),6.8,2.6,boxstyle="round,pad=0.02,rounding_size=0.12",
             fc=LGREEN,ec=GREEN,lw=2))
axR.text(5.0,5.35,"single fused kernel",ha="center",fontsize=13.5,fontweight="bold",color=INK)
axR.text(5.0,4.55,"map → zip → reduce",ha="center",fontsize=11.5,color=INK)
axR.text(5.0,3.95,"values stay in registers / L1",ha="center",fontsize=10.5,color=GREEN,style="italic")
rbox(axR,1.6,0.5,6.8,1.1,"GPU global memory  (DRAM)",LGREY,GREY,fs=12.5,tc="#5a6b70")
arr(axR,(5.0,3.25),(5.0,1.65),color=GREEN,lw=2.2)
axR.text(5.55,2.4,"result only",ha="left",va="center",fontsize=10,color=GREEN)
axR.text(5.0,6.15,"1 launch  ·  1 DRAM write",ha="center",fontsize=11,color="#5a6b70")

fig.savefig(f"{OUT}/fusion_before_after.png"); plt.close(fig)
print("wrote fusion_before_after.png")
