# Deck build scripts

- `gen_slides.py`  → generates ../slides.md (progressive-build deck). Run: `python3 gen_slides.py`
- `mk_timelines.py`→ eager/torch/cuda.compute timelines from real nsys traces (figs/timeline_*.png)
- `mk_concepts.py` → ecosystem / spectrum / jit_pipeline / aot / lazy_array diagrams
- `mkfig_migration.py`, `mkfig_fusion.py` → migration donut/bars, fusion before/after
- `prof_*.py` + `_pickada.py` → profiling harness (select RTX 6000 Ada by name)

Render deck:
  CHROME_PATH=~/.local/chrome/chrome-nosandbox.sh \
  marp --no-stdin --pdf --allow-local-files --browser chrome ../slides.md -o ../slides.pdf

Re-profile timelines (RTX 6000 Ada):
  PYTHONPATH=. nsys profile -o /tmp/ada_eager --force-overwrite true --trace=cuda \
    --sample=none --capture-range=cudaProfilerApi python3 prof_eager.py
