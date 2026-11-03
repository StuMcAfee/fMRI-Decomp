"""Phase 3 figure: eigen-spectrum of the stacked channels with its nulls (log scale).
Usage: python scripts/plot_rank.py <rank output folder>"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder = Path(sys.argv[1])
r = json.loads((folder / "rank.json").read_text())
sp = np.genfromtxt(folder / "spectrum.tsv", delimiter="\t", names=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, n in zip(axes, (len(sp), min(len(sp), 200))):
    i = sp["index"][:n]
    ax.semilogy(i, sp["eigenvalue"][:n], color="#2a78d6", lw=1.3, label="data [c0 | c1]")
    ax.semilogy(i, sp["null_threshold"][:n], color="#d64a2a", lw=1, label=f"phase-randomized null ({r['null_quantile']:.2f} quantile, index by index)")
    if np.isfinite(sp["residual_eigenvalue"]).any():
        ax.semilogy(i, sp["residual_eigenvalue"][:n], color="#52514e", lw=1, ls="--", label="residual channels (thermal null)")
    ax.axhline(r["mp_edge"], color="#9a9893", lw=0.8, ls=":", label=f"Marchenko-Pastur edge {r['mp_edge']:.2f}")
    dt = r.get("deflated_tests") or {}
    if dt:
        ks = sorted(int(k) for k in dt)
        ks = [k for k in ks if k <= n]
        ax.scatter(ks, [dt[str(k)]["threshold"] for k in ks], s=6, color="#1baf7a", zorder=3, label="deflated null (tested k)")
    for k, c, lab in ((r["k_temporal_classic"], "#d64a2a", "k index-by-index"), (r["k_temporal"], "#1baf7a", "k deflated"),
                      (r.get("k_thermal"), "#52514e", "k thermal")):
        if k:
            ax.axvline(k, color=c, lw=0.7, alpha=0.6)
    ax.set_xlabel("component index")
    ax.set_ylabel("eigenvalue (units of whitened noise)")
    ax.grid(color="#e6e5e0", lw=0.5)
axes[0].set_title("full spectrum", fontsize=9, loc="left")
axes[1].set_title("first 200", fontsize=9, loc="left")
axes[1].legend(fontsize=6.5, frameon=False)
mp = r.get("mapca") or {}
mtxt = f"; MAPCA AIC/KIC/MDL {mp.get('aic')}/{mp.get('kic')}/{mp.get('mdl')}" if mp.get("available") else ""
fig.suptitle(f"Phase 3 rank (EXPLORATORY, gate not passed): {r.get('run_id')} ({r.get('source')}); k deflated {r['k_temporal']}, "
             f"index-by-index {r['k_temporal_classic']}, thermal {r.get('k_thermal')}, MP {r['k_mp_edge']}{mtxt}", fontsize=8, x=0.01, ha="left")
fig.tight_layout()
out = folder / "rank_spectrum.png"
fig.savefig(out, dpi=120)
print(out)
