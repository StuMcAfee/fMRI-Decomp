"""Phase 3 figure: eigen-spectrum of the stacked channels with its nulls, and the gated split-half test.
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
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
n = min(len(sp), 200)
i = sp["index"][:n]

ax = axes[0]
ax.semilogy(i, sp["eigenvalue"][:n], color="#2a78d6", lw=1.3, label="data [c0 | c1]")
ax.semilogy(i, sp["null_threshold"][:n], color="#d64a2a", lw=1, label=f"phase-randomized null, index by index ({r['null_quantile']:.2f})")
if np.isfinite(sp["residual_eigenvalue"]).any():
    ax.semilogy(i, sp["residual_eigenvalue"][:n], color="#52514e", lw=1, ls="--", label="residual channels")
ax.axhline(r["mp_edge"], color="#9a9893", lw=0.8, ls=":", label=f"Marchenko-Pastur edge {r['mp_edge']:.2f}")
ax.set_title("spectrum (first 200), whitened by sigma only", fontsize=9, loc="left")
ax.legend(fontsize=6.5, frameon=False)

ax = axes[1]
if "eigenvalue_whitened" in sp.dtype.names:
    ax.semilogy(i, sp["eigenvalue_whitened"][:n], color="#1baf7a", lw=1.3, label="data after common noise whitening")
    ax.axhline(r["mp_edge"], color="#9a9893", lw=0.8, ls=":", label="Marchenko-Pastur edge")
ax.set_title("spectrum after common temporal noise whitening", fontsize=9, loc="left")
ax.legend(fontsize=6.5, frameon=False)

ax = axes[2]
for name, col in (("A_to_B", "#2a78d6"), ("B_to_A", "#d64a2a")):
    tr = (r.get("split_trace") or {}).get(name) or []
    if tr:
        k = [t["k"] for t in tr]
        ax.semilogy(k, [t["stat"] for t in tr], color=col, lw=1.2, label=f"{name}: half variance along direction k")
        ax.semilogy(k, [t["threshold"] for t in tr], color=col, lw=0.8, ls="--", label=f"{name}: null ({r['null_quantile']:.2f})")
ax.axvline(r["k_temporal"], color="#1baf7a", lw=0.8)
ax.set_title(f"gated test: split-half deflated (k = {r['k_temporal']}, halves {r['k_split_halves']})", fontsize=9, loc="left")
ax.legend(fontsize=6.5, frameon=False)
for a in axes:
    a.set_xlabel("component index")
    a.grid(color="#e6e5e0", lw=0.5)
axes[0].set_ylabel("eigenvalue / variance (whitened units)")

mp = r.get("mapca") or {}
mtxt = f"; MAPCA AIC/KIC/MDL {mp.get('aic')}/{mp.get('kic')}/{mp.get('mdl')}" if mp.get("available") else ""
c, cmax = r.get("contamination"), r.get("contamination_validated_max", 0.45)
flag = "inside validated range" if c is not None and c <= cmax else "OUTSIDE validated range"
fig.suptitle(f"Phase 3 rank: {r.get('run_id')} ({r.get('source')}); gated k {r['k_temporal']} (contamination {c}, {flag}); "
             f"index-by-index {r['k_temporal_classic']}, MP {r['k_mp_edge']}{mtxt}", fontsize=8, x=0.01, ha="left")
fig.tight_layout()
out = folder / "rank_spectrum.png"
fig.savefig(out, dpi=120)
print(out)
