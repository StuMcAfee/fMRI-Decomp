"""Phase 4 figure: for each component above the energy null (up to 24), its dR2* map (q, s^-1) and dS0/S0
map (s) on the axial slice through the component's peak, its time course, eta and label.
Usage: python scripts/plot_components.py <decompose k folder> <front-end folder>"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

kdir, fe = Path(sys.argv[1]), Path(sys.argv[2])
summ = json.loads((kdir / "components.json").read_text())
q = np.asarray(nib.load(str(kdir / "q_per_s.nii.gz")).dataobj)
s = np.asarray(nib.load(str(kdir / "s.nii.gz")).dataobj)
bg = np.asarray(nib.load(str(fe / "mu.nii.gz")).dataobj)[..., 0] if (fe / "mu.nii.gz").exists() else None
tc = np.genfromtxt(kdir / "time_courses.tsv", delimiter="\t", names=True)
rows = [r for r in summ["components"] if r["above_null"]][:24]
n = max(len(rows), 1)
fig, axes = plt.subplots(n, 3, figsize=(10, 1.9 * n), gridspec_kw={"width_ratios": [1, 1, 3]}, squeeze=False)
for i, r in enumerate(rows):
    j = r["component"] - 1
    qj, sj = q[..., j], s[..., j]
    mag = np.nan_to_num(np.abs(qj) / max(np.nanpercentile(np.abs(qj), 99.5), 1e-12)) + \
          np.nan_to_num(np.abs(sj) / max(np.nanpercentile(np.abs(sj), 99.5), 1e-12))
    z = int(np.unravel_index(np.argmax(mag), mag.shape)[2])
    for a, (img, name, unit) in zip(axes[i, :2], ((qj, "dR2*", "s^-1"), (sj, "dS0/S0", ""))):
        lim = np.nanpercentile(np.abs(img), 99.5) or 1
        if bg is not None:
            a.imshow(np.rot90(bg[:, :, z]), cmap="gray", alpha=0.5)
        a.imshow(np.rot90(img[:, :, z]), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.9)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{name} z={z} (+/-{lim:.2g}{' ' + unit if unit else ''})", fontsize=6)
    ax = axes[i, 2]
    ax.plot(tc[f"c{j + 1}"], lw=0.6, color="#2a78d6")
    ax.set_title(f"component {j + 1}: {r['label']}, eta {r['eta']}, energy/null "
                 f"{r['energy_single'] / summ['null']['threshold']:.1f}, stability {r['stability_median_abs_r']}", fontsize=6.5, loc="left")
    ax.tick_params(labelsize=5)
fig.suptitle(f"{summ['run_id']} ({summ['source']}), k = {summ['k']}: components above the energy null", fontsize=8, x=0.01, ha="left")
fig.tight_layout()
out = kdir / "components.png"
fig.savefig(out, dpi=110)
print(out)
