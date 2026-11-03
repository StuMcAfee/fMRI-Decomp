"""Contact sheet of the analysis masks across runs: one row per run, four axial slices from the bottom
of the head upwards and a mid-sagittal slice of the echo-1 mean, with the analysis mask (orange) and
the pipeline mask (blue dashed). Usage: PYTHONPATH=src python scripts/plot_mask_qc.py <mask_qc folder> [OUT.png]"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

SURF, INK2 = "#fcfcfb", "#52514e"


def canon(path):
    return np.asarray(nib.as_closest_canonical(nib.load(str(path))).dataobj, dtype=np.float32)


def main(argv=None):
    argv = argv or sys.argv[1:]
    folder = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else folder / "mask_qc.png"
    rows = [r for r in csv.DictReader(open(folder / "mask_qc.tsv"), delimiter="\t") if not r.get("error")]
    fig, axes = plt.subplots(len(rows), 5, figsize=(9, 1.85 * len(rows) + 0.4), squeeze=False)
    fig.patch.set_facecolor(SURF)
    for i, r in enumerate(rows):
        m1 = canon(folder / r["run"] / "mean_echo1.nii.gz")
        mk = canon(folder / r["run"] / "mask.nii.gz") > 0.5
        pm = canon(r["pipeline_mask"]) > 0.5 if r["pipeline_mask"] and Path(r["pipeline_mask"]).exists() else None
        zs = np.where(mk.any(axis=(0, 1)))[0]
        cuts = [int(zs.min() + f * (zs.max() - zs.min())) for f in (0.08, 0.22, 0.45, 0.7)]
        xm = int(np.mean(np.where(mk.any(axis=(1, 2)))[0]))
        vmax = np.percentile(m1[mk], 99)
        for j in range(5):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            sl = (lambda v: v[:, :, cuts[j]].T) if j < 4 else (lambda v: v[xm, :, :].T)
            ax.imshow(sl(m1), origin="lower", cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
            ax.contour(sl(mk).astype(float), levels=[0.5], colors="#eb6834", linewidths=0.7)
            if pm is not None:
                ax.contour(sl(pm).astype(float), levels=[0.5], colors="#3987e5", linewidths=0.5, linestyles="dashed")
            if j == 0:
                lab = r["run"].split("_acq")[0] + f"\n{float(r['volume_ml']):.0f} ml"
                if r.get("pipeline_volume_ml"):
                    lab += f" (pipeline {float(r['pipeline_volume_ml']):.0f})"
                ax.set_ylabel(lab, fontsize=7, color=INK2, rotation=0, ha="right", va="center", labelpad=4)
            if i == 0:
                ax.set_title(f"axial z={cuts[j]}" if j < 4 else "sagittal", fontsize=7, color=INK2)
    fig.suptitle("Analysis mask (orange) vs pipeline mask (blue dashed), echo-1 mean", fontsize=8, color=INK2, y=0.999)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURF)
    print(out)


if __name__ == "__main__":
    main()
