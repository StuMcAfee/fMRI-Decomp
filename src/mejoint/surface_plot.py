"""Parcel values on the cortical surface (fs_LR 32k inflated template), for the parcel ICA figures.

Parcel values are painted onto the template surface by parcel identity: every vertex takes the value of the parcel
it belongs to in the surface version of the same atlas (Schaefer 200/7 dlabel; Icosahedron1002 labels). For route B's
volumetric Schaefer parcels this is a display of parcel values, not a projection of the participant's data onto
their own surface. Rendering is a plain orthographic projection with flat shading (matplotlib only, no OpenGL), so it
runs in batch jobs.

Figures:
  weight maps   each component's map, z-scored over parcels, on four views (left / right, lateral / medial);
  winner-take-all  each parcel coloured by the component with the largest z-scored loading (if above a threshold),
                   hue = the Yeo-7 network the component overlaps most (atlas colours), lightness steps separate
                   components of the same network, flagged (candidate physiological) components in dark grey,
                   component borders darkened; next to the Yeo-7 reference map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NETS7 = ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")
MEDIAL = np.array([0.80, 0.80, 0.78, 1.0])        # vertices without a parcel (medial wall)
NOWIN = np.array([0.93, 0.93, 0.91, 1.0])         # parcels without a clear winner
FLAGGED = np.array([0.33, 0.33, 0.31, 1.0])       # candidate physiological components
VIEWS = (("L", "lateral"), ("L", "medial"), ("R", "lateral"), ("R", "medial"))


# ------------------------------------------------------------------ resources
def load_resources(resources, atlas: str) -> dict:
    """Inflated surfaces and per-vertex global parcel ids for the atlas ('200_7' or 'ico1002_7'), plus the Yeo-7
    network of every vertex and the atlas network colours (from the Schaefer dlabel)."""
    import nibabel as nib

    from . import surface
    R = Path(resources)
    surf = {}
    for h in ("L", "R"):
        f = R / f"tpl-fs32k_hemi-{h}_inflated.surf.gii"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing (run scripts/fetch_fs32k_resources.sh)")
        g = nib.load(str(f))
        surf[h] = (np.asarray(g.darrays[0].data, float), np.asarray(g.darrays[1].data, int))
    dl = R / "Schaefer2018_200Parcels_7Networks_order.dlabel.nii"
    sl, sr, names = surface.schaefer_from_dlabel(dl)
    lt = nib.load(str(dl)).header.get_axis(0).label[0]
    colours = {}
    for k, (nm, rgba) in lt.items():
        n = surface.network_of(nm) if int(k) > 0 else ""
        if n in NETS7 and n not in colours:
            colours[n] = np.array(rgba, float)
    netv = {h: np.array([surface.network_of(names[i]) if i > 0 else "" for i in lab]) for h, lab in (("L", sl), ("R", sr))}
    if atlas.startswith("ico"):
        il = np.asarray(nib.load(str(R / "Icosahedron1002.L.label.gii")).darrays[0].data, int)
        ir = np.asarray(nib.load(str(R / "Icosahedron1002.R.label.gii")).darrays[0].data, int)
        labels = {"L": il, "R": np.where(ir > 0, ir + 938, 0)}
    elif atlas == "200_7":
        labels = {"L": sl, "R": sr}
    else:
        raise ValueError(f"no surface version of atlas {atlas} in the resources")
    return {"surf": surf, "labels": labels, "net_vertex": netv, "net_colours": colours, "schaefer_names": names}


# ------------------------------------------------------------------ rendering
def _camera(h: str, view: str):
    d = np.array([-1.0, 0, 0]) if (h == "L") == (view == "lateral") else np.array([1.0, 0, 0])   # towards the camera
    up = np.array([0, 0, 1.0])
    right = np.cross(up, d)
    return d, up, right


def draw(ax, coords, faces, vcolour, h: str, view: str):
    """Orthographic flat-shaded rendering of one hemisphere: back faces culled, painter's order, Lambert shading."""
    from matplotlib.collections import PolyCollection
    d, up, right = _camera(h, view)
    tri = coords[faces]                                                # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    cen = tri.mean(1)
    if np.mean(np.einsum("fi,fi->f", n, cen - coords.mean(0))) < 0:   # make normals point outwards
        n = -n
    facing = n @ d
    keep = facing > 0
    depth = cen[keep] @ d
    order = np.argsort(depth)
    t2 = np.stack([tri[keep] @ right, tri[keep] @ up], -1)[order]
    fc = vcolour[faces[keep]].mean(1)[order]
    shade = (0.45 + 0.55 * facing[keep][order])[:, None]
    fc = np.concatenate([np.clip(fc[:, :3] * shade, 0, 1), np.ones((len(fc), 1))], 1)
    ax.add_collection(PolyCollection(t2, facecolors=fc, edgecolors=fc, linewidths=0.15, antialiased=False))
    ax.set_xlim(t2[..., 0].min(), t2[..., 0].max())
    ax.set_ylim(t2[..., 1].min(), t2[..., 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def diverging_cmap():
    """Blue <-> red with a neutral grey midpoint (two hues, no hue at zero)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("bwr_neutral", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#8a1f1e"])


def vertex_values(values_by_id: dict, labels: np.ndarray) -> np.ndarray:
    v = np.full(labels.shape, np.nan)
    for pid, val in values_by_id.items():
        v[labels == pid] = val
    return v


def zscore(maps: np.ndarray) -> np.ndarray:
    return (maps - maps.mean(0)) / np.maximum(maps.std(0), 1e-12)


# ------------------------------------------------------------------ figures
def weight_maps(maps, parcel_ids, res: dict, path_stem, titles, vmax: float = 3.0, per_page: int = 10) -> list:
    """Z-scored component maps on four views, per_page components per figure (path_stem_p<i>.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    cmap, norm = diverging_cmap(), Normalize(-vmax, vmax)
    Z = zscore(np.asarray(maps, float))
    k = Z.shape[1]
    files = []
    for page, start in enumerate(range(0, k, per_page)):
        comps = list(range(start, min(k, start + per_page)))
        fig = plt.figure(figsize=(11.0, 1.45 * len(comps) + 0.6))
        gs = fig.add_gridspec(len(comps), 6, width_ratios=[0.62, 1, 1, 1, 1, 0.06], hspace=0.12, wspace=0.02,
                              left=0.01, right=0.93, top=0.96, bottom=0.02)
        for row, j in enumerate(comps):
            vals = dict(zip(parcel_ids, Z[:, j]))
            for col, (h, view) in enumerate(VIEWS):
                vv = vertex_values(vals, res["labels"][h])
                vc = np.where(np.isfinite(vv)[:, None], cmap(norm(np.nan_to_num(vv))), MEDIAL)
                ax = fig.add_subplot(gs[row, col + 1])
                draw(ax, *res["surf"][h], vc, h, view)
                if row == 0:
                    ax.set_title(f"{'left' if h == 'L' else 'right'} {view}", fontsize=8, color="#444")
            tax = fig.add_subplot(gs[row, 0])
            tax.axis("off")
            tax.text(0.0, 0.5, titles[j], fontsize=8, va="center", ha="left", color="#222", transform=tax.transAxes)
        cax = fig.add_subplot(gs[:, 5])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("loading (z over parcels)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        f = Path(f"{path_stem}_p{page + 1}.png")
        fig.savefig(f, dpi=130)
        plt.close(fig)
        files.append(str(f))
    return files


def component_networks(maps, parcel_ids, parcel_networks) -> list:
    """Each component's best Yeo-7 network (largest correlation of the z-scored map with the network indicator; many
    components may share one) and that correlation."""
    Z = zscore(np.asarray(maps, float))
    nets = np.asarray(parcel_networks)
    out = []
    for j in range(Z.shape[1]):
        r = {n: float(np.corrcoef(Z[:, j], (nets == n).astype(float))[0, 1]) if (nets == n).any() else -1 for n in NETS7}
        n = max(r, key=r.get)
        out.append((n, r[n]))
    return out


def winner_take_all(maps, parcel_ids, parcel_networks, res: dict, path, flagged=None, z_min: float = 1.0,
                    title: str = "") -> dict:
    """Winner-take-all map: each parcel's component with the largest z-scored loading (shown only if z >= z_min),
    next to the Yeo-7 reference. Returns the assignment per parcel and the component legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    Z = zscore(np.asarray(maps, float))
    k = Z.shape[1]
    flagged = np.zeros(k, bool) if flagged is None else np.asarray(flagged, bool)
    win = np.argmax(Z, axis=1)
    zwin = Z[np.arange(len(win)), win]
    win = np.where(zwin >= z_min, win, -1)
    cn = component_networks(maps, parcel_ids, parcel_networks)
    # colour per component: network hue, lightness step for the 2nd, 3rd ... component of the same network
    steps = (0.0, 0.45, -0.35, 0.7, -0.55, 0.25, -0.2)
    seen, col = {}, np.zeros((k, 4))
    for j in range(k):
        if flagged[j]:
            col[j] = FLAGGED
            continue
        n = cn[j][0]
        i = seen.get(n, 0)
        seen[n] = i + 1
        base = res["net_colours"].get(n, np.array([0.5, 0.5, 0.5, 1]))[:3]
        s = steps[i % len(steps)]
        rgb = base + (1 - base) * s if s >= 0 else base * (1 + s)
        col[j] = np.r_[rgb, 1.0]
    fig = plt.figure(figsize=(11.5, 4.6 + 0.08 * max(0, k - 20)))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1, 1.35], wspace=0.02, hspace=0.18, left=0.01, right=0.99, top=0.86, bottom=0.02)
    pid = np.asarray(parcel_ids)
    for row, kind in enumerate(("wta", "yeo")):
        for c, (h, view) in enumerate(VIEWS):
            lab = res["labels"][h]
            vc = np.tile(MEDIAL, (lab.size, 1))
            if kind == "wta":
                comp_v = np.full(lab.size, -2)
                for p, w in zip(pid, win):
                    comp_v[lab == p] = w
                vc[comp_v == -1] = NOWIN
                m = comp_v >= 0
                vc[m] = col[comp_v[m]]
                faces = res["surf"][h][1]
                fv = comp_v[faces]
                border = (fv.min(1) != fv.max(1)) & (fv.max(1) >= 0)
                vc[np.unique(faces[border])] *= np.array([0.55, 0.55, 0.55, 1])   # darken component borders
            else:
                nv = res["net_vertex"][h]
                for n in NETS7:
                    vc[nv == n] = res["net_colours"][n]
            ax = fig.add_subplot(gs[row, c])
            draw(ax, *res["surf"][h], vc, h, view)
            if row == 0:
                ax.set_title(f"{'left' if h == 'L' else 'right'} {view}", fontsize=8, color="#444")
        fig.text(0.005, 0.89 - row * 0.44, "winner-take-all (components)" if kind == "wta" else "Yeo-7 reference (atlas)", fontsize=9, color="#222")
    lax = fig.add_subplot(gs[:, 4])
    lax.axis("off")
    handles = [Patch(color=col[j], label=f"c{j}: {cn[j][0]} (r {cn[j][1]:.2f}){' flagged' if flagged[j] else ''}") for j in range(k)
               if (win == j).any()]
    handles += [Patch(color=NOWIN, label=f"no component z >= {z_min:g}"), Patch(color=MEDIAL, label="no parcel (medial wall)")]
    lax.legend(handles=handles, loc="upper left", fontsize=6.5 if len(handles) <= 30 else 5, frameon=False, ncol=1 if len(handles) <= 30 else 2,
               title="component: best Yeo-7 network", title_fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"winner": [int(w) for w in win], "component_network": [(n, round(r, 3)) for n, r in cn],
            "n_parcels_no_winner": int((win < 0).sum()), "n_components_winning": int(len(set(win[win >= 0].tolist())))}


# ------------------------------------------------------------------ generic rows (pipeline report)
def vertex_colours(values_by_id: dict, labels: np.ndarray, cmap, norm, missing=MEDIAL) -> np.ndarray:
    """Per-vertex RGBA from parcel values (parcels without a value: `missing`)."""
    vv = vertex_values(values_by_id, labels)
    return np.where(np.isfinite(vv)[:, None], cmap(norm(np.nan_to_num(vv))), missing)


def label_colours(colour_by_id: dict, labels: np.ndarray, missing=MEDIAL) -> np.ndarray:
    """Per-vertex RGBA from a colour per parcel id."""
    vc = np.tile(np.asarray(missing, float), (labels.size, 1))
    for pid, c in colour_by_id.items():
        vc[labels == pid] = c
    return vc


def rows_figure(res: dict, rows: list, path, legend=None, cbar=None, row_height: float = 1.55, width: float = 10.0,
                dpi: int = 110, titles_left: bool = True) -> None:
    """Several rows of four views; rows = [(title, {"L": vertex colours, "R": vertex colours}), ...]; optional legend
    handles (right) or colour bar (cmap, norm, label)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(rows)
    extra = 0.32 if (legend or cbar) else 0.0
    fig = plt.figure(figsize=(width * (1 + extra * 0.5), row_height * n + 0.35))
    wr = ([0.55] if titles_left else []) + [1, 1, 1, 1] + ([0.9] if legend else [0.06] if cbar else [])
    gs = fig.add_gridspec(n, len(wr), width_ratios=wr, hspace=0.08, wspace=0.02, left=0.01, right=0.97 if cbar else 0.99,
                          top=1 - 0.3 / (row_height * n + 0.35), bottom=0.01)
    off = 1 if titles_left else 0
    for i, (title, vcs) in enumerate(rows):
        if titles_left:
            tax = fig.add_subplot(gs[i, 0])
            tax.axis("off")
            tax.text(0, 0.5, title, fontsize=8, va="center", ha="left", color="#222", transform=tax.transAxes)
        for c, (h, view) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[i, c + off])
            draw(ax, *res["surf"][h], vcs[h], h, view)
            if i == 0:
                ax.set_title(f"{'left' if h == 'L' else 'right'} {view}", fontsize=7, color="#555")
    if legend:
        lax = fig.add_subplot(gs[:, -1])
        lax.axis("off")
        lax.legend(handles=legend, loc="upper left", fontsize=6.5 if len(legend) <= 26 else 5.2, frameon=False,
                   ncol=1 if len(legend) <= 26 else 2)
    elif cbar:
        cax = fig.add_subplot(gs[:, -1])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=cbar[1], cmap=cbar[0]), cax=cax)
        cb.set_label(cbar[2], fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def network_colours(resources, dlabel_name: str) -> dict:
    """Network name -> RGBA from a Schaefer dlabel's label table (the atlas colours)."""
    import nibabel as nib

    from . import surface
    lt = nib.load(str(Path(resources) / dlabel_name)).header.get_axis(0).label[0]
    out = {}
    for k, (nm, rgba) in lt.items():
        n = surface.network_of(nm) if int(k) > 0 else ""
        if n and n not in out:
            out[n] = np.array(rgba, float)
    return out
