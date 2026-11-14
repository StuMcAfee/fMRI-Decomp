"""Exploratory comparison of the route B measures with a clinical label (user, 2026-11-12).

The clinical table is read from wherever the user keeps it and must not be inside the repository (participant data
never go into git); results are written under outputs/ (a link to the research storage, ignored by git). Clinical
variables are only compared with the measures here; they never define regions or measures (CLAUDE.md section 1).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
METRICS = ("pr_without_first", "global_share", "pr", "segregation", "cortex_csf_fraction", "reliability_median", "fd_mean")


def _inside_repo(p: Path) -> bool:
    """True if p lies in the repository tree but not in outputs/ (which links to the research storage)."""
    p = p.resolve()
    try:
        rel = p.relative_to(REPO.resolve())
    except ValueError:
        return False
    return not (rel.parts and rel.parts[0] == "outputs")


def read_table(path, subject_col: str = "subject", label_col: str = "DX") -> dict:
    """{subject number (str): label (str)} from .xlsx (first sheet), .csv or .tsv. Refuses a table kept inside
    the repository."""
    p = Path(path)
    if _inside_repo(p):
        raise ValueError(f"{p} is inside the repository; keep participant data outside it (e.g. on the research storage)")
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError("openpyxl is not installed; save the table as .csv or .tsv") from exc
        ws = openpyxl.load_workbook(p, read_only=True, data_only=True).worksheets[0]
        rows = [[("" if c is None else str(c)).strip() for c in r] for r in ws.iter_rows(values_only=True)]
    else:
        with open(p, newline="") as fh:
            rows = list(csv.reader(fh, delimiter="\t" if p.suffix.lower() == ".tsv" else ","))
    head = [h.strip() for h in rows[0]]
    if subject_col not in head or label_col not in head:
        raise ValueError(f"columns {subject_col!r} and {label_col!r} needed; found {head}")
    i, j = head.index(subject_col), head.index(label_col)
    out = {}
    for r in rows[1:]:
        if len(r) > max(i, j) and r[i] and r[j]:
            s = r[i].split(".")[0]                                     # 12345.0 from a spreadsheet -> 12345
            lab = r[j].split(".")[0] if r[j].replace(".", "", 1).isdigit() else r[j]
            out[s] = lab
    return out


def parse_contrast(text: str):
    """'severe:1,2,3,4 vs 0' -> ('severe', {'1','2','3','4'}, {'0'}); 'mutism:1 vs rest' -> rest = every other label."""
    name, spec = text.split(":", 1)
    a, b = [x.strip() for x in spec.split(" vs ")]
    A = {x.strip() for x in a.split(",") if x.strip()}
    B = None if b == "rest" else {x.strip() for x in b.split(",") if x.strip()}
    return name.strip(), A, B


def compare(participants_tsv, table_path, out_dir, contrasts=(), subject_col: str = "subject", label_col: str = "DX",
            sensitivity_tsv=None) -> dict:
    """Group medians by label, exact Mann-Whitney tests for each contrast, and the same after adjusting each measure
    for mean FD (residuals of a straight-line fit), over the participants in the primary."""
    from scipy.stats import mannwhitneyu

    from . import io
    out = Path(out_dir)
    if _inside_repo(out):
        raise ValueError(f"{out} is inside the repository; write results under outputs/")
    labels = read_table(table_path, subject_col, label_col)
    with open(participants_tsv) as fh:
        parts = list(csv.DictReader(fh, delimiter="\t"))
    sens = {}
    if sensitivity_tsv and Path(sensitivity_tsv).exists():
        with open(sensitivity_tsv) as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                sens[r["run"]] = r
    num = lambda v: float(v) if v not in (None, "", "None", "nan") else np.nan
    rows = []
    for p in parts:
        subj = p["run"].split("_")[0].replace("sub-", "")
        if p.get("in_primary") not in ("True", "true", "1") or subj not in labels:
            continue
        r = {"subject": subj, "label": labels[subj], **{m: num(p.get(m)) for m in METRICS}}
        for k, v in sens.get(p["run"], {}).items():
            if k not in ("run", "in_primary", "main"):
                r[f"sens_{k}"] = num(v)
        rows.append(r)
    metrics = list(METRICS) + sorted({k for r in rows for k in r if k.startswith("sens_")})
    levels = sorted({r["label"] for r in rows}, key=lambda x: (len(x), x))
    by_level = {lv: {"n": sum(r["label"] == lv for r in rows),
                     **{m: _med([r[m] for r in rows if r["label"] == lv]) for m in metrics}} for lv in levels}
    fd = np.array([r["fd_mean"] for r in rows])
    tests = []
    for c in contrasts:
        name, A, B = parse_contrast(c)
        ina = np.array([r["label"] in A for r in rows])
        inb = np.array([(r["label"] not in A) if B is None else (r["label"] in B) for r in rows])
        res = {"contrast": c, "n_a": int(ina.sum()), "n_b": int(inb.sum()), "metrics": {}}
        for m in metrics:
            x = np.array([r[m] for r in rows])
            ok = np.isfinite(x)
            entry = {"median_a": _med(x[ina & ok]), "median_b": _med(x[inb & ok])}
            if (ina & ok).sum() >= 1 and (inb & ok).sum() >= 1 and (ina & ok).sum() + (inb & ok).sum() >= 4:
                entry["p"] = round(float(mannwhitneyu(x[ina & ok], x[inb & ok], method="exact").pvalue), 4)
                okf = ok & np.isfinite(fd)
                if m != "fd_mean" and okf.sum() >= 5:
                    A_ = np.c_[np.ones(okf.sum()), fd[okf]]
                    resid = np.full(x.shape, np.nan)
                    resid[okf] = x[okf] - A_ @ np.linalg.lstsq(A_, x[okf], rcond=None)[0]
                    if (ina & okf).sum() and (inb & okf).sum():
                        entry["p_fd_adjusted"] = round(float(mannwhitneyu(resid[ina & okf], resid[inb & okf], method="exact").pvalue), 4)
            res["metrics"][m] = entry
        tests.append(res)
    n_tests = sum(len(t["metrics"]) for t in tests)
    result = {"table": str(table_path), "label_column": label_col, "n_participants": len(rows), "levels": by_level,
              "contrasts": tests, "n_tests": n_tests, "participants": rows}
    out.mkdir(parents=True, exist_ok=True)
    io.write_json(out / "clinical_compare.json", result)
    io.write_text(out / "clinical_compare.md", render(result, metrics))
    return result


def _med(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return round(float(np.median(v)), 4) if v.size else None


def render(r: dict, metrics) -> str:
    L = [f"# Exploratory comparison with {r['label_column']} ({r['n_participants']} participants in the primary)", "",
         f"Exploratory only: {r['n_tests']} tests, small groups, no correction for multiple comparisons. Exact Mann-Whitney "
         "p; 'FD-adjusted' = the same on residuals after a straight-line fit on mean FD.", "",
         "## Medians by label", "", "| label | n | " + " | ".join(metrics) + " |", "|---|---|" + "---|" * len(metrics)]
    for lv, d in r["levels"].items():
        L.append(f"| {lv} | {d['n']} | " + " | ".join(str(d[m]) for m in metrics) + " |")
    for t in r["contrasts"]:
        L += ["", f"## {t['contrast']} (n = {t['n_a']} vs {t['n_b']})", "", "| measure | median A | median B | p | p FD-adjusted |", "|---|---|---|---|---|"]
        for m, e in t["metrics"].items():
            L.append(f"| {m} | {e['median_a']} | {e['median_b']} | {e.get('p')} | {e.get('p_fd_adjusted')} |")
    return "\n".join(L) + "\n"
