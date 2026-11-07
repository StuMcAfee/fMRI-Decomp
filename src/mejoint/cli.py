"""Command-line entry point: `mejoint <command> ...`.

Commands: inventory, inventory-all, inventory-summary, frontend, frontend-all, mppca-recheck, nuisance-compare,
rank, tissue, decompose (optionally --waves), waves, mask-qc. `mejoint <command> --help` lists the options.
"""
from __future__ import annotations

import argparse
import sys

from . import io


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mejoint")
    sub = ap.add_subparsers(dest="cmd", required=True)

    inv = sub.add_parser("inventory", help="Phase 0 audit of one subject")
    inv.add_argument("--nifti-dir", help="folder with the raw echo NIfTIs + JSON sidecars "
                     "(default: the folder of input_echo1 in the run's result.json)")
    inv.add_argument("--run-dir", help="mbme_tedana output folder for this run (the folder holding result.json)")
    inv.add_argument("--physio-dir", action="append", default=[], help="extra folder to search for physio logs")
    inv.add_argument("--pipeline-script", help="path to mbme_tedana.py (its settings are reported)")
    inv.add_argument("--subject", help="subject label for the report")
    inv.add_argument("--out", required=True, help="output folder for the audit")
    inv.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    inv.add_argument("--no-data", action="store_true", help="skip loading voxel data (no noise-frame/NSS check)")

    alls = sub.add_parser("inventory-all", help="Phase 0 audit of every run folder in the mbme_tedana output")
    alls.add_argument("--pipeline-out", required=True, help="mbme_tedana --out folder (holds one folder per run)")
    alls.add_argument("--pipeline-script", help="path to mbme_tedana.py (its settings are reported)")
    alls.add_argument("--out", required=True, help="output folder (one subfolder per run + phase0_summary.*)")
    alls.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    alls.add_argument("--no-data", action="store_true", help="skip loading voxel data")

    summ = sub.add_parser("inventory-summary", help="rebuild the cross-run Phase 0 summary from existing audits")
    summ.add_argument("--out", required=True, help="folder holding <run>/inventory.json")
    summ.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    fr = sub.add_parser("frontend", help="Phase 2 front end on one mbme_tedana run (maps, channels, summary)")
    g = fr.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="mbme_tedana run folder (holds result.json, volumes_used.tsv, motion)")
    g.add_argument("--echo1", help="raw echo-1 NIfTI alone (no pipeline outputs; equal echo volume counts)")
    fr.add_argument("--source", choices=["raw", "mc"], default="raw", help="raw dcm2niix echoes or motion-corrected")
    fr.add_argument("--out", required=True, help="output folder")
    fr.add_argument("--no-mppca", action="store_true", help="skip the MP-PCA sigma cross-check")
    fr.add_argument("--no-nuisance", action="store_true", help="no-regression variant (skip the Phase 2b projector)")
    fr.add_argument("--mask", choices=["auto", "pipeline", "pipeline_filled"], default=None,
                    help="analysis mask: auto = built from the echoes (default); pipeline(_filled) = mbme_tedana's, for comparison")
    fr.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    fa = sub.add_parser("frontend-all", help="front end on every run + per-echo noise survey table")
    fa.add_argument("--pipeline-out", required=True, help="mbme_tedana --out folder")
    fa.add_argument("--out", required=True, help="output folder (one subfolder per run)")
    fa.add_argument("--source", choices=["raw", "mc"], default="raw")
    fa.add_argument("--mppca", action="store_true", help="also run the (slow) MP-PCA cross-check")
    fa.add_argument("--force", action="store_true", help="rerun runs already done with the same config (after a code change)")
    fa.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    rc = sub.add_parser("mppca-recheck", help="re-evaluate the MP-PCA verdict of existing front-end outputs (level/shape rule)")
    rc.add_argument("summaries", nargs="+", help="summary.json file(s) of front-end runs on raw echoes")
    rc.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    nc = sub.add_parser("nuisance-compare", help="compare regressor sets (6/12/18/24) on a no-regression front-end output")
    nc.add_argument("--frontend-dir", required=True, help="front-end output made WITHOUT regression (e.g. .../mc_noreg)")
    nc.add_argument("--run-dir", required=True, help="mbme_tedana run folder (motion/mcflirt.par, volumes_used.tsv)")
    nc.add_argument("--out", required=True, help="output folder")
    nc.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    dc = sub.add_parser("decompose", help="Phase 4 joint ICA on a front-end output: one k, or the fixed sweep")
    dc.add_argument("--frontend-dir", required=True, help="front-end output folder (channel cache)")
    dc.add_argument("--out", required=True, help="output folder (one subfolder per k)")
    dc.add_argument("--k", type=int, action="append", help="k (repeatable); default: MAPCA estimates from --rank-dir + decompose.k_sweep_fixed")
    dc.add_argument("--rank-dir", help="Phase 3 rank output (rank.json with the MAPCA estimates)")
    dc.add_argument("--tissue-dir", help="tissue classes (mejoint tissue output) for eta by class and the partial-volume diagnostic")

    dc.add_argument("--waves", action="store_true", help="also run the individual traveling-wave stage (writes <out>/waves/)")
    dc.add_argument("--wave-figures", action="store_true", help="with --waves: also write the two wave figures per plotted mode")
    dc.add_argument("--waves-frontend-dir", help="with --waves: front-end folder for the wave stage (default --frontend-dir; "
                    "waves.channels_primary is noreg, so pass the no-regression front end here)")
    _wave_args(dc, standalone=False)

    wv = sub.add_parser("waves", help="individual traveling-wave stage on a front-end output (docs/WAVES_BRIEF.md 6.1)")
    wv.add_argument("--frontend-dir", required=True, help="front-end output folder (channel cache)")
    wv.add_argument("--out", required=True, help="output folder (modes, maps, figures, summary)")
    wv.add_argument("--tissue-dir", help="tissue classes: the 'tissue' class becomes the reference voxels")
    wv.add_argument("--no-figures", action="store_true")
    wv.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    _wave_args(wv, standalone=True)

    wsum = sub.add_parser("waves-summary", help="cross-participant summary of a wave batch (outputs/waves/<run>/<commit>/)")
    wsum.add_argument("--root", default="outputs/waves", help="folder with one subfolder per run")
    wsum.add_argument("--commit", required=True, help="git commit of the batch (the subfolder name under each run)")
    wsum.add_argument("--out", required=True, help="output folder")
    wsum.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    swsum = sub.add_parser("sweep-summary", help="cross-participant summary of the joint-ICA sweep (outputs/phase5/<run>/phase4_sweep_mc/)")
    swsum.add_argument("--root", default="outputs/phase5", help="folder with one subfolder per run")
    swsum.add_argument("--out", required=True, help="output folder")
    swsum.add_argument("--k", type=int, default=50, help="common k for the per-component table (default 50)")
    swsum.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    er = sub.add_parser("eta-recheck", help="eta with empirical noise (level and autocorrelation) for every sweep folder")
    er.add_argument("--phase5-root", default="outputs/phase5", help="folder with one subfolder per run (sweeps)")
    er.add_argument("--phase2-root", default="outputs/phase2", help="folder with the front ends (<run>/mc)")
    er.add_argument("--k", type=int, action="append", help="only these k (repeatable; default every k folder)")
    er.add_argument("--max-lag", type=int, default=40, help="autocovariance lags in frames (default 40 = 30 s)")
    er.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    tk = sub.add_parser("te-check", help="what is the dS0 in the sweep components: sign and ratio, curvature, echo profiles")
    tk.add_argument("--phase5-root", default="outputs/phase5", help="folder with one subfolder per run (sweeps)")
    tk.add_argument("--phase2-root", default="outputs/phase2", help="folder with the front ends (<run>/mc)")
    tk.add_argument("--k", type=int, action="append", help="k (repeatable; default 50)")
    tk.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    ts = sub.add_parser("tissue", help="Phase 5 data-only tissue classes and partial-volume coupling from a front-end output")
    ts.add_argument("--frontend-dir", required=True)
    ts.add_argument("--out", required=True)
    ts.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    dc.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    rk = sub.add_parser("rank", help="Phase 3 rank of the stacked channels of one front-end output folder")
    rk.add_argument("--frontend-dir", required=True, help="front-end output folder (channel cache c0/c1/r0/r1.npy)")
    rk.add_argument("--out", required=True, help="output folder")
    rk.add_argument("--mapca-only", action="store_true", help="spectrum and MAPCA estimates only (fast; batches)")
    rk.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    mq = sub.add_parser("mask-qc", help="build the analysis mask for every run and tabulate it (raw echoes 1 and 4)")
    mq.add_argument("--pipeline-out", required=True, help="mbme_tedana --out folder")
    mq.add_argument("--out", required=True, help="output folder")
    mq.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    im = sub.add_parser("inject-make", help="Phase 5 injection: known sources added to the raw echoes of one run")
    im.add_argument("--run-dir", required=True, help="mbme_tedana run folder of the original run (result.json, volumes_used.tsv)")
    im.add_argument("--level", required=True, help="amplitude level (inject.levels in the config, e.g. strong, weak)")
    im.add_argument("--seed", type=int, default=1, help="placement and time-course seed (default 1)")
    im.add_argument("--out", required=True, help="injection folder (echoes/, truth.json, list.txt)")
    im.add_argument("--baseline-frontend", required=True, help="front end of the original run (analysis_mask.nii.gz)")
    im.add_argument("--tissue-dir", help="tissue classes of the original run (sources kept out of CSF and dropout)")
    im.add_argument("--layout", choices=["focal", "network"], help="one sphere per source, or networks of spheres (default: inject.layout)")
    im.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    ir = sub.add_parser("inject-recover", help="Phase 5 injection: find the injected sources in a decomposition")
    ir.add_argument("--inject-dir", required=True, help="injection folder (truth.json)")
    ir.add_argument("--frontend-dir", required=True, help="front end of the injected echoes")
    ir.add_argument("--k-dir", required=True, help="decomposition folder of the injected echoes (one k)")
    ir.add_argument("--mbme-run", help="mbme_tedana run folder of the injected echoes (tedana comparison, pairing check)")
    ir.add_argument("--baseline-k-dir", help="the same k of the original run (do the other components survive?)")
    ir.add_argument("--no-te-check", action="store_true", help="skip the te-check profile and curvature")
    ir.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    isu = sub.add_parser("inject-summary", help="Phase 5 injection: pool every recovery.json under a folder")
    isu.add_argument("--root", default="outputs/inject", help="folder searched for recovery.json")
    isu.add_argument("--out", help="output folder (default: --root)")
    isu.add_argument("--config", help="config YAML (default: configs/default.yaml)")

    args = ap.parse_args(argv)
    if args.cmd == "inventory" and not (args.nifti_dir or args.run_dir):
        ap.error("give --run-dir (preferred) or --nifti-dir")
    cfg = io.load_config(args.config)

    if args.cmd == "inventory":
        from . import inventory
        res = inventory.inventory_subject(args.nifti_dir, cfg, run_dir=args.run_dir,
                                          physio_dirs=args.physio_dir, subject=args.subject,
                                          load_data=not args.no_data,
                                          pipeline_script=args.pipeline_script)
        inventory.write_outputs(res, cfg, args.out)
        print(inventory.render_markdown(res, cfg))
        print(f"\nWritten to {args.out}")
    elif args.cmd == "inventory-all":
        from . import inventory
        status = inventory.inventory_all(args.pipeline_out, cfg, args.out, pipeline_script=args.pipeline_script,
                                         load_data=not args.no_data)
        n_fail = sum(s["status"].startswith("FAILED") for s in status)
        print(f"\nSummary: {args.out}/phase0_summary.md ({n_fail} failed)")
        return 1 if n_fail else 0
    elif args.cmd == "frontend":
        from . import frontend
        summ = frontend.run_subject(args.run_dir, cfg, args.out, source=args.source,
                                    mppca=False if args.no_mppca else None, mask_source=args.mask, echo1=args.echo1,
                                    nuisance=False if args.no_nuisance else None)
        print(frontend.render_summary(summ))
        if "mppca_check" in summ and summ["mppca_check"]["verdict"] != "OK":
            return 3
    elif args.cmd == "frontend-all":
        from . import frontend
        rows = frontend.run_all(args.pipeline_out, cfg, args.out, source=args.source, mppca=args.mppca,
                               force=args.force)
        print(f"Survey: {args.out}/noise_survey.md")
        return 1 if any("error" in r for r in rows) else 0
    elif args.cmd == "mppca-recheck":
        from . import frontend
        status = 0
        for f in args.summaries:
            m = frontend.recheck_mppca(f, cfg)["mppca_check"]
            print(f"{f}\n  level {m['level']}, shape {m['shape']}, implied c0-c1 noise corr "
                  f"{m.get('implied_c0_c1_noise_correlation')}\n  verdict: {m['verdict']}")
            status = status or (0 if m["verdict"] == "OK" else 3)
        return status
    elif args.cmd == "nuisance-compare":
        from . import nuisance
        nuisance.compare_expansions(args.frontend_dir, args.run_dir, cfg, args.out)
        print(open(f"{args.out}/nuisance_compare.md").read())
    elif args.cmd == "decompose":
        from . import decompose
        rows = decompose.run_sweep(args.frontend_dir, cfg, args.out, rank_dir=args.rank_dir, ks=args.k, tissue_dir=args.tissue_dir)
        print(open(f"{args.out}/sweep.md").read())
        if args.waves:
            from pathlib import Path

            from . import waves
            waves.run_dir(args.waves_frontend_dir or args.frontend_dir, _wave_cfg(cfg, args), Path(args.out) / "waves",
                          tissue_dir=args.tissue_dir, figures=args.wave_figures)
            print(open(Path(args.out) / "waves" / "waves_summary.md").read())
    elif args.cmd == "waves":
        from pathlib import Path

        from . import waves
        waves.run_dir(args.frontend_dir, _wave_cfg(cfg, args), args.out, tissue_dir=args.tissue_dir,
                      figures=not args.no_figures)
        print(open(Path(args.out) / "waves_summary.md").read())
    elif args.cmd == "waves-summary":
        from pathlib import Path

        from . import waves_batch
        waves_batch.run(args.root, args.commit, args.out)
        print(open(Path(args.out) / "waves_batch.md").read())
    elif args.cmd == "sweep-summary":
        from pathlib import Path

        from . import sweep_batch
        sweep_batch.run(args.root, args.out, k_main=args.k)
        print(open(Path(args.out) / "sweep_batch.md").read())
    elif args.cmd == "eta-recheck":
        from . import eta_recheck
        rows = eta_recheck.run(args.phase5_root, args.phase2_root, ks=args.k, max_lag=args.max_lag)
        print(f"{len(rows)} sweep folders rechecked; eta_recheck.json written in each; run sweep-summary to tabulate")
    elif args.cmd == "te-check":
        from . import te_check
        rows = te_check.run(args.phase5_root, args.phase2_root, ks=tuple(args.k or [50]))
        print(f"{len(rows)} sweep folders checked; te_check.json and te_profiles.png in each; run sweep-summary to tabulate")
    elif args.cmd == "tissue":
        from . import tissue
        tissue.run_dir(args.frontend_dir, cfg, args.out)
        print(open(f"{args.out}/tissue.md").read())
    elif args.cmd == "rank":
        from . import rank
        if args.mapca_only:
            rank.run_dir_mapca(args.frontend_dir, cfg, args.out)
            print(open(f"{args.out}/rank.md").read())
        else:
            res = rank.run_dir(args.frontend_dir, cfg, args.out)
            print(rank.render(res))
    elif args.cmd == "mask-qc":
        from . import mask
        rows = mask.qc_all(args.pipeline_out, cfg, args.out)
        return 1 if any("error" in r for r in rows) else 0
    elif args.cmd == "inject-make":
        from . import inject
        tr = inject.make_dir(args.run_dir, cfg, args.level, args.seed, args.out, args.baseline_frontend, args.tissue_dir,
                             layout=args.layout)
        print(f"{len(tr['sources'])} sources; injected echo 1: {tr['echo1_injected']}; list for mbme_tedana: {args.out}/list.txt")
    elif args.cmd == "inject-recover":
        from . import inject
        inject.recover_dir(args.inject_dir, args.frontend_dir, args.k_dir, mbme_run=args.mbme_run,
                           baseline_k_dir=args.baseline_k_dir, te_check=not args.no_te_check)
        print(open(f"{args.inject_dir}/recovery.md").read())
    elif args.cmd == "inject-summary":
        from pathlib import Path

        from . import inject
        inject.summary(args.root, args.out)
        print(open(Path(args.out or args.root) / "inject_summary.md").read())
    elif args.cmd == "inventory-summary":
        from . import inventory
        inventory.write_summary(args.out, cfg)
        print(f"Summary: {args.out}/phase0_summary.md")
    return 0


def _wave_args(p, standalone: bool) -> None:
    """Wave-stage overrides shared by `mejoint waves` and `mejoint decompose --waves` (one code path)."""
    pre = "" if standalone else "wave-"
    p.add_argument(f"--{pre}band", type=float, nargs=2, metavar=("LOW", "HIGH"), dest="w_band", help="band (Hz)")
    p.add_argument(f"--{pre}n-modes", type=int, dest="w_n_modes")
    p.add_argument(f"--{pre}plot-modes", type=int, dest="w_plot_modes")
    p.add_argument(f"--{pre}voxel-scaling", choices=["whitened", "zscore_c0"], dest="w_voxel_scaling")
    p.add_argument(f"--{pre}channels", choices=["both", "c0"], dest="w_channels")
    p.add_argument(f"--{pre}bootstrap", type=int, dest="w_bootstrap")
    p.add_argument(f"--{pre}surrogates", type=int, dest="w_surrogates")
    p.add_argument(f"--{pre}motion-orthogonal", action=argparse.BooleanOptionalAction, default=None, dest="w_motion_orth",
                   help="remove the band-passed motion parameters from the band-passed channels first (default: "
                        "waves.motion_orthogonal, true; --no-...motion-orthogonal for the comparison run)")
    p.add_argument("--no-slice-timing", action="store_true", dest="w_no_stc",
                   help="wave stage without slice-timing correction (a sensitivity check; the artifact is inferior-superior)")


def _wave_cfg(cfg: dict, args) -> dict:
    over = {"band_hz": args.w_band, "n_modes": args.w_n_modes, "plot_modes": args.w_plot_modes,
            "voxel_scaling": args.w_voxel_scaling, "channels": args.w_channels, "bootstrap": args.w_bootstrap,
            "surrogates": args.w_surrogates}
    over = {k: (list(v) if isinstance(v, tuple) else v) for k, v in over.items() if v is not None}
    if args.w_no_stc:
        over["slice_timing_correction"] = False
    if args.w_motion_orth is not None:
        over["motion_orthogonal"] = bool(args.w_motion_orth)
    if "plot_modes" in over:
        over.setdefault("bootstrap_modes", over["plot_modes"])
    return io.merge(cfg, {"waves": over})


if __name__ == "__main__":
    sys.exit(main())
