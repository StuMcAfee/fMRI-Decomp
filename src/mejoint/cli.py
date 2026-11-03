"""Command-line entry point: `mejoint <command> ...`.

Commands so far:
  inventory   Phase 0 audit of one subject (writes inventory.md, inventory.json, volume_means.tsv, done.json)
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

    mq = sub.add_parser("mask-qc", help="build the analysis mask for every run and tabulate it (raw echoes 1 and 4)")
    mq.add_argument("--pipeline-out", required=True, help="mbme_tedana --out folder")
    mq.add_argument("--out", required=True, help="output folder")
    mq.add_argument("--config", help="config YAML (default: configs/default.yaml)")

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
                                    mppca=False if args.no_mppca else None, mask_source=args.mask, echo1=args.echo1)
        print(frontend.render_summary(summ))
        if "mppca_check" in summ and summ["mppca_check"]["verdict"] != "OK":
            return 3
    elif args.cmd == "frontend-all":
        from . import frontend
        rows = frontend.run_all(args.pipeline_out, cfg, args.out, source=args.source, mppca=args.mppca,
                               force=args.force)
        print(f"Survey: {args.out}/noise_survey.md")
        return 1 if any("error" in r for r in rows) else 0
    elif args.cmd == "mask-qc":
        from . import mask
        rows = mask.qc_all(args.pipeline_out, cfg, args.out)
        return 1 if any("error" in r for r in rows) else 0
    elif args.cmd == "inventory-summary":
        from . import inventory
        inventory.write_summary(args.out, cfg)
        print(f"Summary: {args.out}/phase0_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
