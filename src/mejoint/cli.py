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
    elif args.cmd == "inventory-summary":
        from . import inventory
        inventory.write_summary(args.out, cfg)
        print(f"Summary: {args.out}/phase0_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
