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
    inv.add_argument("--nifti-dir", required=True, help="folder with the raw echo NIfTIs + JSON sidecars")
    inv.add_argument("--run-dir", help="mbme_tedana output folder for this run (optional)")
    inv.add_argument("--physio-dir", action="append", default=[], help="extra folder to search for physio logs")
    inv.add_argument("--pipeline-script", help="path to mbme_tedana.py (its settings are reported)")
    inv.add_argument("--subject", help="subject label for the report")
    inv.add_argument("--out", required=True, help="output folder for the audit")
    inv.add_argument("--config", help="config YAML (default: configs/default.yaml)")
    inv.add_argument("--no-data", action="store_true", help="skip loading voxel data (no noise-frame/NSS check)")

    args = ap.parse_args(argv)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
