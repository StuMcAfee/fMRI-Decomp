# fMRI-Decomp (`mejoint`)

Physics-constrained joint decomposition of multi-echo fMRI into components with a dS0 map and a dR2* map.
The design brief is `CLAUDE.md`; gate results are in `docs/decisions.md`.

## Install (cluster login node)

```bash
~/miniconda3_new/envs/mejoint/bin/python -m pip install --no-deps -e .
~/miniconda3_new/envs/mejoint/bin/python -m pip install pydicom
```

## Phase 0 audit of one subject (LSF)

From the repository root, set the paths (quotes matter: they contain spaces), then submit.
The raw NIfTI folder and the exact series are read from the run's `result.json` (`input_echo1`);
set `NIFTI_DIR` only to override that.

```bash
export SUBJECT=<ID> RUN_DIR="<mbme_tedana OUT>/<run>" PIPELINE="<path>/mbme_tedana.py"
bsub < scripts/phase0_one_subject.lsf
```

Writes `outputs/phase0/sub-<ID>/inventory.md` (the table), `inventory.json`, `volume_means.tsv`,
`config_used.yaml`, `done.json` and the environment record (`env_pip_freeze.txt`, `env_threadpool.json`).
`outputs/` is never committed.

## Tests

```bash
~/miniconda3_new/envs/mejoint/bin/python -m pytest -q
```
