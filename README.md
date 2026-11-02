# fMRI-Decomp (`mejoint`)

Physics-constrained joint decomposition of multi-echo fMRI into components with a dS0 map and a dR2* map.
The design brief is `CLAUDE.md`; gate results are in `docs/decisions.md`.

## Install (cluster login node)

```bash
~/miniconda3_new/envs/<ENV>/bin/python -m pip install -e ".[dicom,test]"
```

## Phase 0 audit of one subject

```bash
~/miniconda3_new/envs/<ENV>/bin/mejoint inventory \
    --nifti-dir /path/to/subject/nifti \
    --run-dir /path/to/mbme_tedana/output/<run> \
    --subject <ID> --out outputs/phase0/<ID>
```

Writes `inventory.md` (the table), `inventory.json`, `volume_means.tsv`, `config_used.yaml` and `done.json`.

## Tests

```bash
~/miniconda3_new/envs/<ENV>/bin/python -m pytest -q
```
