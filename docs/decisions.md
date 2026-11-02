# Decisions and gate results

One entry per gate or design decision, newest last. Each entry states what was checked, the evidence, and who decided.

## 2026-11-02: Independent re-check of CLAUDE.md section 5 (before Phase 0)

Script: `scripts/check_section5_claims.py` (numpy only, seed 0).

| Claim | Result |
|---|---|
| 5.2 dR2* SD inflation 2.2-2.6x, corr 0.89-0.92, 3-echo 2.8-3.2x | Reproduced: 2.19-2.57x, 0.89-0.92, 2.76-3.16x |
| Property 1: c0 = -k x OC deviation (linear form, equal sigma) | Reproduced: constant negative ratio across random deviations |
| Property 3: pure dR2* gives c1 = 0, r = 0 | Reproduced (machine precision) |
| Property 4: back-substitution exact for pure, mixed, opposite-sign | Reproduced |
| R01 < 0 after sign convention | Reproduced |
| Property 2 at mu_min/sigma = 3: diag 1.03-1.20, offdiag up to 0.08 | Reproduced: 1.03-1.18, up to 0.087 |
| 5.4 median-based sigma (df = 2) from log-ratio residual | Unbiased within 0.4% from mu4/sigma = 3 upward |

Open issue raised with the user: Phase 2 test (a) asks for +/-5% of I at mu_min/sigma >= 5, but with Rician noise and the log transform the c0 variance is 1.04-1.06 at mu4/sigma = 5 for T2* 45-70 ms (true sigma, 200k samples). The test as written fails at the edge of its own domain. Proposed: +/-5% at mu_min/sigma >= 10 (observed <= 1.7%), and report the 5-10 band separately. Awaiting decision.

## Phase 0: audit

Status: tool built (`mejoint inventory`, tests pass on synthetic folders). Gate pending: single-subject table on real data.

### 2026-11-02: facts supplied by the user

- Environment: `~/miniconda3_new/envs/mejoint` (clone of the tedana env: tedana 26.0.3, plus torch 2.6.0+cu124, dipy, pytest, pyyaml). Always called by full path. `~/miniconda3_new/envs/tedana` stays unchanged for the existing analysis.
- Existing pipeline: `mbme_tedana.py` in `.../CSIMemphis/Analysis/Monitoring Visit/`, outputs in `.../Monitoring Visit/OUT/`.
- Pilot subject for the audit: 41781, run `sub-41781_acq-20260421124817_ser-5`.
- No physiological recordings (respiration, cardiac) exist. Consequence: no RETROICOR/RVT regressors in Phase 2b; respiratory and cardiac effects must be handled by data-driven means (global arm, spectral checks at 0.25-0.5 Hz and aliased cardiac), and reported as a limitation.
- No source of tissue or cavity masks exists. They must be derived (decision deferred to before Phase 5; not needed for Phases 1-4).
- No protocol PDF. CAIPI shift and slice-stack spacing must come from DICOM headers (Siemens ASCCONV) or remain unknown.

### 2026-11-02: facts read from `mbme_tedana.py` (source pasted by the user)

The audit records the sha256 of the cluster copy, so later edits to the script are detectable.

| Step | What the script does |
|---|---|
| Echoes | dcm2niix default names `<base>_<YYYYMMDDhhmmss>_<series>_e{k}.nii(.gz)`, all echoes in one folder; run id `sub-<subject>_acq-<stamp>_ser-<series>` |
| TEs | DICOM headers take priority over JSON sidecars; they must agree within 0.1 ms; written to `<run>/echo_info.json` |
| Volume pairing | Equal counts: used as is. Unequal: data-driven pairing (frame-difference similarity + Viterbi), unverified stretches dropped; indices into the raw echoes in `volumes_used.tsv` |
| Discarded volumes | `--dummy-scans` (default 0) from the start only. No removal of trailing noise frames. First-volume / median signal ratio stored in `result.json` |
| Motion | mcflirt on echo 1 (selected volumes), reference = middle volume; then per volume and per echo one `flirt -applyxfm -interp <interp>` into that reference grid (default `spline`). Single resampling, native space, float32 |
| Intermediates | `--keep-intermediates` keeps `<work>/echo-k_desc-mc_bold.nii.gz`, `echo-1_raw.nii`, mcflirt matrices. `--work-dir` moves `<work>` away from `<run>/work` |
| Mask | nilearn `compute_epi_mask` on motion-corrected echo 1, written to `<run>/brain_mask.nii.gz` and passed to tedana with `--mask` |
| tedana | `--tree tedana_orig --tedpca aic --ica-method robustica --n-robust-runs 30 --seed 42` (script defaults); TEs passed in seconds for tedana >= 26 |
| Outcome | accepted components tagged "Likely BOLD" |
| FD | Power FD, radius 50 mm, unfiltered |

Implications (proposed, to confirm at the phase named):
1. **Phase 5:** the existing counts use robustica (30 runs), not tedana's own default (fastica). Arm A "tedana defaults" is therefore not the existing analysis; the existing analysis corresponds to arm B at `--tedpca aic`. Proposal: add arm A0 = existing pipeline output as is.
2. **Phase 2:** spline resampling can undershoot to <= 0. The log-ratio needs positive values, so a voxel-echo with any frame <= 0 is treated as unusable (enters the df map). Config `frontend.nonpositive_policy`.
3. **Phase 2:** use the pipeline's `brain_mask.nii.gz` as the analysis mask so tedana and mejoint see the same voxels (config `frontend.mask_source`). Check in the Phase 2 maps whether `compute_epi_mask` clips the posterior fossa / skull base.
4. **Phase 2 MP-PCA check on raw echoes:** select raw volumes with `volumes_used.tsv` so raw and motion-corrected series match.
5. **Phase 2b:** FD must be recomputed from `mcflirt.par` with respiratory filtering; the stored FD is unfiltered.
6. **If the audit finds trailing noise frames,** they went through mcflirt and tedana in the existing analysis.
