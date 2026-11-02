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
- Pilot subject for the audit: P01, run `sub-P01_acq-TIMESTAMP_ser-5`.
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

### 2026-11-02: LSF facts confirmed on the first Phase 0 job

- `rusage[mem=4000]` is read as MB and enforced as a hard limit (bhist: MEMLIMIT 3.9 G).
- `#BSUB -o` paths are relative to the submission folder; the job's working directory is the submission folder.
- Exported variables did not all reach the job (SUBJECT arrived, RUN_DIR did not). Cause not established: either the multi-line `export` broke when pasted, or LSF dropped values containing spaces. Job scripts therefore take values as arguments; batch jobs (Phase 6) will read paths from a subject list file, not from environment variables.
- Interactive sessions started from Open OnDemand are LSF jobs whose `LS_SUBCWD` is `/var/www/ood/apps/sys/dashboard`. Scripts must not trust `LS_SUBCWD` alone; job scripts locate the repository from their own path, then the current folder, then `LS_SUBCWD`, and check that `src/mejoint` is there.

### 2026-11-02: Phase 0 single-subject audit (run sub-P01_acq-TIMESTAMP_ser-5)

Produced by `scripts/phase0_one_subject.lsf` on the cluster (table shown to the user).

Confirmed: 4 magnitude echoes, TE 16.20 / 31.22 / 46.24 / 61.26 ms (JSON), 800 volumes per echo, matrix 52 x 64 x 48, TR 0.75 s.

Corrections or additions to CLAUDE.md section 4:
- Voxels are 3.281 x 3.281 x 3.3 mm (not ~3.5 mm); slice spacing 3.3 mm (no gap).
- Siemens MAGNETOM Prisma, 3 T, syngo MR XA60, CMRR `cmrr_mbep2d_bold`. DICOMs are enhanced (XA): the pipeline's DICOM TE check failed, so its TEs came from `--echo-times`; JSON agrees.
- MB factor 6, derived from SliceTiming (48 slices, 8 distinct times); `MultibandAccelerationFactor` is absent from the JSON. MB stack separation 8 slices = 26.4 mm. CAIPI shift unknown.
- GRAPPA 2, partial Fourier 0.875, effective echo spacing 0.31 ms, total readout 15.8 ms.
- Phase-encode direction `i` (DICOM ROW): left-right phase encoding.
- Coil: HeadNeck_20, elements HE1-4 + NE1,2, sum-of-squares combination (noise is noncentral-chi at low SNR).

Absent: phase images, physiological recordings, trailing noise frames, non-steady-state volumes (first-volume ratio 1.003).
Consequences: sigma comes only from the multi-echo residual, cross-checked by MP-PCA on raw echoes (a background estimate is unreliable with SoS multi-coil data); no phase-derived dB0(t) regressor; any thermal denoising (Phase 7) is magnitude-only; the thermal null for rank (Phase 3) is the residual only.

Pipeline run: robustica (30 runs), `--tedpca aic`, seed 42, spline; 22 components, 9 accepted, 8 "Likely BOLD". Unfiltered Power FD: mean 0.46 mm, 330 of 800 volumes > 0.5 mm.

Missing: motion-corrected echoes were not kept (`work/` absent). Needed from Phase 2 on. Proposed: rerun this one run with `--keep-intermediates` into a separate output folder (also a reproducibility check of the count of 8).

Gate status: single-subject table shown. Pending: all-runs audit and user review.

### 2026-11-02: Phase 0 all-runs audit (17 runs in mbme_tedana OUT)

Identical in all 17 runs: TE 16.2 / 31.22 / 46.24 / 61.26 ms, matrix 52 x 64 x 48, voxels 3.281 x 3.281 x 3.3 mm, TR 0.75 s, Prisma XA60, CMRR mbep2d_bold, MB 6 (from SliceTiming), GRAPPA 2, partial Fourier 0.875, phase-encode `i` (left-right), sum-of-squares coil combination, tedana 26.0.3 with the same parameters (robustica 30, aic, seed 42, spline).

No run has phase images, physio files, trailing noise frames or non-steady-state volumes: none of the Phase 0 "ask how to proceed" conditions applies.

Points for the user:
- Coil elements differ from "HE1-4;NE1,2" in 4 runs (P12, P06, P09, P08); values to be read from the summary. Fewer neck elements would lower posterior-fossa SNR and change the g-factor there.
- Unequal echo volume counts in 5 of 17 runs (not only P08): P12, P14, P10, P05, P08. Pairing kept 797, 751, 777 and 797 volumes; P08 (271/278/267/277) failed in the pipeline. Where pairing drops volumes inside a run, the kept volumes are not consecutive in time; filtering, FD and phase-randomised nulls must use the acquisition timeline from `volumes_used.tsv`. The summary now reports these gaps.
- 5 of the 16 completed runs have 0 "Likely BOLD" components (P12, P04, P10, P03, P07). The summary now flags tedana's QC fallback (no provisional accept), which mbme_tedana records.
- No run kept motion-corrected echoes.

Gate status: summary shown to the user; awaiting review and approval.

### 2026-11-02: Phase 0 gate PASSED (user)

User decisions:
- Phase 0 passes. No phase, physio, noise frames or non-steady-state volumes in any run, so sections 5.4 and 9 stand as written (sigma from the residual with an MP-PCA cross-check; magnitude-only denoising if ever used).
- Subject P08 is excluded (pairing failed; ~270 volumes). Recorded in `configs/default.yaml` under `subjects.exclude`.
- All work stays on the single branch `claude/beautiful-hamilton-141ubf` (replaces one branch per phase in CLAUDE.md section 2).
- CLAUDE.md sections 2-4 updated with the confirmed facts.
- Phase encoding: the user was unsure whether left-right was intended. The data are unambiguous (JSON `i`, DICOM `ROW`, 52-line axis = narrower 170 mm FOV), so analysis proceeds with left-right phase encoding as a fact; Phase 2 maps will show distortion and ghosting along that axis.

Open items carried forward:
- Motion-corrected echoes for P01 (rerun with `--keep-intermediates` into `OUT_keep`) before Phase 2.
- 5 runs with tedana's "no provisional accept" fallback: their tedana counts are unreliable; relevant for the Phase 5 and 11 count comparisons.

## Phase 1: simulator and test harness

### 2026-11-02: Phase 1 gate evidence

`src/mejoint/simulate.py` builds 4-echo magnitude data at the real TEs (16.20 / 31.22 / 46.24 / 61.26 ms, TR 0.75 s) from known maps and time courses. Signal: ln S = ln S0 + a(t) - TE (R2* + b(t)) + ln D(t) + spikes, so without dephasing and spikes ln(S / baseline) is exactly a - TE b (CLAUDE.md 5.1). Contents, as section 8 requires:
- focal (Gaussian blob) and diffuse (rectified smooth field) sources: dR2*-only, dS0-only, mixed with the same and with opposite signs;
- T2* spanning 20-70 ms, shorter (about 22 ms) at a synthetic skull base;
- complex Gaussian noise then magnitude: Rician (1 coil) or sum-of-squares noncentral chi (L coils, equal sensitivities);
- optional smooth g-factor-like noise map (1 to 1.5 across the brain);
- smooth global respiratory dR2* (about 0.3 Hz, drifting frequency and amplitude);
- motion-like spikes that violate the model (an S0-like jump plus an independent jump per echo);
- through-slice dephasing |sinc(df TE)| at the skull base (static curvature), optionally modulated by respiration (dynamic curvature).

Choices made (all in `configs/default.yaml`, `simulate:`):
- Amplitudes are the SD of each source at its peak voxel (unit-SD time courses). Defaults 0.5 s^-1 and 0.5 %, so ~95 % of excursions stay within +/-1 s^-1 and +/-1 %. First defaults (SD 1 s^-1, 1 %) gave +/-15 % swings at TE 61 ms in the QC figure, too large for rest.
- Noise is drawn last, so a noise-free run with the same seed has identical maps and time courses. Changing `chunk_voxels` changes the noise realisation.
- Seeds: SeedSequence(project_seed, spawn_key=(1,)) by default; entropy and spawn key are stored in the output.

Tests (`tests/test_simulate.py`, 13): seeded reproducibility; noise-free log-ratio equals a - TE b to 1e-5; T2* range and skull base; source maps, unit-SD time courses, band limits, respiration peak 0.25-0.5 Hz; high-SNR magnitude SD = sigma and background mean = central-chi mean for 1 and 4 coils; Rician floor bias matches the analytic mean (1.549 sigma at mu = sigma); g-factor noise follows its map; spikes leave energy outside span{1, TE} only at the injected frames and voxels; skull-base curvature appears only where injected (static and dynamic); SNR reported and ordered; NIfTI round trip read correctly by the Phase 0 audit; invalid settings refused. Every test states the SNR of its data.

Robustness: all 22 tests pass under 8 different seeds (MEJOINT_TEST_SEED=0..7). Two tests were redesigned during development because they depended on seed luck, not on the simulator: the Rician-floor test now sets T2* 20-25 ms and SNR0 25 itself, and the g-factor range is now scaled within the brain (it was diluted by the background).

Gate: full suite 22 tests in 7.5 s on 4 CPUs (limit 2 min). QC figure: `docs/figures/phase1_simulation.png` (`scripts/plot_simulation.py`).

Not simulated yet (added when a later phase needs them): rigid-body motion with resampling (interpolation-correlated noise), MB slice leakage, cardiac aliasing, gaps in the acquisition timeline.

Gate status: evidence shown to the user; awaiting review.
