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

### 2026-11-02: Phase 1 gate PASSED (user)

User: the simulation "seems reasonable"; proceed to real-data tests with what appears reasonable from the source materials and simulations, starting with subject P04. Source amplitudes stay at SD 0.5 s^-1 / 0.5 % at the peak voxel.
User expectation recorded: "very little signal from the fourth echo in most of the brain". Physics (T2* ~45-58 ms leaves ~25-35 % of S0 at 61 ms) suggests usable echo-4 signal in most cortex; the Phase 2 echo-4 SNR and usability maps on real data will settle this.

## Phase 2: front end

### 2026-11-02: implementation choices (where CLAUDE.md section 5 left room)

1. **Log signal demeaned in time.** y = ln S - mean_t ln S (ln of S over its geometric mean) instead of ln(S / arithmetic mean). The two differ by a per-echo constant of about TE^2 var(b)/2, which is not in span{1, TE}: with the arithmetic mean a pure dR2* source left a constant in c1 and r (noise-free max |c1| 6 % of max |c0|); with the geometric mean c1 = r = 0 to float32 precision. Weights still use the arithmetic mean mu.
2. **sigma_init from the multi-echo residual itself**, not from background (unreliable with sum-of-squares multi-coil data) or MP-PCA (kept as the independent cross-check). Trusted voxels: weakest included echo has mu / sigma_ref > 10, with all usable echoes or, failing that, echoes 1-3 (echo 4 can be near the floor brain-wide). sigma_ref is a heavily smoothed map of the echoes-1-3 estimates; trusting voxels by their own noisy estimate biased sigma low by 3 % in a weak-echo-4 simulation (selection on the estimate). Stops with an error if fewer than 50 voxels are trusted.
3. **Smooth sigma map** = normalised Gaussian convolution (FWHM 3 voxels) of the trusted estimates. Used for echo usability (so the cut is not decided by each voxel's noisy estimate) and in voxels with df = 0 or mu_min / sigma_smooth < 10; per-voxel sigma elsewhere (`whiten_with: voxel`).
4. **Residual energy is reported two ways.** Against the per-voxel sigma it is ~1 by construction (sigma comes from the same residual). Against the smooth sigma ("residual excess") a localised violation shows as > 1, because it inflates that voxel's sigma relative to its neighbours. Test (e) uses the excess, plus the static mono-exponential misfit and the frame-wise energy.
5. **Non-positive frames** (spline undershoot) make that voxel-echo unusable (`nonpositive_policy: exclude_echo`).
6. **MP-PCA cross-check** (dipy 1.12.1 `mppca(arr, *, mask, patch_radius, return_sigma)`, verified from source): on raw echoes, 100 volumes, patch radius 2, patch centres on a sparse grid (every 2nd voxel in-plane, every 6th slice; sigma varies smoothly). On simulated data MP-PCA reads 6 % low (ratio 0.94, stable across echoes, sources and g-factor), so the 20 % stop rule is applied to (residual / MP-PCA) / 1.06.
7. **Posterior fossa** has no anatomical mask; summaries use the inferior third of the brain mask as an approximate posterior-fossa / skull-base level, labelled as such.

### 2026-11-02: Phase 2 synthetic evidence (before real data)

- Tests (a)-(e) plus construction tests: 13 front-end tests; full suite 35 tests in ~28 s; all pass under 8 seeds.
- (a) pooled whitened-noise covariance within +/-5 % of I for mu_min/sigma >= 5 (band 5-10: diagonal 1.02-1.03); my earlier concern (up to 6 %) applies to single voxels at exactly SNR 5 with long T2*, not to the pooled test as written.
- (b) pure dR2*: c1 = r = 0 to float32 precision; with noise, corr(c1, b) matches the null.
- (c) c0 = -k x OC deviation exactly (corr -1.000000000) with a mono-exponential mean; <= -0.9999 with empirical means.
- (d) sigma brain-median error 0.3 % with a g-factor map; 99.7 % of voxels within 3 SE.
- (e) residual excess 1.11 at the dynamically dephased skull base vs 1.00 elsewhere; static misfit ~40x; spike frames stand out; including echoes near the floor raises the excess (1.17), snr_min = 5 removes most of it.
- Full-size synthetic run (52 x 64 x 48, 200 volumes, ~50k brain voxels): front end 16 s, sigma median ratio 0.999; MP-PCA 77 s per echo (sparse centres), reads 2 % low, so `mppca_expected_ratio` = 1.02 (the 6 % on the small grid shows the bias depends on geometry; both are small next to the 20 % stop rule).
- Real data next: subject P04, raw echoes (`scripts/phase2_frontend_subject.lsf`).

### 2026-11-02: STOP: sigma cross-check fails on P04 (raw echoes)

First real-data front end (sub-P04_acq-TIMESTAMP_ser-5, raw echoes, 30866 mask voxels, 800 volumes):
- Echo 4 usable in 87 % of voxels (median echo-4 SNR 14.7); inferior third 52 % (median 5.4). T2* median 51 ms (40 ms inferior third).
- **MP-PCA sigma falls with echo time: 348, 263, 217, 187** (echo 4 / echo 1 = 0.54), while the residual-based sigma (which assumes one sigma for all echoes) is 323, matching echo 1 only. Deviation up to 70 %; the 20 % rule of CLAUDE.md 5.4 is exceeded, so work stopped and the user was told.
- Static mono-exponential misfit median 0.036 log units brain-wide (0.051 inferior third), against 0.001 in simulation.
- Frame-wise residual energy drifts (about 1.4 at the start, 1.0 mid-run, rising with spikes at the end); Spearman with FD only 0.08.
- The pipeline brain mask has a large central hole (deep brain excluded, so also excluded from the existing tedana run). nilearn `compute_epi_mask` (read from source, 0.14.1; cluster version not checked) thresholds intensity, keeps the largest component, erodes/dilates, and never fills holes.

Hypotheses and the diagnostics added to decide between them (same command, rerun):
1. per-echo image scaling (would lower noise and signal together; argued against by T2* 51 ms, since a scaling this large would shorten apparent T2* to ~30 ms);
2. MP-PCA counting non-thermal, signal-proportional fluctuations (head drift on raw data) as noise;
3. genuinely echo-dependent thermal noise.
Diagnostics: noise in air outside the head per echo (ratios valid for any coil count; tested: flat with equal sigma, recovers applied per-echo scale factors); MP-PCA repeated on the 100 frames with least head drift; mean signal per echo next to MP-PCA sigma; static decay split into curvature and odd/even-echo alternation; residual split into r0 / r1; residual energy vs head drift from the median position; unmasked echo-1 image with the mask outline; mask-hole count.
In simulation (equal sigma) all of these are flat: MP-PCA 8.08-8.13 across echoes, background SD ratios 1.00, curvature and alternation ~0.

Consequence if the thermal sigma truly differs between echoes: CLAUDE.md 5.3 (equal sigma) no longer holds; whitening needs per-echo sigma_e, c0 is no longer exactly the OC signal, and the sigma-free residual estimator needs replacing. That would be a design change for the user to decide.

### 2026-11-02: pipeline brain mask drops the deep brain (user)

User: P04 has no large shunt or cavity; the central hole is "a regrettable tendency of the tedana workflow that's common across participants" (mbme_tedana builds the mask with nilearn `compute_epi_mask` on motion-corrected echo 1; it thresholds intensity and never fills holes).
Consequence: the existing tedana runs, and any analysis on the pipeline mask, omit deep grey matter, ventricles and periventricular tissue in many subjects.
Added: `frontend.mask_source` / `--mask` / third job-script argument: `pipeline` (default, same voxels as tedana) or `pipeline_filled` (enclosed holes filled in 3D and slice-wise). The summary reports both voxel counts and the hole size. Decision on the analysis mask pending (user).

### 2026-11-02: analysis mask built from the input echoes; independence from tedana (user)

User: "our analysis should always use a filled mask, or ideally should be able to run independently without tedana intermediates or outputs. It should have a way of creating a mask from the input data that is robust."

Implemented (`src/mejoint/mask.py`, config `mask:`; `frontend.mask_source: auto` is now the default):
1. head vs air: echo-1 temporal mean > 4 x the air level (median of the 3-voxel image border, i.e. the noise floor), largest component, holes filled;
2. coil-shading correction: divide by a 40 mm FWHM smoothed copy inside the head;
3. brain vs scalp: corrected intensity > 0.45, one erosion, largest component, regrow;
4. fill every enclosed hole in 3D and slice-wise, one closing.
Tests (`tests/test_mask.py`): on a synthetic head with centre-dark shading (centre 0.25-0.6 x edge), T1-dark ventricles, a skull gap and a scalp, Dice > 0.93-0.95 with the true brain, ventricles always included, < 5 % of the scalp; a whole-image Otsu threshold (the compute_epi_mask approach) loses most of the ventricles on the same phantom. On simulated runs the mask matches the simulated brain (Dice > 0.95).
Not yet validated on real data: the P04 run with the new default will show both outlines (analysis orange, pipeline blue dashed). Real heads have orbits, sinuses, neck and a resection cavity in some patients; the thresholds may need tuning after inspection across subjects.

Independence: the front end runs from the raw echo files alone (`mejoint frontend --echo1 <..._e1.nii.gz>`; TEs from the JSON sidecars; equal echo volume counts required). Still from mbme_tedana: volume pairing (5 of 17 runs) and motion correction (needed from Phase 3/4 on). Open decision: proposed to import mbme_tedana's own pairing and motion-correction functions as a library (a mejoint `preprocess` stage that runs them without tedana), rather than reimplementing them (CLAUDE.md section 4).
