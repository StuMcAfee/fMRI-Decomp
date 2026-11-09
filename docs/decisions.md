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

### 2026-11-03: P04 results with the diagnostics; per-echo noise estimator; mask v2

Reproducibility: the `--keep-intermediates` rerun of P04 gave "0 likely-BOLD components (of 19 total; 13 accepted)", the same count and component number as the original run.

Raw-echo front end with the first automatic mask (summary from the user):
- MP-PCA per-echo sigma is the same on the 100 quietest frames (349, 261, 211, 179) as on frames 0-99 (348, 258, 209, 179): head drift does not cause its echo dependence.
- MP-PCA sigma fits sigma_e^2 = 128^2 + (0.032 x mean signal_e)^2 within 4 % on all four echoes: a fixed floor plus a part proportional to signal (high-rank within 5 x 5 x 5 patches, so MP-PCA counts it as noise). Whether that part is thermal is not decidable from MP-PCA alone.
- Air outside the head: temporal SD ratios 1, 0.57, 0.48, 0.47 (a drop from echo 1 to 2, flat after). No single noise model fits, and the air SD (~13-29) is ~25x below the in-brain levels, so this reconstruction's background does not represent in-brain noise. Inconclusive.
- No per-echo image scaling: median ln(mean / mono-exponential fit) per echo is within 0.017 of 0.
- Frame-wise residual energy tracks head drift from the median position (Spearman 0.92; drift up to 1.6 mm): on raw echoes the residual carries motion-driven model violations, which also inflates the residual-based sigma (337).
- The first automatic mask was too large (1687 ml; head 2664 ml): it took in neck and face below the brain (inferior third: echo 4 usable in 11 % of voxels, T2* p10 14 ms).

Added (2026-11-03):
1. **Per-echo noise ratios from the residual** (`frontend.per_echo_noise`): the residual's 2 x 2 covariance per voxel is a known mixture of the four per-echo variances (weights depend on T2*); pooling voxels solves for the ratios. BOLD and S0 signal never enter the residual, so physiology cannot inflate it (motion can). Simulation at 800 volumes and ~18k voxels: equal noise gives 1.00 with seed-to-seed SD 1.3-1.7 %; [1, 0.6, 0.5, 0.45] and [1, 0.74, 0.60, 0.51] are recovered within ~0.02, with a g-factor map and all sources present. A smoothed-scale variant was worse (biased) and was dropped. This is the in-brain test of CLAUDE.md 5.3's equal-sigma assumption; run it on the motion-corrected echoes (ratios survive resampling, CLAUDE.md 5.4).
2. **Mask v2**: brain step also requires a brain-like decay, mean(echo 4) / mean(echo 1) > exp(-(TE4 - TE1) / 25 ms), which coil shading cannot affect; neck and face muscle decays faster than brain or CSF. One voxel of regrowth into bright head tissue afterwards. Phantom with a neck joined only through the cord: intensity-only mask includes 100 % of the neck muscle, mask v2 1.4 %; ventricles 100 %, brain Dice 0.998. Trade-off: brain with T2* < 25 ms at the very edge (next to air/bone) can be trimmed; `mask.t2s_min_ms` sets it.
3. **Mask QC across runs** (`mejoint mask-qc`, `scripts/mask_qc_all.lsf`): echoes 1 and 4 only, a table of volumes and Dice with the pipeline mask, and one contact sheet with both outlines for every run.

Phase 2 remains stopped on the sigma question.

### 2026-11-03: P04 motion-corrected echoes (front end before the per-echo estimator)

- Residual energy still tracks head drift after motion correction (Spearman 0.82; raw 0.92). Not only misregistration: when the head moves, the B0 inhomogeneity near air/bone moves relative to the brain, changing through-slice dephasing and so the curvature of the decay (seen only by the residual). Added: per-echo noise ratios are also computed on the half of frames nearest the median head position (`low_drift`), to compare with all frames.
- Residual sigma median 312 (raw 324): single spline resampling lowers the noise by ~4 %.
- Voxel-echoes with frames <= 0 rose from [75, 194, 131, 62] (raw) to [164, 565, 639, 705] (spline undershoot); `nonpositive_policy: exclude_echo` handles them.
- Air-noise ratios changed after resampling (SD 1, 0.65, 0.56, 0.50 vs raw 1, 0.57, 0.48, 0.47): the background is not a usable thermal reference in this data.
- Mask (v1) 1715 ml: same over-inclusion as raw; mask v2 pending real-data check.
- Test note: on the small simulated head (T2* 20-70 ms, skull base 22 ms) mask v2 keeps ~90 % of the simulated brain; the T2* > 25 ms limit trims short-T2* edge brain. Real-data contact sheets decide the limit.

### 2026-11-03: per-echo noise on P04 (motion-corrected); mask v3; P02 excluded

**Per-echo noise from the residual (P04, mc echoes, 20000 voxels):** all frames [1, 0.86, 0.61, 0.77]; voxel halves [1, 0.77, 0.58, 0.68] / [1, 0.96, 0.64, 0.86]; frames nearest the median head position [1, 0.84, 0.63, 0.70]. Echo 1 is the noisiest by all three methods (MP-PCA, air, residual), so CLAUDE.md 5.3's equal-sigma assumption is not supported in this data. The size is uncertain: the halves differ by up to 0.2 (~0.05 in simulation) and echo 4 > echo 3 is physically odd. Likely cause: the real residual is not pure noise (it tracks head drift after motion correction; the mean decay is slightly concave brain-wide, median ln(mu/fit) per echo [-0.004, 0.011, 0.005, -0.017], as through-slice dephasing produces). Decision on the noise model pending (user).

**Mask v2 across 16 runs (mask QC):** volumes 1043-1600 ml (median 1292; pipeline median 1335). User: the outline variably excludes the edges of the cerebellum and orbitofrontal cortex, "important regions". Cause: the 25 ms decay limit also removes brain whose late echoes are lost to dephasing. Against the pipeline mask v2 lost a median 88 ml (max 165) at the edges and gained 64 ml (max 205) in the deep brain.

**Mask v3:** the decay limit now only decides what is connected to the brain; the core then regrows up to 4 voxels (~13 mm) into adjacent tissue bright at echo 1, whatever its decay. Phantom with a dephased lateral-inferior rim (T2* 15 ms) and an anatomical neck: rim kept 33 % (v2) -> 96 % (v3); neck muscle 1.4 % -> 8.9 % (a band around the cord); scalp 2 %; ventricles 100 %. 3 voxels: rim 83 %, muscle 6 %; 5 voxels: rim 100 %, muscle 12 %. Tissue with almost no signal even at echo 1 (severe dropout) stays outside: an EPI-only mask cannot recover it; an anatomical (T1-based) mask could.

**P02 excluded** (user): metal artefact or similar in the right cerebral hemisphere (also visible as an abnormal mask, 844 ml). 15 runs remain.

### 2026-11-03: stay on the gated route; mask frozen; noise survey next (user)

User: continue on the gated route; avoid getting sidetracked perfecting the mask.
- Mask v3 is frozen as the Phase 2 analysis mask. Any refinement (e.g. a T1-based mask) is deferred to Phase 5, which needs tissue masks anyway.
- Remaining Phase 2 blocker: the noise model. Next step: front end on the raw echoes of all 15 runs (`scripts/noise_survey_all.lsf`, no MP-PCA) and a table of per-echo noise ratios (`noise_survey.md`). Consistent ratios across runs would make the echo dependence a protocol/reconstruction property that can be set once.
- A `mejoint preprocess` step (mbme_tedana's pairing and motion correction without tedana) is deferred to Phase 6; Phases 3-5 use P04, which already has motion-corrected echoes.
- Mask v3 contact sheet (15 runs, 2026-11-03): cerebellar hemispheres and orbitofrontal cortex inside the mask in every run; deep brain included. Volumes 1210-1789 ml (pipeline 963-1675). Small lateral islands in the lowest slice in a few runs (P17, P11, P07 at z = 3): inferior temporal/cerebellar tissue or some skull-base non-brain tissue; accepted for Phase 2, revisit with tissue masks at Phase 5.

### 2026-11-03: traveling waves on hold (user)

User added `docs/WAVES_BRIEF.md` and `docs/mockups/waves/`, and asked to hold all wave work until the decomposition of real data is reached. Nothing wave-related is built or branched before then.

### 2026-11-03: overflow warnings in the mono-exponential fit (noise survey log)

Cause: voxels with only one usable echo (< 1 % of the mask in P04). Their Gauss-Newton system is singular; rounding left a tiny positive determinant and a runaway step. Their T2*/S0 were already NaN, and every voxel is fitted independently, so other voxels were unaffected (test: exact T2* in all well-posed voxels). The channels, sigma and per-echo noise ratios never use this fit, so the noise survey is unaffected.
Fix: fit only voxels with >= 2 usable echoes, skip ill-conditioned steps, bound the steps and ln S0. Regression test with one-echo, rising, flat, very fast and non-monotonic decays runs with warnings treated as errors.
Correction: T2* values near the 500 ms cap (e.g. a 90th percentile of 219 ms in the inferior third) come from rising or flat decays (CSF, inflow), not from this bug.

### 2026-11-03: noise survey (15 runs) and per-echo weighting adopted (user: option A)

Survey (raw echoes, residual-based per-echo ratios, frames nearest the median head position): echo 1 noisiest in 14 of 15 runs; median [1, 0.761, 0.671, 0.602], interquartile range [1, 0.72-0.82, 0.61-0.74, 0.52-0.66]; the 6 runs whose voxel halves agree within 0.15 give the same median ([1, 0.76, 0.63, 0.61]); MP-PCA on P04 gave [1, 0.74, 0.60, 0.51]. Per-run estimates are unreliable (halves disagree by > 0.15 in 9 of 15 runs; residual tracks head drift), so one study-wide vector is used. Cause not established (unverified candidates: echo-1-specific reconstruction, a signal-proportional fluctuation largest at echo 1).

Decision (user, 2026-11-03): **option A**. `frontend.weighting: per_echo`, `frontend.echo_noise_ratios: [1.0, 0.761, 0.671, 0.602]`. Implementation: the noise-scaled mean mu_e / r_e replaces mu_e wherever the equal-sigma construction used it as the noise scale (sigma-free residual, trusted-voxel test, echo SNR and usability, whitening); the mono-exponential fit weights residuals by 1 / r_e; the MP-PCA check compares per echo with sigma_v r_e. `weighting: equal` is kept for the Phase 4 tedana-equivalence test and as a sensitivity analysis of every main result. CLAUDE.md 5.3 and 5.4 updated.

Evidence (simulation with these ratios, 28 x 28 x 16, 400 volumes): whitened channel/residual covariance within 0.9 % of I with per-echo weights; up to 65 % off with equal weights. Tests now simulate the configured ratios by default; properties defined under equal noise (test (c), the sigma bootstrap) run with equal data and equal weights. The floor case of test (e) was recalibrated from a six-seed spread (excess 1.080-1.105 included vs 1.023-1.044 excluded). 53 tests pass under six seeds.

Phase 2 gate still needs: P04 rerun with per-echo weights (raw with the per-echo MP-PCA check; mc for the maps).

### 2026-11-03: Phase 2 gate run on P04 with per-echo weights: MP-PCA check fails a second time (STOP)

Motion-corrected run: mask 1424 ml (40077 voxels); echo 4 usable in 79 % of voxels (36 % of the inferior third); residual excess 0.97 (0.94 inferior); sigma_v (echo-1 noise) median 439. This run's own per-echo estimate [1, 0.69, 0.54, 0.53] is lower than the study-wide ratios and its voxel halves agree within ~0.1; it is close to MP-PCA's ratios for P04 ([1, 0.74, 0.60, 0.51]).
Raw run: mask 1416 ml; residual excess 0.97; sigma_v 455; per-echo estimate unstable again ([1, 0.91, 0.59, 0.84], halves up to 0.15 apart; residual vs drift Spearman 0.92).
MP-PCA check (raw), residual model sigma_v r_e / MP-PCA sigma_e: 1.33, 1.33, 1.45, 1.53 (quiet window the same). The shape across echoes now nearly matches (under equal weights: 0.93-1.73); the overall level is ~33 % above MP-PCA and exceeds the 20 % rule. Second failure of this check: stopped and asked the user (CLAUDE.md section 2).
Candidate explanations: (1) slow model violations (drift-related changes in a non-mono-exponential decay) inflate the residual-based sigma; (2) MP-PCA reads low because MB-6/GRAPPA noise is spatially correlated (unverified size). Proposed test, no method change: sigma and per-echo ratios also from frame-to-frame differences of the residual (white thermal noise survives differencing, slow violations do not).

### 2026-11-03: differenced-residual diagnostic (user: option A)

Added, as a diagnostic only (whitening unchanged): sigma and per-echo ratios from frame-to-frame differences of the sigma-free residual (`sigma_from_residual(..., difference=True)`, `per_echo_noise(..., difference=True)`). White thermal noise has variance 2 sigma^2 after differencing; slow (drift-paced) model violations largely cancel. Fast violations (e.g. respiration-paced) do not, and a spike now touches two differences (the median stays robust).
Simulator: optional `slow_violation` (drift-paced curvature change outside span{1, TE}).
Validation (30 x 32 x 18, 600 volumes, per-echo noise): white noise only: residual sigma / true 0.999, differenced 1.000; with the slow violation (amplitude 0.03): residual 2.06, differenced 1.001; differenced per-echo ratios correct within 0.06 in both.
Outputs: `sigma_diff.nii.gz`; summary lines "Differenced-residual / residual sigma" and differenced per-echo ratios; MP-PCA table gains "Differenced residual sigma" and "Differenced / MP-PCA" per echo.
Decision rule agreed in advance: differenced / MP-PCA near 1 (within 20 %) -> the residual sigma was inflated by slow violations (propose the differenced estimator, CLAUDE.md 5.4 step 4 change, to the user); still ~1.3 -> MP-PCA likely biased low for this reconstruction (revisit the 20 % rule).
Tests: 55 pass under three seeds; the two heaviest simulation tests are marked slow (`-m "not slow"` for a quick check).

### 2026-11-03: differenced diagnostic on P04: mixed result against the agreed rule (STOP, proposal to the user)

Motion-corrected: differenced / residual sigma p10/50/90 0.80 / 0.93 / 0.99; differenced per-echo ratios [1, 0.720, 0.606, 0.546] (halves within 0.012).
Raw: differenced / residual sigma 0.77 / 0.92 / 0.99; differenced ratios [1, 0.732, 0.609, 0.560] (halves within 0.01). Non-differenced ratios in the same run: [1, 0.91, 0.59, 0.84] (halves up to 0.13 apart); low-drift frames [1, 0.85, 0.63, 0.71] (halves up to 0.36 apart).
MP-PCA check (raw), model sigma / MP-PCA sigma per echo: residual 1.33, 1.33, 1.45, 1.53; differenced 1.12, 1.13, 1.22, 1.30. MP-PCA's own ratios: [1, 0.757, 0.616, 0.522].

Reading:
1. Slow violations inflate the residual sigma by about 8 % (brain median), more in some voxels (p10 0.77). Echo 1, which no ratio affects, moves from 1.33 to 1.12.
2. The differenced ratios are stable (halves within 0.01), agree raw vs mc, and are close to MP-PCA's shape. The non-differenced ratios that set the study-wide vector are not stable. The study-wide vector [1, 0.761, 0.671, 0.602] is therefore likely too high at echoes 3-4 (it came from drift-contaminated residuals).
3. With this run's differenced ratios instead of the study-wide ones, differenced / MP-PCA would be about 1.12, 1.09, 1.11, 1.20 (approximate: rescaled from the table, not rerun; sigma_v also depends weakly on the ratios through the residual basis). A uniform ~10 % excess over MP-PCA remains; candidates are a modest MP-PCA low bias on MB-6/GRAPPA data or fast (e.g. respiration-paced) violations. Neither is verified.

Against the rule: echoes 1-2 within 20 %, echoes 3-4 not (1.22, 1.30), with the study-wide ratios. Gate still stopped.

Proposal (needs user approval; changes CLAUDE.md 5.4 step 4):
(a) sigma from frame-to-frame differences of the residual: sigma^2 = median_t(sum_k (Delta u)^2) / (2 median chi2_df). Config `frontend.sigma_estimator: differenced` (implemented, default still `residual`; simulation: differenced sigma used for whitening is within 3 % of true with and without the slow violation).
(b) Re-derive the study-wide ratios as the median of the differenced ratios over the 15 runs (noise survey rerun; the survey now reports them).
(c) Rerun P04 (raw with MP-PCA, mc for maps). Rule fixed in advance: all four echoes within 20 % -> the MP-PCA check passes. Otherwise stop: no further tuning of the noise model to pass this check; the user decides whether the remaining gap is an accepted limitation.
Infrastructure added (no change to the analysis defaults): `frontend-all --force` / `noise_survey_all.lsf <OUT> force`; survey columns for differenced ratios, their halves and differenced / residual sigma; `sigma_plain.nii.gz` map.

### 2026-11-03: differenced sigma estimator and new study-wide ratios adopted (user approved (a)-(c))

Noise survey rerun (15 runs, raw echoes, all with `force`): differenced per-echo ratios median **[1, 0.721, 0.631, 0.560]**, interquartile range [1, 0.678, 0.608, 0.529] to [1, 0.768, 0.669, 0.584]. Voxel halves agree within 0.03 in 10 of 15 runs (P01, P06, P11, P05 and P07 are 0.05-0.17 apart). Non-differenced estimates in the same rerun remain unstable (e.g. P06 [1, 1.37, 1.00, 1.15], P09 [1, 1.38, 0.66, 1.04]). Differenced / residual sigma brain median 0.87-0.98 across runs (P09 lowest). The low-drift median moved from [1, 0.761, 0.671, 0.602] (first survey) to [1, 0.721, 0.617, 0.558] (this rerun); the voxel selection changed with the per-echo weights, another sign that the non-differenced estimate is sensitive to which voxels are used.

Decision (user, 2026-11-03):
(a) `frontend.sigma_estimator: differenced` (CLAUDE.md 5.4 step 4 updated); `residual` kept as a sensitivity analysis.
(b) `frontend.echo_noise_ratios: [1.0, 0.721, 0.631, 0.560]` (CLAUDE.md 5.3 updated).
(c) Rerun P04 (raw with MP-PCA; mc for the maps). Rule fixed in advance: all four echoes within 20 % of MP-PCA (after the simulated 2 % bias) -> the MP-PCA check passes; otherwise stop, no further tuning of the noise model; the user decides whether the remaining gap is an accepted limitation.

Standard error of the differenced estimator: 1.103x (df 1) and 1.124x (df 2) the non-differenced one (simulation, white noise; unbiased); `sigma_rel_se(..., difference=True)` and test (d) use it. 57 tests pass under seeds 0, 1 and 2.

### 2026-11-03: MP-PCA check split into level and shape (user decision, before the P04 rerun results)

The single rule (every echo's model / MP-PCA sigma within 20 %) mixed two things with different consequences:
- **Shape**, the echo ratios. Errors change the echo weighting. Worst case over sign patterns and T2* 20-70 ms: 5 % -> channel variances off by 10 %, c0-c1 noise correlation 0.05; 10 % -> 21 %, 0.11; 20 % -> 44 %, 0.22.
- **Level**, the overall scale. It only rescales the channels and leaves the separation unchanged. It affects echo usability (snr_min), the noise-debiasing term of eta, and the residual-based thermal rank null. The gated rank estimate and the eta significance test use data-driven nulls.

The 20 % figure was a planning-session choice, not derived.

New rule:
- **Shape**: each echo's ratio / echo 1's within 1 +/- 0.10 (`frontend.mppca_shape_tolerance`).
- **Level**: echo-1 ratio after the simulated bias within 1 +/- 0.20 (`frontend.mppca_level_tolerance`; unchanged).

The summary also reports the channel noise covariance implied if MP-PCA's echo ratios were the truth.

This is stricter than before (the level limit is unchanged; shape is tightened). It was set before the rerun results, but after the approximate estimate (level ~1.10, shape within ~0.07). The reason for the change is the consequence analysis above, not the expected outcome. MP-PCA is not ground truth: a fail means the two estimators disagree, not which one is wrong.

`mejoint mppca-recheck <summary.json>` re-evaluates outputs made before this rule. 62 tests pass under seeds 0 and 1.

### 2026-11-03: Phase 2 gate run on P04 (raw, differenced estimator, ratios [1, 0.721, 0.631, 0.560]): shape passes, level misses by 0.004 (STOP, user decides)

MP-PCA check, level/shape rule:
- **Shape:** [1, 0.952, 1.024, 1.072], within the 10 % limit. If MP-PCA's echo ratios were the truth, channel noise variances would be off by up to 4.2 % and the c0-c1 noise correlation would be +0.057.
- **Level:** 1.204 after the simulated bias, against a limit of 1.20. Raw ratio 1.228; quiet window 1.218. Per-voxel differenced sigma alone (without the smooth-map fill in 30 % of voxels) gives 1.196 raw, 1.173 after the bias. The rule uses the sigma actually used for whitening, so the verdict is STOP.

Why the level rose from the predicted ~1.12: sigma_v (echo-1 noise) depends on the echo ratios through the residual basis. Lowering r for echoes 2-4 raised the differenced sigma_v from 397 to 422 (+6 %). The earlier rescaling had assumed this coupling was weak; it is not.

This run's own differenced ratios [1, 0.719, 0.603, 0.546] (halves within 0.02) agree with the study-wide vector within 0.03.

Non-differenced residual energy against the differenced sigma is now 1.18. This is expected: 1 / 0.92^2, the slow violations. It no longer means a model failure.

As agreed in advance, no further tuning of the noise model. The user decides whether the level gap is an accepted limitation.

### 2026-11-03: level gap accepted as a documented limitation (user: option A); Phase 2 gate evidence

Decision (user): the MP-PCA level gap (1.204 against a limit of 1.20; shape passes) is accepted as a limitation and is not treated as a model failure. It is reported as "thermal sigma level uncertain within about 20 % between two independent estimators (residual, MP-PCA)". The `snr_min` = 3 and `sigma_estimator: residual` sensitivity runs bracket its effect. Later, a spatial-noise-correlation measurement and an MP-PCA simulation will be added as a reported diagnostic only, not as a gate. They test whether MP-PCA reads low on correlated MB-6/GRAPPA noise and will inform which sigma level to trust for eta's noise correction.

Gate evidence, P04 (raw and mc, differenced estimator, ratios [1, 0.721, 0.631, 0.560]):
- **Tests:** 62 synthetic tests pass, including (a)-(e).
- **Maps** (mu, T2*, sigma, usable echoes / df, residual energy, inferior slices z = 3, 7, 13):
  - The analysis mask keeps the cerebellum and orbitofrontal cortex.
  - Echo 4 is usable in 81 % of the brain and 40 % of the inferior third.
  - Fewer usable echoes at the skull base and orbitofrontal sinus interface (dephasing).
  - Long T2* in CSF / 4th ventricle.
  - The sigma map is smooth, with a central rise that MP-PCA reproduces spatially. The two disagree in level, not in pattern.
- **Residual energy against the differenced sigma:** 1.18 (raw) and 1.15 (mc), highest at tissue edges. Frame-wise residual energy follows head drift (Spearman 0.94 raw, 0.86 mc; a step at the mid-run position jump), not FD (0.06, 0.09). Interpretation (unverified): position-dependent changes of a non-mono-exponential decay (e.g. B0 / through-slice dephasing changing with head position). Phase 2b motion regressors should absorb the slow part.
- **mc vs raw:** sigma 418 vs 428 (spline interpolation lowers it about 2 %). Differenced ratios agree ([1, 0.718, 0.603, 0.541] vs [1, 0.719, 0.603, 0.546]).
- **mc non-positive frames** (spline undershoot): [45, 182, 404, 667] voxels per echo. Under `nonpositive_policy: exclude_echo` these voxel-echoes are unusable (echo 4: 1.7 % of the mask). This policy still needs the user's confirmation.

Label fix: "Residual energy vs smooth sigma" now reads "1 = white thermal noise, > 1 = slow violations or model misfit". With the differenced sigma, slow violations show up here rather than inflating sigma.

### 2026-11-03: Phase 2 gate passed (user); nonpositive_policy exclude_echo confirmed; Phase 2b built

User: Phase 2 gate passes; `nonpositive_policy: exclude_echo` confirmed (a voxel-echo with any frame <= 0 is unusable).

Phase 2b, nuisance projection (`nuisance.py`, config `nuisance:`). Choices (defaults; all in the config):
- **Regressors:** the 24-parameter expansion of the mcflirt parameters (6 parameters, their frame-to-frame changes, and the squares of both, demeaned), plus one constant per contiguous segment. Squares are included because the Phase 2 residual tracked distance from the median head position, which is not linear in the parameters. No physiological or phase regressors exist (Phase 0).
- **Respiratory filter:** each run's breathing peak is found in the motion-parameter spectrum within 0.15-0.60 Hz, measured against a robust power-law background fit; drift alone scores 1.3-1.6 in simulation. If its prominence is >= 3, the parameters are band-stopped at the peak +/- 0.06 Hz (Butterworth order 4, zero phase). Otherwise they are left unfiltered and the reason is reported. Filtered parameters are used for the regressors and for FD (`fd_filtered_mm` in frames.tsv); unfiltered FD is kept alongside. `motion_filter: lowpass` (0.10 Hz) and `none` are available.
- **Gaps** (e.g. P10): segments come from volumes_used.tsv; filtering, frame-to-frame changes and FD never cross a gap; each segment gets its own constant.
- **Where it is applied:** to the channels c and the residual r. This is equivalent to applying it to the log-ratio echoes, because identical regressors commute with the per-voxel echo mixing (test: equal to float32 precision).
- **df correction (CLAUDE.md: "reduce df by p + 1"):** sigma is divided by sqrt of the noise variance the projector keeps: (T - rank) / T for the residual estimator, tr(D P D^T) / (2 (T - 1)) for the differenced one (both exact, Monte Carlo checked). Frame-wise residual energy is corrected per frame by 1 - leverage, so frames with unusual motion are not under-read. The rank is recorded in summary.json and the regressors in `nuisance_regressors.tsv`, for the Phase 3 rank null.
- **No-regression variant:** `frontend --no-nuisance` (job script 4th argument `noreg`, output folder `_noreg`). Every regression run also reports its own residual energy and drift / FD correlations without regression.

Tests (tests/test_nuisance.py; 69 tests in total pass under seeds 0-2, each run twice):
- gaps and FD;
- breathing peak found within one frequency bin and > 98 % of its power removed, slow motion kept within 5 %;
- no band-stop without a peak;
- the projector removes its regressors exactly, is idempotent, and its df factors and per-frame leverage match Monte Carlo;
- commutation of projecting echoes vs channels;
- simulation with a motion-paced violation: non-differenced sigma 2x true without regression, within 3 % with it; differenced sigma within 3 %; residual covariance = kept variance x I within 0.05; c1 noise-only for pure dR2*;
- end-to-end run with and without regression.

### 2026-11-03: Phase 2b on P04 (mc, 24 regressors): slow violations removed; projector removes half of c0's energy (to review)

- **Breathing peak:** 0.26 Hz (about 16 breaths/min), prominence 15.5; band-stopped at 0.20-0.32 Hz. Filtered FD mean 0.24 -> 0.13 mm, 95th percentile 0.51 -> 0.29 mm.
- **Slow violations removed:**
  - differenced / non-differenced sigma brain median 0.927 -> 0.978;
  - residual energy vs smooth sigma 1.154 -> 1.037 (r0 1.16 -> 1.04, r1 1.11 -> 1.02);
  - frame-wise residual energy median 1.33 -> 1.11.
- **Correlations of frame-wise residual energy:**
  - with head drift: 0.86 -> -0.40. The sign flip is probably confounded with time (high drift early with little fast motion; more fast motion later). Unverified.
  - with unfiltered FD: 0.13; with filtered FD: 0.02. The remaining frame-wise residual tracks breathing pseudo-motion (the TE-nonlinear part of respiratory B0, CLAUDE.md section 10), which is deliberately not regressed.
- **Non-differenced per-echo ratios** are now stable ([1, 0.711, 0.585, 0.489], halves within 0.05; before: halves up to 0.36 apart) and close to the differenced ones ([1, 0.716, 0.608, 0.540]). This supports the reading that the earlier instability came from motion-related violations.
- **sigma:** 414.6 vs 418.2 without regression.
- **Energy kept** (thermal noise alone would keep 0.969): c0 0.50, c1 0.71, r0 0.78, r1 0.84. The 24 motion regressors remove about half of c0's energy. Part of that is chance overlap: the slow regressors (parameters and squares) share the low-frequency band where resting-state BOLD also lives. The size of that overlap is not yet known.

### 2026-11-03: surrogate-regressor check added (user: go fast); Phase 3 rank built, gate NOT passed

Surrogate check (`nuisance.n_surrogates: 5`): projectors from phase-randomized motion parameters (same spectrum and cross-spectra, scrambled timing), built like the real projector. The summary reports energy kept with them next to the real one. Test: on pure noise the chance removal equals the df share.

Phase 3 (`rank.py`, `mejoint rank`, `scripts/phase3_overnight.lsf`):
- **Spectrum:** stacked [c0 | c1] (undefined c1 rows excluded), T x T Gram in float64.
- **Nulls:** thermal (residual rows, aspect matched); Marchenko-Pastur edge; MAPCA AIC / KIC / MDL (mapca package, normalize=True as in tedana; c0 and c1 maps side by side with a gap); phase-randomized surrogates (mirror-extended, so the slowest half-cycle mode of non-periodic data is represented).

Gate test: K = 30 planted components at 2-20x the noise edge.
- White noise: deflated null recovers 30 exactly.
- Coloured noise (AR(1), rho 0-0.6 per row):
  - index-by-index parallel analysis under-counts (22-26). The null inherits the components' power when rows share a spectrum.
  - deflated (sequential) parallel analysis over-counts (runs to k_max). Its null's top eigenvalue sits ~2 % below the data's first noise eigenvalue once the strong components are removed.
  - Mirror extension fixes the randomization itself (pure coloured noise: null within ~1 %). The deflation step is the open problem.
- The coloured-noise, pure-coloured-noise and projected-data cases are marked as known failures (strict xfail) until fixed. On pure coloured noise the deflated search also runs to k_max, so on real (coloured) data expect k deflated = k_max; the overnight rank step will be slow for that reason.

**Gate not passed.** The overnight run on P04 is exploratory: front end with the surrogate check, then all rank estimates and the real-data spectra. The spectra are required for the gate anyway and will show which estimator to fix.

Candidate fixes for the next session (not yet tested): AR(p) surrogates per row instead of phase randomization; deflation with the null's own top directions matched; or a noise reference from the residual channels after rescaling by their measured coloured spectrum.

### 2026-11-04: exploratory rank on P04 (mc, 24 regressors)

78569 rows x 800 frames; effective 774 frames. The full spectrum drops to ~1e-15 after index 774, so the 25 projected dimensions plus the mean are removed exactly.

| Estimate | k |
|---|---|
| deflated | 168 |
| index by index | 17 |
| residual ("thermal") | 3 |
| Marchenko-Pastur edge | 147 |
| MAPCA AIC / KIC / MDL | 28 / 24 / 13 |

The real data behave as the simulation predicted:
- The index-by-index null sits far above the data beyond k ~ 17 (it inherits shared spectral power).
- The deflated null tracks just below the data from k ~ 17 to 168, a small systematic gap like the ~2 % seen with coloured noise. It does stop (at 168, not k_max = 300), where the data are already below the white-noise edge (1.21).

New finding: the residual channels are not a pure thermal reference. Their top eigenvalues are ~27, ~10, ~8, i.e. a few strong components shared across voxels that lie outside span{1, TE}. Candidates: TE-nonlinear respiratory B0 / dephasing (the frame-wise residual tracks unfiltered FD), echo-specific ghosting, MB leakage. Not verified. Their bulk is ~1.1-1.2. The "thermal" estimate (k = 3) compares against the residual's top eigenvalue and is therefore not a thermal null. It should use the residual's bulk, after the residual's own structured components are removed.

Gate still not passed.

### 2026-11-04: surrogate-regressor check on P04 (mc)

Share of energy kept after projection (5 surrogate sets: same spectrum and cross-spectra as the motion parameters, scrambled timing):

| Channel | Real regressors | Surrogates, mean [range] | Removed by chance | Motion-specific (extra) |
|---|---|---|---|---|
| c0 | 0.50 | 0.72 [0.68-0.76] | 28 % | ~22 % |
| c1 | 0.71 | 0.84 [0.82-0.86] | 16 % | ~13 % |
| r0 | 0.78 | 0.87 [0.85-0.89] | 13 % | ~9 % |
| r1 | 0.84 | 0.90 [0.89-0.91] | 10 % | ~6 % |

The real regressors remove more than any surrogate, so part of the removal is motion-specific. But regressors of this spectrum also remove ~28 % of c0 by chance: slow signal, which includes scanner drift and possibly neural slow fluctuations. The df share alone is ~3 %. This is a cost of the 24-parameter set. A lighter set (6 or 12 terms; fewer slow regressors) has not been compared yet.

### 2026-11-04: regressor set kept at 24 (from the comparison); Phase 3 gated estimator replaced (user: option 1)

**Regressor sets on P04 (mc), share of c0 energy:**

| Set | Removed by chance | Motion-specific | Residual frame energy |
|---|---|---|---|
| 6 | 0.27 | 0.17 | 1.109 |
| 12 | 0.23 | 0.23 | 1.102 |
| 18 | 0.28 | 0.21 | 1.088 |
| 24 | 0.26 | 0.24 | 1.085 |

Every set removes the residual's link to head drift (Spearman 0.86 -> -0.30 to -0.41). Chance removal is the same for all sets, within the surrogate spread (~ +/- 0.04): it comes from the 6 slow parameters themselves. The 24-term set removes the most motion-specific energy and leaves the cleanest residual. Kept at 24 (user deferred to the recommendation). The ~25 % chance removal of slow signal is a limitation of any motion regression here; the no-regression arm stays as the comparison.

**Phase 3 gated estimator:** split-half deflated parallel analysis on noise-whitened channels (rank.py docstring).

| Simulated case (gate tests) | Result |
|---|---|
| 30 components, narrowband or broadband, white or coloured noise | within +/- 1 |
| 1 component | exact |
| pure coloured noise | 0-1 |
| projected data (24 regressors) | within 10 % |
| 60 narrowband components | 60-66 |
| 60 broadband components | over-count 10-37 % (known limitation, xfail) |

A contamination diagnostic (shared-signal share of the median row variance) is reported with every run. Its validated range is being calibrated: the first estimate of 0.45 was too low, since a passing 30-component case scored 0.476.

**Calibration of the contamination diagnostic (two seeds, coloured noise):**

| Case | k found | Contamination |
|---|---|---|
| 30 components (narrowband or broadband) | 29-31 | 0.469-0.481 |
| 60 narrowband | 60-61 | 0.665 |
| 60 broadband | 64 and 74 | 0.680-0.692 |

Validated limit set at **0.50**. Values from 0.50 to ~0.67 are untested and flagged. Gate tolerance: +/- 10 %, never finer than +/- 1. Each tested direction passes by chance with probability 5 %; observed K = 1 -> k = 2 once.

**Phase 3 gate tests (simulation) pass under seeds 0-2:** 30 components (4 kinds), 1 component (4 kinds), pure coloured noise, projected data, spatial blocks, end-to-end. The slow 60-component checks: narrowband must pass; broadband is a known limitation (xfail, flagged by the diagnostic).

The real-data part of the gate (spectra plotted) follows on P04.

### 2026-11-04: Phase 3 on P04 with the new estimator: k ran to k_max = 300; split changed to readout slabs

Result:
- gated k = 300 in both halves; contamination 0.22 (inside the validated range);
- index by index 17; MAPCA 28 / 24 / 13;
- beyond ~100 directions, half B's variance along half A's directions stays a constant ~7 % above its null (0.70 vs 0.65) up to 300.

A flat, non-decaying excess suggests noise shared between the halves rather than 300 components. Not verified. Plausible cause in this acquisition: MB-6 and GRAPPA-2 unaliasing correlates thermal noise between voxels acquired together (slice + 8, shifted along PE by the unknown CAIPI shift; and PE + half FOV). The 4^3 cube split put these partners in different halves, breaking the split's assumption of independent noise.

Change (no new tuning of thresholds): halves are now alternating 4-voxel slabs along the readout axis (`rank.split_mode: readout_slab`). Every MB / GRAPPA partner shares the readout coordinate, so it stays in its partner's half whatever the CAIPI shift.

New diagnostic: correlation of the residual r0 between voxels 8 slices apart (MB partners) vs 4 slices apart (control), at every circular PE shift, plus the GRAPPA half-FOV partner. If MB noise correlation exists, its peak estimates the CAIPI shift (to be checked against the protocol; unverified). Tests: planted partner at (shift 13, dz 8) found, control flat; readout slabs keep slice + 8 and any PE shift in one half.

### 2026-11-04: readout-slab rerun on P04: k still 300; MB-alias hypothesis rejected (STOP, user decides)

Result: gated k = 300 (both halves), the same as with cubes.

Noise correlation in r0:
- MB partners (slice + 8): peak -0.007 at any PE shift; control (slice + 4): 0.001. No MB alias noise correlation, so no CAIPI shift can be inferred.
- GRAPPA half-FOV partner: 0.077, real but modest (the slabs already keep these together).
- Neighbour along PE (shift 1): 0.19, likely from motion-correction interpolation (not checked on raw data).

New candidate, with simulation support: noise whose amplitude varies over time in the same way in every voxel (e.g. breathing depth, CO2 or arousal modulating physiological noise; motion periods). The phase-randomized null assumes stationary noise and cannot reproduce this, so many directions pass. Simulation (30 broadband components, coloured noise):

| Shared noise-amplitude envelope (SD) | Gated k | Halves |
|---|---|---|
| none | 32 | 31 / 32 |
| 0.14 | 31 | 30 / 32 |
| 0.28 | 68 | 49 / 86 |

The size of such an envelope in the real c0 noise is unknown.

Third failed attempt on real data for the Phase 3 gate (CLAUDE.md section 2: stop and ask).

### 2026-11-04: last attempt (user) failed in simulation; Phase 3 closes on the fixed sweep (agreed fallback)

Attempt: divide every frame by the shared noise amplitude a(t) before whitening. a(t) is the median squared data after removing the index-by-index components; dividing is a common temporal transform, so ranks are kept.

| Simulated case (30 broadband components, coloured noise) | Gated k | Envelope SD, true / estimated |
|---|---|---|
| envelope SD 0.15 | 37 (halves 42 / 32) | 0.15 / 0.13 |
| envelope SD 0.30 | 92 (halves 120 / 64) | 0.30 / 0.24 |
| stationary | 34 (32 without the correction, i.e. slightly worse) | 0 / 0.03 |

An earlier variant (envelope re-applied to the null after whitening) did not help either (107 vs 120); the order was wrong, since the whitening filter mixes frames.

Conclusion:
- With tens of thousands of voxels the split-half test is precise enough that any departure from its null (here, residual nonstationarity) counts as shared structure.
- A single "true rank" is probably not well defined for these data: shared structure exists at all scales. The user's framing applies: we are near the limit of what physics, measurement and analysis can provide.
- The correction is off (`rank.null_envelope: false`). a(t) is measured and reported as a diagnostic of noise nonstationarity, and the time-varying-amplitude case is a documented limitation (xfail).

**Decision (user, agreed in advance as the fallback): Phase 3 closes without a gated k.** Phase 4 uses the fixed sweep: MAPCA AIC / KIC / MDL (13 / 24 / 28 on P04) and k = 50, 70, 90. Every result is reported across k. The split-half estimator, the index-by-index count, the whitened Marchenko-Pastur crossing (~55 on P04) and the envelope SD are reported for each subject as descriptions, not as gates.

What the attempts established:
- no MB-alias noise correlation;
- GRAPPA half-FOV correlation 0.08;
- neighbour correlation 0.19 (interpolation);
- structured components in the residual channels;
- the index-by-index null inherits shared spectral power;
- deflation depletes coloured noise;
- phase-randomized nulls cannot represent nonstationary noise at this sample size.

### 2026-11-04: Phase 4 built (option 1 + metrics); simulation gate passes

**Implementation** (`decompose.py`, `metrics.py`, `mejoint decompose`, `scripts/phase4_subject.lsf`, `scripts/plot_components.py`):
- Stacked [c0 | c1] rows; T x T Gram PCA in float64; reduce to k ourselves.
  Verified in scikit-learn 1.9.1: with `whiten=False`, FastICA ignores `n_components` (with a warning) and treats the input as already whitened; `components_` is the unmixing matrix. The cluster's version is recorded in `done.json`.
- 30 FastICA restarts in parallel (joblib; 1 BLAS thread each), clustered on |map correlation| (= |w_a . w_b| in the whitened space). Per cluster: centrotype, Iq, restarts present.
- Joint GLM maps; back-substitution to s and q (s^-1); sign = positive skewness of [m0 | m1].
- Stability across the 30 restarts: Hungarian on |corr|, median per component.

**Metrics: three changes from CLAUDE.md 7.1 as written, each found in simulation:**
1. **eta uses cross-fitted maps.** Each voxel's betas come from a time course estimated from the other voxel half (readout slabs). The fixed-regressor debiasing of 7.1 assumes the regressor was not chosen with the same noise. ICA time courses are, so a pure dR2* source showed E_S0 = 8.9 of 98 (eta 0.09, "mixed"). Subtracting a constant selection bias over-corrected strong components (eta 4.0).
2. **Energy null through the same selection, family-wise.** Single random or phase-randomized time courses ignore that components are top-variance directions: on pure noise every component sat ~10x above that null. The null is now the top-k PCA directions of phase-randomized surrogates (20), and a component must beat the 0.95 quantile of the largest of the k.
   - Two passes: components above a first, conservative null are removed before the final surrogates. Otherwise the surrogates inherit the strong components' power (threshold 154 rejected a 148-energy dS0 source).
   - On pure noise nothing is removed, because removing selected noise directions depletes the null (a false positive when it was done).
3. **Labels use matched-SNR bands plus a 0.05 cross-talk margin.** The bands hold thermal noise only, while ICA leaves a few % cross-talk (pure dS0 eta 0.971 against a band starting at 0.978). Real-data bands by injection belong to Phase 5.

**Gate tests** (tests/test_decompose.py; seeds 0 and 1: 12 passed each; plus an end-to-end test). Simulation: 32 x 32 x 18 grid, 400 volumes, thermal SNR per echo 106 / 78 / 57 / 42, tSNR ~106 / 107 / 90 / 75.

| Test | Result |
|---|---|
| Planted recovery: 2 dR2*-only (0.5 s^-1), 2 dS0-only (1.5 %), mixed same / opposite sign, at k = 6 and 10 | time courses |r| >= 0.976; q >= 0.976; s >= 0.966 (footprint + 2 voxels) |
| eta bands | dR2* ~0.00 (R2*-dominant); dS0 0.97-1.01 (S0-dominant); mixed 0.36-0.38 (mixed) |
| Pure noise (2 seeds) | no component above the null |
| dR2*-only data | every significant component R2*-dominant (largest eta 0.075) |
| Extra components at k = 10 | at most one just above the null |
| tedana equivalence (linear form, equal weights, k = number of signals, 2 seeds) | matched |r| = 1.000 |
| Stability across 30 restarts | all planted components > 0.95 |

Detection limit: dS0 sources at 0.5 % (SD at peak) were barely above the null at these SNRs; hence 1.5 % in the tests (inside CLAUDE.md's 0.5-2 %).

Equivalence beyond the signal count: at k = 8 with 6 signals, the 6 signal components still match at >= 0.997, and the noise directions are arbitrary (ICA cannot identify Gaussian directions).

Remaining for the gate: real-data run on P04 across the sweep (MAPCA 13 / 24 / 28 and 50 / 70 / 90), with stability reported.
User confirmed all three changes (cross-fitted eta; family-wise surrogate null with two passes; band margin), 2026-11-04. CLAUDE.md 7.1 updated. P04 Phase 4 sweep submitted by the user.

### 2026-11-04: Phase 4 sweep on P04 (mc, 24 regressors): first real-data results

| k | Restarts converged | Median stability | Variance in k dims | Labels |
|---|---|---|---|---|
| 13 | 30 / 30 | 1.000 | 34 % | all "mixed", all above null |
| 24 | 30 / 30 | 1.000 | 39 % | all "mixed", all above null |
| 28 | 30 / 30 | 1.000 | 40 % | all "mixed", all above null |
| 50 | 30 / 30 | 0.999 | 44 % | all "mixed", all above null |
| 70 | 30 / 30 | 0.997 | 47 % | all "mixed", all above null (energy / null 11-430) |
| 90 | 30 / 30 | 0.999 | 50 % | all "mixed", all above null |

**Component types** (figures at k = 24 and 70):
- motion spikes (single large events, edge maps);
- fast breathing-band oscillations with edge and posterior-fossa maps;
- slow network-like fluctuations.

**eta** orders them physically: slow network-like components 0.18-0.23 (the lowest), breathing-band and edge components ~0.43-0.47, motion-spike components up to ~0.6.

**Problems found:**
1. **||s||, ||q|| bootstrap intervals sat above their own estimates** (e.g. 1.17 [1.25, 1.81]). A plain map norm is biased upward by noise, and resampling frames adds noise. Fixed: the norms and every bootstrap replicate now use the cross-fitted maps with the expected noise removed, using the weighted-refit beta variance diag(G^-1 M^T W^2 M G^-1). Simulation: planted norms recovered within 2 % and inside their intervals (new test). The P04 sweep needs a rerun for the corrected norms.
2. **The energy null is uninformative on real data:** every component is above it at every k. Same root cause as Phase 3: coloured, nonstationary noise counts as structure for phase-randomized surrogates. The tests pass because simulated noise is stationary.
3. **Labels are uninformative on real data:** all "mixed", because the matched-SNR bands contain thermal noise only and are +/- 0.002-0.02 wide for strong components. Real-data bands need injection (Phase 5). Until then eta is read as a continuous measure.

**Open question:** eta ~0.2 for the most BOLD-like components. Candidates: real intravascular / inflow dS0 (CLAUDE.md 7.1), or partial volume with CSF. A voxel mixing tissue and CSF has a BOLD change that is not linear in TE, and a linear fit in TE then produces an apparent dS0 intercept. Unverified; testable with a two-compartment simulation.

**Note on stability:** the 30 restarts use the same data, so stability measures how reproducibly ICA converges, not reliability. Split-half reliability is a Phase 5 test.

### 2026-11-04: Phase 4 gate passed (user); Phase 5 begins

User decisions:
- **Phase 4 gate passed** on the simulation tests and the reported stability. On real data, the energy null and the labels are treated as uninformative until Phase 5 provides injection-based eta bands and split-half reliability. eta is read as a continuous measure meanwhile.
- **Tissue classes:** each session has a T1w scan. Two options will be built:
  - T1-based (segmentation registered to EPI space);
  - **data-only** (classes derived from the multi-echo data alone), so the analysis can still run from this data only.
- **Traveling-wave detection** (`docs/WAVES_BRIEF.md`) is implemented after Phase 5 and before Phase 6.

Phase 5 order (proposed):
1. Partial-volume check in simulation (does tissue/CSF mixing create apparent dS0 for BOLD?).
2. Tissue classes, data-only first, then T1-based (needs the T1w location).
3. Injection into raw echoes through the full pipeline (eta bands on real data).
4. Arms A-G; motion check; MB-leakage test; split-half reliability.

### 2026-11-04: Phase 5 step 1: partial volume with CSF creates apparent dS0 for BOLD (our derivation, numerical)

Two-compartment voxel: tissue T2* 45 ms carries a pure dR2* change of 0.5 s^-1; CSF T2* 200 ms does not. Passed through the real front-end projection and back-substitution (per-echo weights, echo-1 SNR 100).

eta of the voxel's change:

| Tissue fraction | CSF / tissue S0 = 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| 0.9 | 0.003 | 0.012 | 0.039 |
| 0.8 | 0.015 | 0.047 | 0.127 |
| 0.7 | 0.037 | 0.104 | 0.231 |
| 0.6 | 0.073 | 0.179 | 0.331 |
| 0.5 | 0.127 | 0.266 | 0.419 |

- The apparent dS0/S0 is negative (-0.09 to -0.27 %) for this dR2* increase (the signal falls): the same sign as the signal change and opposite in sign to dR2*. The apparent dR2* is diluted (e.g. 0.13 instead of 0.5 s^-1). (Wording corrected 2026-11-05; it first said "opposite in sign to the signal change". The numbers were right.)
- Whitened energy of the change at tissue fraction 0.6: c0 98.1 %, c1 1.6 %, r0 0.3 %, r1 ~0. It is invisible per frame in the residual.
- Its static fingerprint is a convex mean decay (curvature about +0.015). P04's static curvature has its 90th percentile at +0.074, so many voxels have convex, partial-volume-like decays.

Consequence: eta ~0.2 for BOLD-like components on P04 is explained by partial volume alone. eta in mixed-tissue voxels is not evidence of physiological dS0. This changes how eta must be read (CLAUDE.md 7.1, "do not hard-label dS0 as noise", also means "do not read dS0 as physiological without partial-volume control").

Options, for the user:
- (a) report eta stratified by CSF partial volume (T1 segmentation, or data-only from the static decay);
- (b) model correction: a per-voxel two-compartment design, -TE w_v(TE) with w_v the tissue signal fraction from a bi-exponential fit of the mean decay or from T1 partial-volume maps (changes the model; needs validation);
- (c) a coupling diagnostic: partial volume predicts s anti-correlated with q within a component, with a ratio set by each voxel's static decay.

### 2026-11-04: Phase 5 data-only tissue classes built; partial-volume diagnostic is work in progress

Built:
- `tissue.py` and `mejoint tissue`: a two-compartment fit of the mean decay (fixed long T2* 200 ms), the long-T2* fraction f_long, the coupling kappa, and data-only classes (tissue / mixed / CSF-dominated / short T2* / few echoes).
- Simulator option `csf_partial_volume`.
- Per-component eta and energy share by class (option a).

Validation:
- f_long recovered within +/- 0.01 (0-0.8);
- concave (dephased) decays give 0;
- kappa matches the direct two-compartment projection within 1 %;
- in a partial-volume simulation, f_long vs the true CSF fraction has r = 1.00.

**Option c (coupling diagnostic) is not finished.** It compares the zero-parameter prediction s = kappa q with the best constant s/q ratio.
- It correctly calls the mixed source (ratio 0.031-0.034 vs true 0.030) and the dS0 source "not partial volume".
- For dR2* sources in strong partial volume, kappa q in voxels with f_long 0.5-0.9 is large and multiplied by a noisy q, which swamps the debiased energies (pv_explained -2.6 although the per-class medians matched the prediction within 2 %).
- Test marked xfail (not strict) and paused by the user's request to start the wave extension.

Next step when resumed: weight voxels by the precision of kappa q, or fit in the CSF fraction domain.

## Wave extension (branch claude/waves)

### 2026-11-04: W0: brief and mock-ups read; branch created; conflicts flagged; questions asked

- Branch `claude/waves` created from `claude/beautiful-hamilton-141ubf` at 90f5730. The mock-ups (`docs/mockups/waves/`) existed only on `main` (uploaded 2026-11-03) and were copied into this branch.
- `make_wave_figures.py` was checked: plotting code with simulated maps, no decomposition. It is used for style only (palette, layout). No earlier CPCA code is used or consulted (user decision 2026-11-03).

**Conflicts with the current code** (the brief was written at Phase 2):
1. **decompose exists now.** `--waves` / `--wave-figures` go straight into `mejoint decompose` (brief 6.2), alongside the standalone `mejoint waves`. One code path.
2. **Per-echo noise.** The front end whitens with per-echo weights (r = [1, 0.721, 0.631, 0.560]), and R comes from those weights. The wave code takes R from the front end, as the brief requires, and back-substitution stays exact. But the c0-only equivalence with OC (brief 4.6) holds only with equal weights (CLAUDE.md 5.3 property 1), so that test runs with `frontend.weighting: equal`, as in Phase 4.
3. **Nuisance projection.** The cached channels already have the 24-regressor projection applied. On P04 it removed about half of c0's energy, ~25 % by chance overlap with slow signal, which is exactly the 0.01-0.10 Hz wave band. Whether waves use the regressed or the no-regression channels is a user decision.
4. **Selection bias in eta and in the energy null** (Phase 4 findings): modes are top-variance directions, so
   - eta uses cross-fitted complex maps;
   - the null uses the top-n directions of surrogates, family-wise.
   Phase-randomized surrogates are uninformative on real data here (coloured, nonstationary noise; Phases 3-4), so "above null" will need the bootstrap / split-half evidence on real data.
5. **Partial volume** (Phase 5): CSF partial volume inflates eta with an apparent dS0 that is in phase with -q (the BOLD-positive part; wording corrected 2026-11-05, it first said "in phase with q"), so it does not create a TE-dependent lag. TE lag is robust to it; eta is not. Report both, with the PV diagnostic.
6. **Gray matter** is not available data-only. The data-only classes are tissue / mixed / CSF / short T2* / few echoes. The phase convention ("q in gray matter") and the speed statistics use the data-only tissue class until the T1 option exists.
7. **Slice timing** is not in the front-end cache. It is read from the JSON sidecar of the raw echo 1 (found through result.json), and recorded.
8. **ANTs:** the brief's path is under `rhel7_apps`, but the cluster is RHEL8. To verify at W5.
9. **Atlases** (Schaefer-400; Ren et al. 2019) must be downloaded and checked (space, licence, labels). Unverified; W5.
10. Bolt et al. (2022) is cited as a conceptual reference only; details not verified.

### 2026-11-04: W0 gate passed: user answers

1. **Hierarchy / parcels:** impose parcels as little as possible; individualized and data-driven. Waves should be shown as gradient-like phase lag across voxels. Consequences:
   - the individual stage centres on voxelwise lag maps, lag gradients and the fitted propagation axis;
   - the group stage's parcel-based correspondence (brief 7.2) will be revisited at W4 in a data-driven form;
   - the hierarchy-order lag profile is replaced by the subject's own path profile.
2. **Labels:** none until the batch stage.
3. **Brainstem / deep cerebellar nuclei:** leave out.
4. **Flatmaps / surface views:** later.
5. **Channels:** no-regression channels as primary, motion-regressed as sensitivity (band overlap of the motion regressors with 0.01-0.10 Hz).
6. **Modes:** `waves.n_modes` = 10 by default.
7. **Labels file location and ID format:** later, with batch processing.

### 2026-11-04: W1 gate passed: simulator extension

`simulate.py`: `waves` (neural planar or radial, vascular, standing), `slice_timing` (multiband groups: slices z and z + n_groups acquired together; interleaved group order by default; an explicit SliceTiming list overrides) and `slice_leakage`.
- Delays are applied exactly in the Fourier domain to a band-limited random time course.
- Waves draw random numbers only when configured, so every other simulation is unchanged (existing simulator tests pass).

Tests (tests/test_waves_sim.py, noise-free ground truth; 8 tests, 7-9 s per run; seeds 0-2):
- planar delays = distance / speed (within 0.02 s);
- radial delays = distance from the centre / speed;
- pure dR2* waves have no echo-to-echo delay;
- the standing wave has no lag;
- with slice timing a later-acquired slice LEADS by its offset (the sign the correction must undo);
- vascular: the dR2* part lags the dS0 part by the hemodynamic delay (2.00 s), the vessel's echoes disagree in timing, and tissue carries dR2* only;
- leakage raises the correlation between slices acquired together;
- invalid kinds and empty bands are rejected.

**Flag (brief inconsistency):** brief 4.7 gives "lag = phase / (2 pi f_c)" but its gradient formula carries a minus sign. With analytic time courses ~ exp(+i w t), a voxel that lags by tau has map phase -w tau, so the implementation uses lag = -arg(q) / (2 pi f_c), consistent with the gradient formula, and the W2 tests verify the sign on simulated waves.

### 2026-11-04: W2: individual wave stage built; simulation tests pass, slow set included

`src/mejoint/waves.py` (fresh implementation), `src/mejoint/wave_figures.py`, `mejoint waves`, `mejoint decompose --waves [--wave-figures] [--waves-frontend-dir]` (the same `waves.run_dir`), `scripts/waves_overnight.lsf`.

Simulations (tests/test_waves.py): 32 x 32 x 24 grid, 3.28 x 3.28 x 3.3 mm, 600 volumes, TR 0.75 s, real TEs, per-echo noise ratios, MB 6 slice timing; tSNR per echo [87, 66, 47, 36]; waves band-limited to 0.03-0.05 Hz; seed 0.

| Test | Result |
|---|---|
| planar neural wave, 6 mm/s along +x | speed 5.94 mm/s, direction error < 0.03 degrees, R = 0.98, fit quality 1.00, lag vs slice-time slope after correction ~0 |
| lag sign (lag = -arg(q) / (2 pi f_c)) | later voxels have larger lag; SD of (lag - planted delay) < 1 s |
| standing wave, slice-timing correction on | lag vs slice-time slope -0.013, no propagation (speed > 1000 mm/s) |
| standing wave, correction off | slope -1.009 (the artifact, with the sign the correction undoes) |
| eta | neural ~0 (inside its matched-SNR dR2* band); vascular 0.145, above the band |
| TE4 - TE1 lag (95 % bootstrap interval of the median beyond the null) | neural [-0.27, 0.33] s (includes 0); vascular [0.52, 0.81] s (excludes 0, positive) |
| c0 only, equal weights, linear form, z-scored vs an independent reference (SVD of the band-passed analytic OC signal, written separately) | Hungarian-matched complex |r| = 1.000 for the top 3 modes |
| pure noise (20 runs, 20 x 20 x 16 grid, 450 volumes, 40 surrogates) | 1 / 20 runs with any mode above the family-wise null (5 %; pass: at most 1) |
| speed interval coverage (100 runs, planar 6 mm/s, q 0.25 s^-1, snr0 60, 100 bootstrap resamples) | 0.98 (pass: 0.90-0.98). At the upper edge: the intervals are slightly conservative (wide), not too narrow |
| `mejoint waves` / `decompose --waves` on a real-layout folder | SliceTiming read from the echo-1 sidecar; modes, maps, both figures, summary, done.json written |

Fixes found on the way (each verified by the tests above):
- **Slice-timing correction sign.** A slice acquired delta into the TR samples the signal delta later, so it must be delayed by +delta. The first version advanced it (slope -1.95 "corrected"); fixed (slope 0.017).
- **Equivalence test.** The test conjugated the time courses before matching; the analytic time courses of both codes are already the same (the right singular vectors' rows). The code was right, the test was wrong.
- **"Voxels where the mode is present"** is now |q| above 3 noise SDs of q (from the calibrated complex beta noise), not a fraction of the median amplitude, which diluted focal modes.

**Flags (please review):**
1. **TE-dependent lag needs s and -TE q of the same sign.** z_e = s - TE_e q. For BOLD-positive hyperaemia (inflow raises S0, dR2* falls) the two parts add and the TE lag is well defined. If they have opposite signs (dS0 up while dR2* rises), z_e passes near zero at TE = s/q, the phase jumps by about half a cycle between echoes, and the unwrapped TE lag is arbitrary. The vascular test therefore simulates the BOLD-positive case (q = -0.5 s^-1). The first version of the test used q = +0.5 and failed for this reason. On real data, a half-period TE lag in a voxel means "sign-opposed mixture", not a timing difference; the phasor panel (diagnostics E) shows which case applies.
2. **Fast waves are weakly constrained.** A 60 mm/s wave crosses this 105 mm field in 1.7 s: 0.43 rad of phase at 0.04 Hz. In one vascular simulation the speed was 66.9 mm/s [62.2, 71.8], missing 60; in the test seed it was 58.7-62.9. Speeds well above (field size x f_c) should be read as "fast", not as numbers.
3. **The plane-wave fit does not see the slice-timing artifact** of interleaved MB acquisition (the standing wave without correction still fits as "fast", 6839 mm/s): the artifact is not a smooth gradient. The lag vs slice-time slope is the check, and it is reported for every mode.
4. **Below-null modes** have meaningless TE lags (60-90 % of voxels "beyond the null"). Interpret every per-mode measure only for modes above the energy null; the summary table carries the eigenvalue / null ratio first.
5. **Voxelwise eta** in figure 1 C is computed from debiased energies smoothed by 6 mm (display only, `waves.eta_display_fwhm_mm`); per-mode eta uses unsmoothed energies.
6. **Orientation.** Figures are shown in RAS (participant's right on the right of the image) from the header's orientation codes; checked on a simulated LPS header (the wave appears mirrored, as it should). Not yet seen on a real header.
7. **Runtime.** 32 x 32 x 24 grid, 600 volumes: about 2 min with 30 bootstrap resamples and 10 surrogates (4 cores). At real size (benchmark, 2 shared cores): complex PCA of 200k stacked rows x 800 frames about 15 s; band-pass + slice-timing correction of 100k voxels x 2 channels about 55 s single-threaded, now threaded over voxel chunks (LSB_DJOB_NUMPROC threads; identical results, 2x on 3 shared cores). Expected on 8 cores: primary run (200 bootstrap, 200 surrogates) about 2-3 h, the sensitivity runs about 1 h together. Not yet measured on the cluster.

### 2026-11-05: W3 first real-data run on P04 (mc, no regression): observations, review pending

Job: `scripts/waves_overnight.lsf`, 8 cores, 1 h 40 min for all steps (wave stages and the joint-ICA sweep), max memory 9.0 GB of 32 GB requested. Every step completed.

Primary wave run (`outputs/waves/<run>/noreg`): 800 contiguous volumes, slice timing from the raw echo-1 JSON, reference voxels = data-only tissue class.
- **Only mode 1 is above the family-wise energy null** (eigenvalue / null 2.79; f_c 0.033 Hz; 25.9 % of the band-passed variance). Modes 2-10 are below it (0.95 down to 0.23) and are not interpreted.
- **Mode 1** is a large-scale mode: plane-wave speed 29.1 mm/s [22.3, 34.5], fit quality 0.76, but local directional coherence R = 0.12. The lag profile along the fitted axis is not linear: about 0-2 s over the first ~110 mm, then a rise of about 6-8 s over the last ~30 mm. A plane wave is a poor summary of this mode; the lag map is the better one.
- **eta** = 0.23 for mode 1 (0.25-0.43 for modes 2-10), far above the thermal-only dR2* bands (about +-0.01). Same pattern as the Phase 4 sweep: the thermal bands are too narrow for real data, and partial volume with CSF alone can produce eta ~0.2 (Phase 5). Not evidence of physiological dS0.
- **TE4 - TE1 lag**: mode 1 median 0.09 s [-0.08, 0.30] (includes 0), but 63 % of voxels "beyond the null". In all three plotted modes the voxels beyond the null pile up near +-half a period (15.2 s, 22.7 s and 15.6 s), which is the signature of W2 flag 1 (s and -TE q of opposite sign, so z_e passes near zero and the unwrapped phase jumps by about half a cycle), not of a timing difference: a 15 s delay between the dS0 and dR2* parts is not physiological. The posterior-fossa phasor of mode 1 crosses the origin (sign flip between TE1 and TE4). (Corrected 2026-11-05: this entry first said that CSF partial volume predicts this. It does not: partial volume gives an apparent dS0 in phase with -q, so z_e keeps one sign across echoes; see the next entry.) The histogram also clips values at +-half a period, which exaggerates the edge bins.
- **Lag vs slice-time slope** of mode 1 after correction: 0.33 (simulation: ~0). In lag terms this is about 0.23 s across the 0.75 s TR, small against the ~8 s lag range, but not zero. The no-correction run gives the comparison.

Joint-ICA sweep with tissue classes (k = 13, 24, 28, 50, 70, 90): every component "mixed" and above the energy null at every k; stability 0.997-1.000. Unchanged from Phase 4: the thermal-only eta bands and the energy null are uninformative on real data (injection is the planned reference).

Proposed: a sign-aware dS0-dR2* timing measure per voxel (the relative phase of s and -q folded to +-90 degrees, plus a "sign-opposed" flag), compared with the CSF fraction f_long, reported beside the brief's TE4 - TE1 lag (agreed and built 2026-11-05, below).

Mode 1 figure (`wave_mode_1.png`), added 2026-11-05:
- Over most of the brain the dR2* map is faint and nearly in phase (lag near 0 within the 30 s cycle): a near-synchronous, global-like mode. The strongest voxels are at the brain's edges, around the midline at the ventricle level, and in the most inferior slices.
- The cycle snapshots (sagittal) are strongest along the superior surface (vertex) and at the posterior-inferior and anterior-inferior edges, and the anterior-inferior and posterior-inferior edges have opposite sign at the same moment.
- In the space-time plot the first ~120 mm along the fitted axis oscillate together; the last two bins (~120-140 mm, the far end of the axis) are stronger and shifted in time. That end bin drives the 6-8 s lag rise and the 29 mm/s fit.
- Two readings, not yet separable: (a) venous / systemic blood-borne oscillation, with the draining sinuses late (hypothesis); (b) slow head motion (these are the no-regression channels): opposite-sign anterior and posterior edges fit a pitch rotation. The motion-regressed run and a correlation of the mode time course with the motion parameters would separate them.

Sensitivity runs (same job), mode 1:

| Run | Var % | Eigenvalue / null | f_c Hz | Speed mm/s | Fit / R | eta | TE4-TE1 median s (beyond null %) | Lag-slice slope |
|---|---|---|---|---|---|---|---|---|
| no regression (primary) | 25.9 | 2.79 | 0.033 | 29.1 [22.3, 34.5] | 0.76 / 0.12 | 0.234 | 0.09 (63) | 0.33 |
| 24-term motion regression | 25.9 | 2.86 | 0.032 | 38.0 [29.0, 45.5] | 0.78 / 0.13 | 0.229 | 0.30 (61) | 0.47 |
| no slice-timing correction | 26.4 | 3.25 | 0.033 | 30.1 | 0.76 / 0.12 | 0.233 | 0.07 (63) | -0.40 |
| c0 only, z-scored (OC-like) | 14.7 | 1.56 | 0.031 | 27.3 | 0.88 / 0.24 | n/a | n/a | 0.05 |

- **Motion regression barely changes mode 1** (absolute eigenvalue about 10 % lower; same maps, same late end in the space-time plot). Motion that is linear in the 24 regressors is unlikely to be the source. Non-linear motion effects (spin history) are not excluded.
- **Slice-timing correction shifts the slope by +0.73** (simulation: +1.0), so it acts in the right direction and nearly the right size; the residual +0.33 corresponds to about 0.2 s across the TR.
- **Mode 1 is also the leading mode of the OC-like c0 path**, with a smaller share (14.7 %) when every voxel is z-scored: in whitened units its energy is concentrated in high-amplitude voxels (edges, vessels, ventricles).
- No other mode is above the null in any run except mode 2 of the c0 path (1.03, marginal).

### 2026-11-05: W3 additions built (user decisions: a-c, e; phase maps; all four display recommendations)

User decisions (2026-11-05):
- (a) motion correlation per mode; (b) sign-aware dS0 / dR2* timing; (c) phase, amplitude and eta by region; (e) waves-only reruns.
- Figures show **phase**, not time: hsv, 0 = the reference voxels, + = later. Main map (i) from the mode; check map (ii) the direct Hilbert relative phase (each voxel's c0 phase minus the mode's phase, averaged over time). Phase 0 = the tissue-class mean. Fading by each voxel's explained fraction (map i) or phase-locking value (map ii). Space-time panel shows the band-passed **data** in distance bins with the mode's crest lines from its instantaneous phase; the mode's own reconstruction is a small companion panel (it has propagation built in).
- The fixed-period dashed lines of the first figures drifted off the crests because the mode time course is broadband (crest spacing varies); the new lines follow the instantaneous phase.

Built:
- `waves.direct_relative_phase`, `motion_correlation` (null: phase-randomized motion parameters, 95th percentile of the largest of 6), `s0_r2_timing`, `class_summary`, `spacetime_data`; new maps `phase_lag_deg`, `hilbert_phase_lag_deg`, `plv`, `explained_fraction`, `s0_vs_r2_phase_deg`; `modes_by_region.tsv`; new summary tables.
- `scripts/waves_overnight.lsf '<run>' [all|waves] [config]`; wave outputs now go to `outputs/waves/<run>/<git commit>/` so reruns never overwrite earlier results; memory request lowered to 2000 MB per core (the first run peaked at 9.0 GB).

**Correction (my error, found by the new simulation test):** CSF partial volume gives an apparent dS0 with the **same sign as the signal change**, i.e. in phase with the BOLD-positive part -q (opposite in sign to dR2* itself). Direct calculation (tissue T2* 45 ms, CSF 200 ms, dR2* = -1 s^-1): apparent dS0 = +0.30 %, +0.50 %, +0.38 % at CSF fractions 0.1, 0.3, 0.6. Three earlier statements had the sign relation wrong in words (the Phase 5 entry, W0 conflict 5, and the W3 phasor remark; all corrected above, numbers unchanged), as did README Figure 6's caption. Consequences:
- partial volume does not produce sign flips across echoes, so it does **not** explain the half-period pile-up of the TE4 - TE1 lag on P04;
- the remaining candidates are (1) noise larger than the thermal noise the TE-lag null assumes (physiological noise; the same miscalibration as the eta bands), and (2) a real dS0 of the same sign as dR2* (S0 up while R2* rises, e.g. edge or motion effects).

**Noise for the sign check.** The per-voxel s and q estimates share noise (CLAUDE.md 5.2), which by itself favours opposite signs: on simulated pure noise every passing voxel was "opposed". The check therefore (1) uses empirical noise per voxel (the thermal beta variance scaled by the voxel's residual band power relative to unit noise through the same pipeline; physiological noise included), and (2) reports a pure-dR2* null (expected number of passing voxels and their opposed fraction). The brief's eta and TE-lag nulls are unchanged (thermal); switching the TE-lag null to the empirical noise is an option for the user.

Tests (tests/test_waves.py, all pass; real TEs, TR, MB 6 slice timing):
- vascular wave (dS0 inflow leads the BOLD-positive dR2* by 3 s): 89 voxels with both parts beyond noise (null: 0.45 expected), 2 % opposed, dS0 timing -2.90 s (planted -3.0 s);
- pure dR2* wave with CSF partial volume (fraction up to 0.7): 2623 voxels with apparent dS0, 0.1 % opposed, timing 0.03 s; same-sign fraction by CSF fraction 0.01 / 0.35 / 0.60 / 0.53 (0-0.1 / 0.1-0.3 / 0.3-0.5 / 0.5-1.0);
- the sign null reproduces pure dR2* plus correlated noise (count within 15 %, opposed fraction within 0.05);
- a planted motion-driven mode correlates 1.00 with its motion parameter, above the null;
- the command test checks the new maps and tables, the motion file found through the pipeline folder, and that the direct Hilbert phase locks to a simulated wave (median PLV > 0.8).

### 2026-11-05: W3 second real-data run on P04 (waves only, commit 09de749): observations

Primary (no regression) and motion-regressed runs agree; numbers below are the primary run, mode 1 (the only mode above the energy null).

- **Motion coupling.** The mode time course correlates 0.60 with the band-passed y translation (null 95th percentile 0.40), and still 0.53 on the motion-regressed channels. The 24-regressor projection is fitted over the whole band, where the motion traces are dominated by slow drift, so it leaves in-band coupling. The earlier inference "motion regression barely changes mode 1, so motion is unlikely" is withdrawn: regression did not remove it, but the mode follows a motion trace. Direction of cause (motion driving the signal, or a shared physiological driver) is not determined.
- **Real dS0, mostly sign-opposed.** 2198 voxels have both dS0 and dR2* beyond the empirical noise (2.0 expected for pure dR2*). 91 % are sign-opposed (S0 and R2* change in the same direction at the same time; folded timing -0.07 s), so their effects on the signal partly cancel. As a fraction of the voxels where the mode is present: opposite sign 0.25 / 0.17 / 0.21 / 0.27 and same sign 0.02 / 0.04 / 0.03 / 0.05 across CSF fraction bins 0-0.1 / 0.1-0.3 / 0.3-0.5 / 0.5-1.0. The opposite-sign fraction does not follow the CSF fraction; the small same-sign fraction rises with it, as partial volume predicts.
- Consequences: for this mode, eta (0.23) is carried mainly by sign-opposed dS0, not by CSF partial volume; and the half-period pile-up of the TE4 - TE1 lag is these voxels (z_e changes sign between echoes), not noise.
- **Spatial structure.** Explained fraction median 0.088 in tissue (PLV 0.25). The binned band-passed data are nearly synchronous over the first ~120 mm of the fitted axis (data and reconstruction look alike); the last ~15 mm are shifted. The phase table is relative to the |q|-weighted convention, not to the tissue mean as the caption states: tissue -17.6 degrees, CSF -8.9, inferior third +7.1 (spread 0.75, weak). To fix: phase 0 should be the explained-fraction-weighted tissue mean (the agreed convention).
- The crest lines kink where the mode's amplitude is low (around 100-140 s): the instantaneous phase is ill-defined there.

### 2026-11-05: fix and checks a-d built; batch tooling for all participants (user decisions)

User: "implement the fix plus a-d. Then run it on all participants to see if this correlation with y-translation is unique."

- **Fix:** phase 0 = the circular mean phase of q over the reference (tissue) voxels weighted by each voxel's explained fraction (was |q|-weighted, dominated by edge voxels). The reference row of the region table is now 0 by construction (tested).
- **(a) Direction in anatomical terms:** the fitted direction (towards later voxels) is converted with the image's orientation codes to RAS and words (e.g. "posterior-superior"); `axcodes` is recorded in waves.json. Tested on a RAS simulation (+x = "right", +y = "anterior").
- **(b) Edge and interior regions:** edge = mask voxels with a face neighbour outside the mask (one voxel, about 3.3 mm). The region table now also gives, per region, the fraction of the voxels where the mode is present that carry a significant same-sign or opposite-sign dS0.
- **(c) Motion-orthogonal sensitivity** (`--motion-orthogonal`, `waves.motion_orthogonal`): the band-passed analytic motion parameters (6 complex = 12 real regressors) are projected out of the band-passed channels; the thermal-noise calibration and the surrogate null go through the same projection. Tested: the projection leaves nothing in the motion subspace, the mode's motion correlation becomes 0, and a simulated wave keeps its speed.
- **(d) TE4 - TE1 lag** is computed over the voxels where the mode is present minus those with a significant opposite-sign dS0 (the lag is not defined there); the number excluded is reported. The bootstrap uses the same voxels.
- Per-parameter motion correlations and the strongest parameter are columns of modes.tsv.

**Batch (description across participants, not the group stage):**
- `scripts/submit_waves_batch.sh runs.txt` submits an LSF job array (`wavesb`, 8 cores and 2000 MB per core per element, at most 8 at a time) of `scripts/waves_batch.lsf`, which runs `waves_overnight.lsf <run> batch`: front ends (mc, with and without motion regression), tissue classes, waves primary (no regression, figures, 200 bootstrap / 200 surrogates), motion-regressed and motion-orthogonal (50 / 50). A dependent job (`ended(wavesb)`) runs `mejoint waves-summary`, writing `outputs/waves_batch/<commit>/waves_batch.{md,tsv,json}` and `waves_batch_motion.png` (mode 1 vs each motion parameter per participant; largest coupling vs mean FD).
- Dry run here on two simulated participants with a reduced config: every step ok, summary written.
- **Prerequisite:** each run needs its motion-corrected echoes (`work/echo-k_desc-mc_bold.nii.gz`, mbme_tedana `--keep-intermediates`). Only P04 has them in `OUT_keep` as far as this record shows; the submit script lists runs without them, and those elements stop in preflight.

### 2026-11-05: whole pipeline for every participant (user request)

User: "run the entire pipeline for everyone for which there's suitable data" (the 17-run echo-1 list, minus the excluded P02 and P08).

- `scripts/submit_pipeline_batch.sh <echo-1 list> [mbme_tedana.py] [OUT_keep]` (login node):
  1. cleans the list (quotes, blank lines, #comments; subjects in `subjects.exclude` dropped), so line numbers match mbme_tedana's `--index`;
  2. runs `mbme_tedana.py check` (echo files and TEs for every line, no processing) and stops if it fails;
  3. submits one LSF job per participant (`mejoint_full[1-N]%8`, 8 cores, 4000 MB per core, `span[hosts=1]`) and the wave summary after all have ended.
- Each job (`scripts/pipeline_batch.lsf`): mbme_tedana `run --index i --keep-intermediates` into OUT_keep with the existing analysis settings (`--echo-times 16.20 31.22 46.24 61.26 --interp spline --ica-method robustica --n-robust-runs 30 --tedpca aic --tree tedana_orig --seed 42`, from the source pasted 2026-11-02; mbme_tedana skips runs already finished, e.g. P04), FSL 6 via modules, tedana env python by full path; then `waves_overnight.lsf <run> full`: front ends (with and without regression), tissue classes, rank (`mejoint rank --mapca-only`: spectrum and MAPCA AIC / KIC / MDL, the k values the sweep needs; the slow phase-randomized estimators are skipped), waves (primary, motion-regressed, motion-orthogonal), joint-ICA sweep with tissue classes.
- Tested here end to end with stand-ins for mbme_tedana and bsub (two simulated participants): list cleaning (an excluded subject dropped), check, submission commands, both jobs through every stage.
- Not verified here: the cluster run of mbme_tedana itself (module name `fsl/6` and the options are as recorded; the first job's log will show).

### 2026-11-06: whole-pipeline batch on all 15 runs: wave summary (code e67f906)

All 15 runs completed (mbme_tedana with --keep-intermediates, front ends, tissue, rank (MAPCA), waves, joint-ICA sweep). Wave summary (`outputs/waves_batch/e67f906/`), leading mode (mode 1):

| | No regression | 24-term motion regression | Band-passed motion projected out |
|---|---|---|---|
| Above the energy null | 15 / 15 | 15 / 15 | 12 / 15 |
| Motion-coupled (largest motion r above its null) | 12 / 15 | 6 / 15 | 0 / 15 (by construction) |
| Strongest parameter | trans_z 6, trans_y 5, rot_x 1 | trans_y 4, trans_z 1, rot_z 1 | - |
| r with trans_y, median (range) | 0.57 (0.12-0.82) | 0.24 (0.10-0.60) | 0 |

- **Not unique to P04.** In 12 of 15 runs the leading mode follows the y and z translations or the x rotation. With the 52 x 64 x 48 axes (phase encoding along x, left-right) these are probably anterior-posterior and superior-inferior translation and pitch, i.e. motion in the sagittal plane (mcflirt parameter axes not verified).
- **Coupling does not track the amount of motion.** Mean FD 0.09-0.46 mm; r = 0.82 at FD 0.095 mm (P07) and 0.88 at 0.16 mm (P03). Small slow motion common to all participants, or a shared driver of signal and apparent motion (for example breathing-depth changes, which affect both CO2-driven BOLD and head position; no physiological recordings to test this).
- **The 24-term regression removes only part** of the in-band coupling (6 runs still coupled); it is fitted over the whole band, where the motion traces are mostly drift.
- **A leading mode survives motion projection** in 12 of 15 runs (eigenvalue / null 0.88-2.67), usually weaker than without projection (e.g. P04 2.79 to 1.88, P14 2.72 to 1.12, P15 3.00 to 1.15); in 3 runs it is unchanged or stronger (P16, P17, P11). The projection also removes any non-motion signal that happens to share the motion's in-band time course, so this is conservative.
- The three runs whose mode 1 is not motion-coupled (P16, P17, P05) include the two with the highest eta (0.54, 0.49) and many sign-opposed dS0 voxels in edge and interior alike.
- Sign-opposed dS0 per present voxel is somewhat higher at the edge than in the interior in most runs, but substantial in the interior too; dS0 counts fall with regression or projection in the motion-coupled runs (e.g. P04: 2198, 2067, 871; P14: 19145, 2874, 853).
- **Direction is not consistent across participants** (posterior, superior, inferior, left, anterior), so the leading mode shows no common traveling wave.

### 2026-11-06: coupling lag, sweep summary, motion-projected waves primary (user decisions 1-3)

1. **Lag of the motion coupling** (`waves.motion_correlation`): the angle of the complex correlation between the mode time course and each band-passed motion parameter gives the delay of the mode's signal in the reference (tissue) voxels after the parameter, at the mode's f_c (+ = signal later). A motion parameter's sign is arbitrary, so the angle is folded to +-90 degrees (+-a quarter period, about +-7.5 s at 0.033 Hz); a delay and the same delay +- half a period with the opposite sign are not distinguished. Reported per parameter in modes.tsv (`motion_lag_s_*`, `motion_best_lag_s`), on the motion bar chart (diagnostics H) and in the batch summary (median and range over coupled runs; a lag panel in waves_batch_motion.png, measured on the no-projection comparison run, where the coupling exists). Test: a mode following a motion parameter 2 s later with the opposite sign gives +2.0 s (within 0.3 s).
2. **`mejoint sweep-summary`** (`src/mejoint/sweep_batch.py`): per run, labels at every k with the MAPCA estimates and stability; at a common k (default 50) eta (all, tissue class, CSF class, partial volume removed), partial-volume verdicts (components above the null only), stability, variance and the largest correlation of a component time course with FD; `sweep_batch.{md,tsv,json}`, `sweep_batch_components.tsv`, `sweep_batch_eta.png`. Run automatically after a batch when the sweep ran.
3. **Primary wave analysis = band-passed motion projected out** (`waves.motion_orthogonal: true` by default; `--no-motion-orthogonal` for comparisons). The batch writes the primary to `noreg_motion_orth` (figures, 200 bootstrap / 200 surrogates) and the comparisons to `noreg` (no projection) and `reg` (24-term regression), 50 / 50 each. Without motion parameters the projection is skipped and the summary says so. Wave claims rest on the primary; the comparisons show what the motion coupling contributes.

### 2026-11-06: joint-ICA sweep across all 15 runs (`mejoint sweep-summary`, outputs of e67f906)

- **Labels:** "mixed" for nearly every component at every k in every run (exceptions: some R2*-dominant components in P01 and P17 at k >= 50, one in P15 at k = 90); every component above the energy null. As expected from Phase 4: the thermal-only eta bands are too narrow on real data, so the labels carry no information yet.
- **eta at k = 50:** median 0.42 across runs (range of run medians 0.35-0.47); tissue class 0.41, CSF class 0.44. Nearly the same in tissue and CSF, and in every run.
- **Partial volume:** the diagnostic judged all 750 components (50 per run) "not partial volume". "eta with the partial-volume prediction removed" rises to 0.92-1.00. Both say the dS0 points the other way from the partial-volume prediction (partial volume: dS0 in phase with -q; observed: largely in phase with q), consistent with the sign-opposed dS0 of the wave sign check. The "removed" value is therefore not a corrected eta; it only shows the mismatch.
- **Motion:** component time courses correlate at most 0.03-0.11 with FD (motion-regressed channels).
- **Dimensionality:** MAPCA AIC 15-66; the highest (66, P01) is also the run with the most motion (mean FD 0.46 mm) and the largest variance captured (58 %).
- **Stability** 0.987-1.000.
- **Concern (to test):** an eta of about 0.4 that is uniform across components, runs and tissue classes looks more like a calibration effect than physiology. eta is debiased with the THERMAL beta noise; if the c1 channel carries noise beyond thermal (physiological or interpolation-related), the dS0 energy is under-debiased and every component's eta drifts towards the value of that excess noise, weak components most. The wave sign check, which uses empirical noise, still found real (mostly sign-opposed) dS0 in the leading wave mode, so this would inflate, not create, the dS0.

### 2026-11-06: two checks of eta ~0.4 built (user: "implement both checks")

1. **eta vs component strength** (in `mejoint sweep-summary`): per run at the common k, Spearman rho between eta and the component's energy relative to the family-wise energy null, and the median eta of the 5 strongest and the 5 weakest components above the null; `sweep_batch_eta_strength.png` plots eta against strength for all runs (reported eta, and the empirical eta once available). If eta reflects noise left after the thermal debiasing, it rises as components weaken towards the null (rho < 0, weakest > strongest).
2. **eta with empirical noise** (`mejoint eta-recheck`, `scripts/eta_recheck_all.lsf`): post hoc on the saved sweep outputs (no new ICA). Per voxel and channel, the residual after all k time courses, its autocovariance up to 40 frames (30 s), and the exact beta variance under it, var(beta_j) = sum_l gamma(l) c_j(l). The residual has no power in the regressors' directions, so this under-states the betas' noise (about 10 % in a test with 3 slow regressors and AR(1) 0.7 noise); a per-channel calibration factor from noise simulated with the channel's median autocovariance (2000 rows) corrects it (to within about 2 %). Reported per run and k: noise level relative to thermal in c0 and c1 (1 = thermal), colour factor (beta variance with / without the autocorrelation), calibration factors, and per component eta as reported, with the empirical noise level only, and with level and autocorrelation. Written to `eta_recheck.json` in each k folder and tabulated by sweep-summary.
- Tests: pure dR2* components with unit white c0 noise and doubled AR(1) c1 noise: reported eta > 0.1 (spurious), empirical eta within 0.05 of 0, noise level c1 2.0, colour factor > 2 (3 seeds). On a simulated participant with thermal noise only: noise levels 1.00-1.08, empirical eta equal to the reported eta within 0.01.
- The empirical eta is a sensitivity analysis beside the reported eta; making it primary is the user's decision after seeing it.

### 2026-11-06: eta checks on all 15 runs; what is the dS0? (checks A-C built)

Results of the two eta checks (e67f906 outputs, k = 50):
- **Not a noise-correction effect.** eta does not depend on component strength (Spearman rho -0.48 to +0.64 across runs, no consistent sign; eta ~0.4 from 2x to 1000x the energy null). Empirical noise is at the thermal level (c0 0.97-1.13, c1 0.74-1.12; colour factor 0.88-1.00), and the empirical-noise eta is essentially the reported one (0.36-0.47 vs 0.35-0.47). The calibration hypothesis is rejected: the dS0 energy is in the maps.
- **Size:** eta 0.4 corresponds to a dS0 of about 30 ms x dR2* (3 % per s^-1). An echo-time offset cannot produce it (5 ms gives eta 0.02).

User hypothesis (2026-11-06): the S0 change may be neurovascular through changes in local blood volume, not only oxygenation. Our reading: a volume change of a compartment with another T2* and effective S0 (e.g. blood with inflow enhancement at TR 0.75 s) gives a TE-independent part (dS0) plus a part not proportional to TE (curvature), with S0 and R2* rising together when the compartment is brighter at TE 0 and decays faster (venous blood replacing tissue); a pure inflow change gives dS0 without curvature; a CSF-volume change the opposite sign.

Checks built (`mejoint te-check`, `src/mejoint/te_check.py`, `scripts/te_check_all.lsf`; post hoc on the sweep outputs and the front end's saved channels, no new ICA):
- **A. Sign and ratio:** voxels with both dS0 and dR2* beyond empirical noise vs a pure-dR2* null count; fraction with S0 and R2* changing together (the null fraction is high too, because the s and q estimates share noise, so the COUNT over the null is the informative part); ratio s/q in ms.
- **B. Curvature:** debiased energy in r0 (TE^2 outside span{1, TE}) and r1 (alternation) as fractions of the channel energy, z against noise (including the calibration's shared Monte Carlo error), and the voxelwise correlation of the curvature beta with the S0-channel beta. r0 is sign-fixed (R22 > 0; the front end leaves it to the QR routine).
- **C. Echo profiles:** the four channel betas mapped back to the echoes through the rebuilt projection (matches the front end's R to 1e-7), in each component's 200 strongest voxels, sign-aligned to dR2* > 0: median profile, fraction with TE1 and TE4 opposite, zero crossing, and (exploratory, 1 residual dof) the T2* of a compartment whose volume change fits best. Figure `te_profiles.png` per k folder.
- Tests (synthetic voxels through the real front end, 4 seeds): volume exchange of a 20 ms compartment at 2.5x tissue S0: curvature fraction 0.026, opposed > 0.9, TE1/TE4 opposite > 0.9, zero crossing between TE1 and TE4, best compartment T2* 15-25 ms. Single-compartment dR2*: curvature fraction 0.0000, no zero crossing, dS0 at the null. dR2* with a static 5 % blood compartment (partial volume): curvature detectable (z 12-22) but about 500x smaller than the exchange, no zero crossing. **Curvature is therefore not specific to a volume change; its size and the zero crossing are.** With thousands of voxels the curvature z is heavy-tailed (|z| up to 6.8 without curvature), so the fraction is the measure and z only supports it.

## Two-stage design (user, 2026-11-06)

### 2026-11-06: plan, return point, step 1 prepared

User's design: stage 1 = the current joint decomposition, unchanged (thermal whitening, stacked [c0 | c1], eta): it measures the physics and identifies noise. Stage 2 = a network-scale analysis on the data after stage-1 noise removal. Order: (0) return point; (1) diagnostic before building anything; (2) stage 2 only if approved, with fixed rules (two stages, no iteration; removal criteria in config before real data: motion correlation above a phase-randomized null, edge / CSF / vessel location by labelled proxies, residual-energy coupling; eta never a criterion on its own; one non-aggressive regression with motion in the same projector; stage-2 input = cleaned c0 z-scored per voxel, smoothing or Schaefer-400 + Ren parcels as config options; dimensionality MAPCA AIC + 20 / 30 / 50 with nulls through the same removal and df accounting; physics claims from stage 1 only; tedana denoised and plain OC through the identical stage 2; referees split-half reliability and parcel connectivity, QC-FC at batch); (3) simulation tests (broad network + focal vessel-like and edge-motion noise; recovery, and no removal of network variance under partial overlap); (4) CLAUDE.md edits (sections 7, 8, 11) proposed for approval first; waves unchanged.

**Step 0.** `git pull origin claude/beautiful-hamilton-141ubf` on the current branch (claude/waves) merged the project branch's one commit (the partial-volume wording fix, already identical here; no content change): merge commit 69428fd, pushed. Tag `pre-two-stage` at 69428fd: this session can push its branch but not tags (refused four times), so the tag is created and pushed from the cluster clone.

**Conflicts flagged:**
- The current branch is claude/waves, not the project branch (CLAUDE.md section 2). The two-stage work goes on claude/waves as instructed.
- CLAUDE.md section 9 says "no spatial smoothing before decomposition"; stage-2 smoothing needs a CLAUDE.md edit (it applies to stage 2 only; stage 1 stays unsmoothed).
- Parcels need registration (ANTs, T1 or a template to native space; the waves brief's ANTs path is for RHEL7 and unverified on this RHEL8 cluster) and the atlases (Schaefer-400, Ren 2019) are not downloaded or checked.
- Step 1 compares with tedana, which ran on unregressed OC data in its own mask (nilearn compute_epi_mask, which drops deep brain), with robustica 30 at its own AIC dimension; "current" (a) uses the 24-term motion-regressed front end. Matching is over the voxels both masks share; the no-regression front end can be run as a second pass.
- No tissue masks yet: edge (from the mask), CSF and vessel (from the data-only T2* / f_long classes) proxies, labelled as such.

**Step 1 prepared** (`scripts/diag_c0_variants.py`, `.lsf`; nothing added to the package): variants (a) `decompose.decompose` with the Phase 4 seeds (at k = AIC it reproduces the sweep's k folder), (b) and (c) the same 30-restart path on c0 alone (thermal; z-scored per voxel), each at k = MAPCA AIC and 25; matched to tedana by the Hungarian assignment on |spatial r| (c0-channel maps vs tedana's component maps; tedana file names searched, not assumed) and |temporal r| (vs tedana's mixing matrix); tedana accepted / Likely BOLD components reproduced (best spatial |r| >= 0.7); montages and maps per variant. Tested on a simulation with six planted sources standing in for tedana: matched spatial |r| 0.83-0.91, temporal 0.93-0.96, five of six recovered in every variant.

### 2026-11-06: two-stage work moves to its own branch (user decision)

User: commit the decomposition work and start a new branch that is not waves. Everything was already committed and pushed on claude/waves (9f30dc5). New branch **claude/two-stage** from that commit, so it carries all work to date: Phases 0-5, the wave extension (unchanged from here on), the batch pipeline, the eta and dS0 checks, and the step-1 diagnostic. Two-stage work happens only on claude/two-stage; claude/waves stays as it is; claude/beautiful-hamilton-141ubf is the original project branch (Phases 0-5). The return point is the branch **claude/pre-two-stage** at 69428fd, the state before the two-stage work (tags cannot be pushed from this session or from the cluster, so a branch that is never committed to stands in for the tag). CLAUDE.md section 2 still names the single project branch; updating it waits for the user's approval with the other CLAUDE.md edits.

### 2026-11-06: step 1 result on P04 (motion-regressed front end), review pending

tedana: 19 components (13 accepted, 6 tagged Likely BOLD); matched over 30,637 voxels in both masks.

| k | Variant | Variance in k dims | Spatial: median matched r, >= 0.7 | Temporal: median, >= 0.7 | tedana accepted / Likely BOLD reproduced |
|---|---|---|---|---|---|
| 28 (AIC) | (a) [c0 | c1], thermal | 39.8 % | 0.59, 4 of 19 | 0.72, 10 | 3 / 13, 1 / 6 |
| 28 | (b) c0, thermal | 49.4 % | 0.55, 5 | 0.66, 6 | 4 / 13, 1 / 6 |
| 28 | (c) c0, z-scored | 31.7 % | 0.63, 7 | 0.70, 10 | 4 / 13, 3 / 6 |
| 25 | (a) | 38.9 % | 0.59, 4 | 0.63, 9 | 3 / 13, 1 / 6 |
| 25 | (b) | 48.4 % | 0.58, 4 | 0.67, 7 | 3 / 13, 1 / 6 |
| 25 | (c) | 30.8 % | 0.61, 7 | 0.70, 9 | 5 / 13, 2 / 6 |

- All variants reproduce tedana only in part (about half of its components by time course at |r| >= 0.7, a quarter to a third by map). (c) is best by a small margin; weighting and k change little.
- The montages of (c) are dominated by edge rings, the frontal pole, lateral edges, an inferior occipital cluster and posterior-fossa speckle; bilateral grey-matter networks are not evident by eye. The display is limited (5 axial slices, unsmoothed maps thresholded at |z| > 2, array orientation).
- Not like-for-like yet: our variants used the 24-term motion-regressed front end, tedana did not regress; masks differ (ours includes deep brain and cerebellum); k differs (tedana 19); tedana's maps are z statistics, ours betas (temporal matching is free of the map-type difference).
- The matching tests agreement with tedana, not "conventional networks" (tedana's own components were not inspected for networks; no network templates in native space yet).

### 2026-11-06: two-stage approach dropped; scientific aim restated (user)

- **Two-stage approach dropped** (user): small expected benefit at the cost of the physics-based decomposition's advantages. Its work stays on claude/two-stage for reference; nothing from it enters the pipeline. Next work returns to the joint decomposition.
- **Aim (user, 2026-11-06):** participants are survivors of medulloblastoma, at least 5 years after diagnosis, surgery and adjuvant therapy, adolescents and adults. Kundu's group used ME-ICA to identify independent BOLD components in a data-driven way and reported that their number changes with age and development, with findings suggesting that this integration of networks is driven by cerebral-cerebellar interactions (user's summary; the papers are not checked here). Prediction: after childhood cerebellar injury, intrinsic activity reflects less integrated networks. Requirement: identify them in a principled, data-driven way, separating what is likely neuronally driven from noise. tedana's results here are mixed (5 of 16 runs with no "Likely BOLD" component).

### 2026-11-06: focus set; check 1 (is the echo-1 effect created by motion correction?) prepared

User (2026-11-06): survivors are predicted to show less integration, i.e. more independent networks in the cerebral cortex. No reference group yet (healthy age- and sex-matched participants later); clinical variables and reference anatomy exist but the network detection is to be developed from the single acquisition's physics alone; cortical localization is judged by eye. **Focus: get the decomposition right, so that components can be called likely neuronal or likely noise.** Work continues on a new branch, **claude/decomposition** (from claude/two-stage at its last commit, carrying all work and records).

Plan in order: (1) is the uniform echo-1 effect (eta ~0.4, curvature, zero crossing near 30 ms) created by processing? (2) calibrate the neuronal / noise bands on real data by injection (Phase 5); then the component call.

Check 1 (`scripts/te_check_raw_vs_mc.py`, `.lsf`; diagnostic only): the k = 50 components of each run's motion-corrected sweep (their time courses) are measured with the identical method (te_check.check, plain least-squares betas on the same time courses) in the motion-corrected front end and in the raw-echo front end (no motion correction, the same paired frames; from the noise survey). mbme_tedana estimates motion on echo 1 alone and applies it to every echo; if that creates the effect, it shrinks or vanishes in the raw echoes. The raw echoes carry uncorrected motion, so a persisting effect is informative and a vanishing one decisive. Test: identical data in both front ends give identical numbers (paired ratio 1.00).

### 2026-11-06: check 1 result (P04, P12; JSONs, the figure step had crashed on a missing value, fixed)

The same k = 50 components measured in the motion-corrected and the raw-echo front ends:
- Echo profiles of the strongest components are nearly unchanged (P04 comp 27: mc [2.1, 0.19, -0.38, -0.30] %, raw [2.02, 0.23, -0.32, -0.27] %; P12 comp 8: mc [1.92, 0.82, -0.86, -1.36], raw [1.93, 0.62, -1.10, -1.22]).
- Ratio unchanged (31.4 vs 32.4 ms; 33.5 vs 33.2 ms). Zero crossings more frequent in raw (26 -> 37; 31 -> 41); TE1/TE4 opposite more frequent (0.53 -> 0.66; 0.64 -> 0.72).
- **Motion correction does not create the effect.** Caveats: the raw front ends have no nuisance projection (rank 0 vs 25); raw curvature z is negative and many raw curvature fractions are missing (uncorrected motion inflates the empirical noise; a non-positive noise variance is now treated as missing). Only the profiles are a fair comparison.

### 2026-11-06: two findings about te-check (from the injection tests); one is a bug, fixed

1. **The profile zero crossing and the ratio near 32 ms are what noise gives.** te-check's profile is the median over a component's 200 strongest voxels (by |dR2*| z). The noise in the dS0 and dR2* estimates is positively correlated (r ~ 0.9, CLAUDE.md 5.2), so voxels selected on noise change sign between echoes at the noise-weighted mean TE: sum(W TE) / sum(W) with W = (exp(-TE/T2*) / r_e)^2 = 28.9, 33.8, 36.5 ms at T2* 30, 45, 60 ms. Pure-noise simulation (8 components, all below the null): zero crossing 34-36 ms, TE1/TE4 opposite in ~97 % of the selected voxels, ratio 36-39 ms, curvature ~0. The real components' zero crossings (25-33 ms) and ratios (31.5-34.6 ms) fall in this range. **So these two numbers cannot by themselves show a volume change.** What they do not explain: the size of the strongest real profiles (~2 % at TE1, about 10x the beta noise of a typical voxel at 800 frames). The injection runs measure the profile at the source's own voxels, which avoids the selection.
2. **Bug (fixed): the curvature z was biased upward whenever the nuisance projection was on.** The empirical noise calibration (`eta_recheck.beta_noise` / `calibrate`) simulated noise without the projection but applied the degrees-of-freedom correction for the removed regressors to it; the factor came out too small by ~p / (T - k - 1) and the noise was under-estimated (3.3 % at T = 800, k = 50, p = 25). Summed over ~10^5 voxels this gives a false curvature z of roughly 0.03 x sqrt(V / 2) ~ 7. Pure noise with projection: before, median curvature z 4.1 (T = 200) and 1.4 (T = 800, ~8,000 voxels); after, 0.3 and 0.5; without projection 0 both times. Fix: the simulated noise is projected with the same basis (`nuisance_Q.npy`); all callers pass it. Regression test added (`tests/test_te_check.py`). **Affected earlier results:** te-check curvature z and fractions (and the "What is the dS0?" table built from them) and eta-recheck (slightly). They need rerunning (`scripts/te_check_all.lsf`, `scripts/eta_recheck_all.lsf`, then `mejoint sweep-summary`). The main eta (thermal noise, sigma already corrected for the removed df in the front end) is not affected.

### 2026-11-06: injection calibration built (Phase 5; user approved)

`src/mejoint/inject.py`, CLI `inject-make`, `inject-recover`, `inject-summary`; `scripts/inject_run.lsf`, `scripts/submit_inject.sh`; settings `inject:` in configs/default.yaml.
- **Sources:** one dR2*-only, one dS0-only and one mixed source (dS0/S0 = 0.032 s x dR2*, same sign, so its profile crosses zero at 32 ms like the real components) in each of three data-only regions: cortical (centre 10-16 mm from the mask edge, upper two thirds), deep (>= 25 mm from the edge), inferior (inferior third, posterior half; approximate posterior fossa). Spheres 10 mm radius with a cos^2 taper, >= 80 % in the tissue or mixed classes, centres >= 30 mm apart. Band-limited (0.01-0.1 Hz) mutually orthogonal time courses. Two levels, separate runs: strong (dR2* SD 0.5 s^-1, dS0 1 %) and weak (0.2 s^-1, 0.4 %; about resting BOLD size).
- **Injection:** into the raw echoes, before any processing, at the raw volumes that form each analysis frame (volumes_used.tsv), as S + mu (exp(w (a - TE b) u) - 1): exact in the log, measured noise untouched; integer files keep their type, voxels outside the spheres stay bit-identical. JSON sidecars and DICOM folders are linked, not copied.
- **Pipeline:** mbme_tedana (same settings, so tedana also sees the sources), the mejoint front end (mc, motion regression), joint ICA at k = 50 with the original run's seeds.
- **Measured per source:** best component (|r| of time courses, after the nuisance projection), above null, label; amplitude recovered vs injected (dS0, dR2*) and their ratio in ms; eta of the component, of its footprint, and the oracle eta (true time course as a fixed regressor, no ICA: separates front end from ICA); te-check profile and curvature, plus the profile and curvature at the source's own voxels (no noise selection); tedana's best match (kappa, rho, class); whether the other components survive (vs the original k = 50).
- **Not testable by injection:** multiband slice leakage (the signal is added after reconstruction), coil combination, anything upstream of the magnitude images.
- **Tests** (`tests/test_inject.py`): time courses orthogonal and in band; injection exact in the log; file injection keeps pairing, type and untouched voxels; placement constraints; end to end on simulated echoes (sources found with |r| > 0.9; oracle eta 0.02 for dR2*, 1.00 at the footprint for dS0; ratio -1 ms for dR2* and 31.6 ms for the 32 ms mixed source; recovered amplitudes within 10 %); the run-folder path through make_dir, front end, decomposition and recover_dir. The job script was dry-run end to end with a stand-in for mbme_tedana (no FSL here).
- **Observation from the simulations:** a weak focal source that ICA does not isolate gets eta ~0.3-0.6 for its best component, while its oracle eta is correct. Low-energy components are not interpretable by eta, as CLAUDE.md 7.1 already says.

### 2026-11-07: injection result (P04, P12; strong and weak; 36 sources) and the fixed te-check

- **Not one source was found**, by our joint ICA at k = 50 or by tedana. Best component |r| median 0.18 (max 0.30), tedana's best 0.14 (max 0.25): chance level for slow time courses. tedana's "accepted / Likely BOLD" labels of those best matches are therefore meaningless. The injected signal is there: with the true time course the recovered amplitudes are 0.87-1.07 of the injected ones.
- **The front end measures the physics correctly** (true time course, the source's own voxels): ratio dS0/dR2* 1.4 and -0.7 ms for dR2*-only sources (truth 0), 31.9 and 33.6 ms for mixed (truth 32); curvature fraction -0.001 to 0.023 for every type. Oracle eta at the source: strong dR2* 0.05 (correct); weak dR2* 0.29, dS0 0.47-0.61 (truth 1), mixed 0.43: in ~100-voxel footprints chance correlation with the background pulls eta toward ~0.35. **Processing from the magnitude images onward does not create curvature or the 32 ms ratio.**
- **eta of any slow time course over the whole brain is ~0.33** (every source type, 0.24-0.44): chance correlation with the data's broad fluctuations outweighs a focal source over ~10^5 voxels, and those fluctuations have eta ~0.35. The native components' eta 0.41 is close to this background value.
- **Curvature survives the fix:** at k = 50 the native components' curvature fraction is 0.085-0.161 with z 18-65 in every run (sweep_batch te_fixed). With the injection result above, the curvature is in the images (upstream of the front end: reconstruction, coil combination, partial Fourier, or the signal: motion effects not linear in TE, a short-T2* compartment), not made by our processing.
- **Why nothing was found (to be measured):** in white noise a focal source is found once its channel variance summed over voxels exceeds ~3x sqrt(rows / T) (= 10 at 80,000 stacked rows, 800 frames; numerical check: |r| 0.93 at 30, 0.42 at 10). Real data add large structured signals that fill the top PCA dimensions, so a focal source can sit far below dimension 50. Added to the recovery: the source's summed variance, the white-noise limit, and the share of its time course inside the top 50 / 90 / 150 / 300 PCA dimensions of the data (chance k / T), with the k needed for 80 %. `scripts/inject_recover_all.sh` reruns the recovery on the saved outputs.

### 2026-11-07: network-scale injection built (user approved)

The focal test asked the decomposition to find 10 mm spheres; the aim is distributed networks. New layout (`inject.layout: network`, `inject.network` in the config; `--layout network` in `inject-make`, `inject_run.lsf layout=network`, `submit_inject.sh --layout network`):
- One network per type (dR2*-only, dS0-only, mixed at 32 ms), each 3 sphere pairs mirrored across the midline (left-right axis; partner within 12 mm of the mirror point) in the cortical region plus 1 sphere in the inferior region (approximate posterior fossa): 7 spheres of 10 mm, ~800 voxels, one shared time course. Every sphere of every network >= 25 mm from every other; spheres mostly in tissue or mixed classes.
- Same amplitudes (strong, weak), time courses, injection, pipeline and measures as the focal runs, including the detectability measures. Outputs in `outputs/inject/<run>/network_<level>_s<seed>/`; seeds differ from the focal runs (extra spawn key); the pooled summary separates the two layouts.
- Tests: placement on a brain-sized mask (counts, separation, mirrored pairs on opposite sides, no overlap); job script dry-run end to end with a stand-in for mbme_tedana.

### 2026-11-07: network injection result (3 of 4 runs; P04 strong missing) and what it means

- **Networks (7 spheres, ~860 voxels):** none recovered at |r| >= 0.9. The strong dR2*-only network in P12 came closest: |r| 0.75, recovered dR2* 0.90 of the truth, ratio 0.95 ms (truth 0), eta of its footprint 0.10; 74 % of its time course lies in the top 50 PCA dimensions and 80 % needs k = 58. Strong dS0-only and mixed networks: |r| 0.23 and 0.21; 80 % of their time courses needs k = 96 and 148. Weak (resting-size) networks: |r| 0.16-0.21, k for 80 % 171-313.
- **Focal sources:** 80 % of their time courses needs k 213-348 (the chance level for slow time courses is similar; see below).
- **The source's energy decides it, against the data's own large-scale structure, not the white-noise limit:** the strong dR2* network had 109 units (10x the white-noise limit), the strong dS0 network 65, the weak ones 21-43, all above the white-noise limit, yet only the strongest was partly found at k = 50. At equal % change a dS0 source carries less energy in the channels than a dR2* one, and the mixed source (its signal cancels near 32 ms) even less.
- **Oracle eta at the source is pulled toward the background:** with the true time course, strong dR2* network 0.13 (truth 0), dS0 network 0.59 (truth 1). A fixed slow regressor picks up energy from the data's own slow fluctuations in any footprint; the thermal debiasing does not remove it.
- **Fixes and additions (recovery only):** source energy was NaN when a footprint held a voxel without channel data (fixed); "chance" for the captured share was given as k / T (0.06 at k = 50, T = 800), but the data's top dimensions are slow, so slow time courses score far higher (the focal sources, whose energy is near zero, score 0.21-0.28 at k = 50 on real data); now measured with 50 never-injected time courses from the same generator. The same 50 give the background energy and eta in each footprint and an oracle eta with the background removed (simulation: 0.94 and 1.04 for dS0, -0.03 and 0.03 for dR2*). Recovery outputs can now be named per k (`--out-name`); `scripts/inject_k_sweep.lsf` / `submit_inject_k_sweep.sh` rerun only the decomposition of each saved injection at k = 90 and 150 and find the sources; the pooled summary lists every k.

### 2026-11-08: injection at k = 50, 90, 150 (focal and network; P04 strong network only at k = 50)

- **Higher k separates the strong networks** (best component |r|, k 50 / 90 / 150): strong dR2* network 0.75 / 0.91 / 0.92 (P12; P04 at k 50: 0.23); strong dS0 network 0.23 / 0.66 / 0.83; strong mixed network 0.21 / 0.26 / 0.70; weak dR2* network 0.18 / 0.22 / 0.67; weak dS0 and weak mixed networks and all focal sources not found (focal strong dR2* best 0.79 at k 150). This follows the "k for 80 %" measure: 58, 96, 148, 171 for those four.
- **With the background removed, eta at the source is right on real data** (true time course, the source's own voxels): dR2* networks -0.06 to 0.06 (truth 0), dS0 networks 0.80 to 1.00 (truth 1), focal dR2* -0.02 / 0.04, focal dS0 0.79 / 1.08, mixed 0.42-0.50 (the 32 ms mixed source's own eta is ~0.4). Without it: 0.12-0.14 for dR2*, 0.41-0.63 for dS0. The background a never-injected slow time course picks up has eta 0.23-0.43.
- **The eta of the recovered component is not**: the strong dR2* network found at k 90 / 150 has component eta 0.24 / 0.22 and is labelled "mixed" (truth 0), while its dR2* amplitude (0.98-1.01) and ratio (-0.1 / 0.6 ms) are right. A component's eta is computed over the whole brain (~40,000 voxels), where the energy any slow time course picks up from the data's own slow fluctuations is far larger than the source; its eta is that background's, ~0.25-0.45. **This is the likely reason every native component has eta ~0.42**: the reported eta measures the background, not the component. The existing nulls (phase-randomized single time courses, surrogate PCA directions) set the significance threshold and remove a selection bias; they do not remove this background from E_S0 and E_R2.
- **Bug fixed:** the echo profile and curvature at the source fitted the component and the true time course together; when the component matches the truth well, the two split the signal (in the figures the "component" and "oracle" profiles added up to the expected one). They are now fitted separately; test added.

### 2026-11-08: background-corrected eta; working k = 150 (user decisions)

- **k = 150 is the working k for now** (user): the injections show network-sized sources separating only around k = 150. Added to `decompose.k_sweep_fixed` (50, 70, 90, 150) and made the k of new injection runs (`inject.k`). Results at other k stay reported.
- **Background-corrected eta** (`src/mejoint/eta_background.py`, `mejoint eta-background`, `mejoint eta-background-summary`; `scripts/eta_background_run.lsf`, `scripts/submit_eta_background.sh`): for each component time course, 50 surrogates per method; their mean E_S0 and E_R2 (single-regression maps over the brain) are removed from the component's own before eta. Surrogates: **circular shifts** of the component's own time course by at least 30 s, the end wrapping to the beginning (user, 2026-11-08: a stronger control than phase scrambling in local field potential work; it keeps the exact spectrum and every within-series feature), and **phase randomization**, side by side. Two scopes: the whole brain (with a 5-95 % interval from single surrogates), and the component's **support** (voxels whose energy exceeds 3 noise SDs in one half of the run, energies and surrogates measured in the other half, both directions summed), because over the whole brain the background a given time course picks up varies by about as much as a network's energy.
- **Tests:** a pure-dR2* and a pure-dS0 source on a strong, broad, slow background of mixed physics (10 simulated datasets): thermal-only eta 0.20 +- 0.02 and 0.85 +- 0.01; corrected (circular shift, whole brain) 0.00 +- 0.04 and 1.04 +- 0.04; support version the same within 0.05. Circular shifts keep the time course's values (rotation) and phase surrogates its amplitude spectrum. A synthetic test that drew sources and background from one orthogonalized set was wrong (the sources then pick up no background); fixed.
- **The circular-shift p is descriptive, not a significance test:** a component's time course is fitted to the same data, so it beats shifted copies of itself even on pure noise (dry run: all six noise components p <= 0.02). Which components count stays with the pipeline null (surrogate data through the same selection, `above_null`); the summaries select components that way.
- **Also:** a single-k decomposition into an existing sweep folder now adds its row to sweep.json / sweep.md instead of replacing them. Recovery reads `eta_background.json` of the k folder and reports the corrected eta of the matched component next to the truth (oracle eta at the source without background).
- **Next:** validation on the injected networks at k = 150 (corrected eta near 0 for dR2*, 1 for dS0, ~0.4 for the 32 ms mixed source), then the native components of P04 and P12 at k = 50, 90, 150.

### 2026-11-08: route B planned; cerebellum excluded as an outcome region (user)

User: route B (region-level separation of BOLD-like and S0-like signal, BOLD-like connectivity and integration
measures) makes sense; work is on planning while the cluster connection is down. Cerebellar BOLD is not a focus:
anatomy, damage and radiation delivery vary too much to interpret it; those factors are estimated in separate efforts
and used as independent variables for their remote effects on the cerebral cortex. Draft plan for approval:
`docs/ROUTE_B_PLAN.md` (decisions D1 to D6 listed there). Nothing is built until it is approved.

### 2026-11-09: route B plan approved (user)

Decisions in `docs/ROUTE_B_PLAN.md` section 9, as recommended: D1 cortex as the outcome region, thalamus and striatum
as secondary regions (cerebellum and brainstem excluded); D2 Schaefer 200 parcels / 7 networks primary, 400 / 17 as
sensitivity (the injection test sets the minimum region size); D3 FSL for registration (default; ANTs / FreeSurfer
only if verified); D4 effective dimensionality (participation ratio) of cortical BOLD-like activity as the primary
integration outcome, segregation and the count above the circular-shift null secondary; D5 24-term motion regression
for the primary analysis, band-passed motion projection as sensitivity; D6 CLAUDE.md edits to be proposed for
approval. Work starts with the simulation tests (plan step 2), which need no cluster time.

### 2026-11-09: output folder; route B step 2 (simulation tests) built

- **Output folder (user):** future jobs write to `/research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/MEJOINT_QC/`. Earlier results are inputs to later jobs, so the repository's `outputs/` is moved there once and replaced by a link of the same name (`scripts/use_output_root.sh`); every script keeps using `outputs/...`. `.gitignore` also ignores the link. Job logs stay in the repository's `logs/`.
- **Route B core** (`src/mejoint/regions.py`): region averaging of the log-ratio echoes weighted by inverse thermal noise; the front end's split applied to the region signal (S0-like a(t), R2*-like b(t), two leftover channels); region noise from the leftover channels' frame-to-frame differences; shares in the 0.01-0.1 Hz band; coupling of b(t) with the noise removed from the variances; participation ratio, count of eigenvalues above the circular-shift null, segregation, global share; circular-shift null with each region shifted by its own random lag.
- **Tests** (`tests/test_regions.py`, 3 seeds): noise-free regions return a(t) and b(t) exactly; the region noise from the leftover channels matches the truth within 15 % for independent and for correlated voxel noise (a shared 30 % makes it 18x the independent-voxel value); BOLD-like share > 0.95 for pure R2*, < 0.05 for pure S0, within 0.05 of the truth for a mixed region; leftover share < 0.05 without curvature and > 0.1 with a curved echo profile; four driven networks give participation ratio > 2.5 against < 1.5 for one shared driver, four eigenvalues above the circular-shift null, segregation > 0.5; independent regions sit inside their null (about 5 % of edges beyond it); global share within 0.1 of the truth.
- **Observation:** an echo profile strong at echo 1 only (like several real components) is mostly absorbed as an S0 + R2* mixture at region level; only its non-linear part shows as leftover. Two test set-ups were wrong at first (time courses forced exactly uncorrelated by the generator; a first-echo-heavy profile used as the curvature case); fixed.

### 2026-11-09: route B step 3 (anatomy) built; T1 list received

- **T1 list (user):** 11 participants (P04 among them; P12 not listed, P08 listed but excluded). The T1 of a run is found by subject number and scan time in the list (`regions.t1_list`, or the list file given to the job), else by a search under `regions.t1_root`/<subject>/ for `*T1*MPRAGE*.nii*`.
- **Atlases:** Schaefer 2018 200 / 7 and 400 / 17 (MNI152 FSL 2 mm) and their label tables copied unchanged into `atlases/schaefer2018/` with the CBIG MIT license and source note (no download needed on the cluster). Harvard-Oxford subcortical (thalamus; striatum = caudate + putamen + accumbens) from FSL's own data folder, labels found by name in its XML.
- **Registration** (`scripts/anatomy_run.lsf`, `scripts/submit_anatomy.sh`; FSL 6): fslreorient2std, bet -R (f = `regions.bet_f`, 0.4), fast; epi_reg (boundary-based) of the front end's mean echo 1 within the analysis mask to the T1; flirt 12 dof then fnirt (T1_2_MNI152_2mm) to MNI; inverse warp; atlases into EPI space by nearest neighbour, grey-matter probability by trilinear interpolation. The functional data are not resampled. No field map: left-right distortion is not corrected. Untested here (no FSL in this session); every participant's registration is checked by eye (QC figures).
- **Region labels** (`src/mejoint/anatomy.py`, `mejoint anatomy`): cortex = Schaefer parcel, inside the analysis mask, FAST grey matter >= 0.3, not CSF-dominated or dropout in the data-only classes; then L / R thalamus and striatum (not CSF / dropout; FAST grey matter not applied in deep nuclei); regions under `regions.min_voxels` (20) flagged. Outputs per run: `labels_<atlas>.nii.gz`, `regions_<atlas>.tsv/.json`, `qc_<atlas>.png` (T1 white-matter edge and regions on the EPI), `qc_t1_to_mni.png`.
- **Tests** (`tests/test_anatomy.py`): T1 matching (list with quotes, folder search, none found); the bundled tables (200 / 400 regions, 7 / 17 networks); subcortical grouping (brainstem never a region); the label rules on a toy volume; run_dir outputs.

### 2026-11-09: CLAUDE.md edits approved and applied (D6)

User approved all ten proposed edits (`docs/CLAUDE_md_proposal.md`, E1-E10) and they are applied: title and goal (scientific aim; route B main, joint decomposition descriptive; cerebellum excluded as an outcome region); working branch `claude/decomposition`; output folder (MEJOINT_QC) and FSL 6; participants, T1 location and the built T1 option; architecture; section 7 marked descriptive with working k = 150 and the background-corrected eta; Phase 5 outcome and route B phases R1-R5 with gates (old Phase 6 replaced by R5; waves paused); region averaging is not smoothing; three new pitfalls; outcomes (primary: effective dimensionality of cortical BOLD-like activity).

### 2026-11-10: background-corrected eta fails validation; anatomy QC improved

**Corrected eta against known truth (injection, k = 50 / 90 / 150; P04 and P12).** Matched components with |r| >= 0.7:

| injected | k | n | eta reported | corrected, circular shift: brain / support | corrected, phase: brain / support | truth |
|---|---|---|---|---|---|---|
| dR2* network, strong | 150 | 2 | 0.22 | 0.21 / 0.17 | 0.22 / 0.18 | 0.05 |
| dR2* network, weak | 150 | 1 | 0.27 | 0.20 / 0.18 | 0.20 / 0.18 | -0.06 |
| dS0 network, strong | 150 | 2 | 0.48 | 0.43 / 0.44 | 0.43 / 0.48 | 0.88 |
| dR2* network, strong | 50 / 90 | 1 / 1 | 0.29 / 0.24 | 0.26 / 0.22, 0.22 / 0.18 | same | 0.05 / 0.06 |
| dR2* focal, strong | 150 | 1 | 0.39 | n/a / 0.60 | n/a / 0.92 | -0.02 |

- **The correction barely moves a component's eta, and the corrected values stay pulled toward ~0.2-0.45 whatever the truth** (the dS0 network reads 0.43-0.48 against 0.88). What the surrogates remove (chance overlap with the background) is not what biases a component. A component time course that matches the injected one at |r| ~0.9 still carries ~20 % of its variance from other, native signal, and its maps pick up that native structure (eta ~0.4) by genuine covariance, over the whole brain and inside its own voxels alike.
- **Conclusion: component-level eta cannot classify components as BOLD-like or S0-like in these data, with or without the correction.** The native components' corrected eta (k = 150: median 0.42, 10-13 of 300 below 0.2, 4-10 above 0.8) is therefore not interpretable as physiology. The joint decomposition stays descriptive (CLAUDE.md section 7), as already decided; no further work on component eta is proposed. Route B is unaffected: it uses no component time courses, and the region-level separation with the true timing gave the truth (0.05 for dR2*, 0.88 for dS0, background removed).
- Observation, not followed up: in both participants the native eta histograms have two peaks (~0.22 and ~0.47) at every k, reported and corrected alike.

**Anatomy (R2), first look (P04, P12).** T1 to MNI: the template outline follows the brain in both participants (the speckle outside the head was the outline method; it now draws the template's brain surface). Regions on the EPI: cortical parcels follow the cortex, thalamus and striatum sit in the deep grey matter, no regions in the cerebellum. EPI to T1: the old panel (white-matter edge at EPI resolution) could not be judged; replaced by `qc_epi_to_t1.png` (the EPI resampled into T1 space by epi_reg, and the T1, both with the T1 white-matter boundary at 1 mm). Rerunning `scripts/anatomy_run.lsf` reuses the registrations and only redoes the figures and labels. The R2 gate waits for that figure.

### 2026-11-10: anatomy (R2) results for P04 and P12

- Logs: P12's T1 was found by the folder search (same layout as P04's). 200 / 7: 186 (P04) and 195 (P12) of 204 regions usable (>= 20 voxels); 15,010 and 17,377 cortical voxels, median 72 and 78 per region. 400 / 17: 326 and 358 of 404 usable (median 35 and 41). Thalamus L / R 195 / 194 and 222 / 157 voxels; striatum 210 / 186 and 254 / 220.
- EPI to T1 (`qc_epi_to_t1.png`, white-matter boundary at 1 mm): brain outline, midline, ventricles and cerebellum align, the EPI's darker white matter fills the boundary in all three planes, no visible shift (assistant's reading; gyral detail cannot be judged at 3.3 mm). T1 to MNI: aligned.
- Regions below 20 voxels are the dropout zones: orbitofrontal and temporal-pole parcels in both (P12: L OFC 2, L TempPole 2, R OFC 1-3, R TempPole 1-3, R Default PFCdPFCm 3; P04 also L OFC 1, L TempPole 1 and 3, L / R Default Temp 1-2, L Cont Temp 1, R Cont PFCl 1, L Cont Cing 2). The Limbic network keeps 4 of 12 parcels in P12 and 2 of 12 in P04: treat it as unreliable in network-level measures.
- Usable regions differ between participants and the participation ratio depends on the number of regions: R3 / R4 report the integration measures on the regions usable in every participant (common set) and on each participant's full set.
