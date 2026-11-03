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

### 2026-11-02: LSF facts confirmed on the first Phase 0 job

- `rusage[mem=4000]` is read as MB and enforced as a hard limit (bhist: MEMLIMIT 3.9 G).
- `#BSUB -o` paths are relative to the submission folder; the job's working directory is the submission folder.
- Exported variables did not all reach the job (SUBJECT arrived, RUN_DIR did not). Cause not established: either the multi-line `export` broke when pasted, or LSF dropped values containing spaces. Job scripts therefore take values as arguments; batch jobs (Phase 6) will read paths from a subject list file, not from environment variables.
- Interactive sessions started from Open OnDemand are LSF jobs whose `LS_SUBCWD` is `/var/www/ood/apps/sys/dashboard`. Scripts must not trust `LS_SUBCWD` alone; job scripts locate the repository from their own path, then the current folder, then `LS_SUBCWD`, and check that `src/mejoint` is there.

### 2026-11-02: Phase 0 single-subject audit (run sub-41781_acq-20260421124817_ser-5)

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
- Coil elements differ from "HE1-4;NE1,2" in 4 runs (42267, 43565, 49236, 51497); values to be read from the summary. Fewer neck elements would lower posterior-fossa SNR and change the g-factor there.
- Unequal echo volume counts in 5 of 17 runs (not only 51497): 42267, 47591, 48246, 53374, 51497. Pairing kept 797, 751, 777 and 797 volumes; 51497 (271/278/267/277) failed in the pipeline. Where pairing drops volumes inside a run, the kept volumes are not consecutive in time; filtering, FD and phase-randomised nulls must use the acquisition timeline from `volumes_used.tsv`. The summary now reports these gaps.
- 5 of the 16 completed runs have 0 "Likely BOLD" components (42267, 47159, 48246, 48324, 53870). The summary now flags tedana's QC fallback (no provisional accept), which mbme_tedana records.
- No run kept motion-corrected echoes.

Gate status: summary shown to the user; awaiting review and approval.

### 2026-11-02: Phase 0 gate PASSED (user)

User decisions:
- Phase 0 passes. No phase, physio, noise frames or non-steady-state volumes in any run, so sections 5.4 and 9 stand as written (sigma from the residual with an MP-PCA cross-check; magnitude-only denoising if ever used).
- Subject 51497 is excluded (pairing failed; ~270 volumes). Recorded in `configs/default.yaml` under `subjects.exclude`.
- All work stays on the single branch `claude/beautiful-hamilton-141ubf` (replaces one branch per phase in CLAUDE.md section 2).
- CLAUDE.md sections 2-4 updated with the confirmed facts.
- Phase encoding: the user was unsure whether left-right was intended. The data are unambiguous (JSON `i`, DICOM `ROW`, 52-line axis = narrower 170 mm FOV), so analysis proceeds with left-right phase encoding as a fact; Phase 2 maps will show distortion and ghosting along that axis.

Open items carried forward:
- Motion-corrected echoes for 41781 (rerun with `--keep-intermediates` into `OUT_keep`) before Phase 2.
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

User: the simulation "seems reasonable"; proceed to real-data tests with what appears reasonable from the source materials and simulations, starting with subject 47159. Source amplitudes stay at SD 0.5 s^-1 / 0.5 % at the peak voxel.
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
- Real data next: subject 47159, raw echoes (`scripts/phase2_frontend_subject.lsf`).

### 2026-11-02: STOP: sigma cross-check fails on 47159 (raw echoes)

First real-data front end (sub-47159_acq-20260603110010_ser-5, raw echoes, 30866 mask voxels, 800 volumes):
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

User: 47159 has no large shunt or cavity; the central hole is "a regrettable tendency of the tedana workflow that's common across participants" (mbme_tedana builds the mask with nilearn `compute_epi_mask` on motion-corrected echo 1; it thresholds intensity and never fills holes).
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
Not yet validated on real data: the 47159 run with the new default will show both outlines (analysis orange, pipeline blue dashed). Real heads have orbits, sinuses, neck and a resection cavity in some patients; the thresholds may need tuning after inspection across subjects.

Independence: the front end runs from the raw echo files alone (`mejoint frontend --echo1 <..._e1.nii.gz>`; TEs from the JSON sidecars; equal echo volume counts required). Still from mbme_tedana: volume pairing (5 of 17 runs) and motion correction (needed from Phase 3/4 on). Open decision: proposed to import mbme_tedana's own pairing and motion-correction functions as a library (a mejoint `preprocess` stage that runs them without tedana), rather than reimplementing them (CLAUDE.md section 4).

### 2026-11-03: 47159 results with the diagnostics; per-echo noise estimator; mask v2

Reproducibility: the `--keep-intermediates` rerun of 47159 gave "0 likely-BOLD components (of 19 total; 13 accepted)", the same count and component number as the original run.

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

### 2026-11-03: 47159 motion-corrected echoes (front end before the per-echo estimator)

- Residual energy still tracks head drift after motion correction (Spearman 0.82; raw 0.92). Not only misregistration: when the head moves, the B0 inhomogeneity near air/bone moves relative to the brain, changing through-slice dephasing and so the curvature of the decay (seen only by the residual). Added: per-echo noise ratios are also computed on the half of frames nearest the median head position (`low_drift`), to compare with all frames.
- Residual sigma median 312 (raw 324): single spline resampling lowers the noise by ~4 %.
- Voxel-echoes with frames <= 0 rose from [75, 194, 131, 62] (raw) to [164, 565, 639, 705] (spline undershoot); `nonpositive_policy: exclude_echo` handles them.
- Air-noise ratios changed after resampling (SD 1, 0.65, 0.56, 0.50 vs raw 1, 0.57, 0.48, 0.47): the background is not a usable thermal reference in this data.
- Mask (v1) 1715 ml: same over-inclusion as raw; mask v2 pending real-data check.
- Test note: on the small simulated head (T2* 20-70 ms, skull base 22 ms) mask v2 keeps ~90 % of the simulated brain; the T2* > 25 ms limit trims short-T2* edge brain. Real-data contact sheets decide the limit.

### 2026-11-03: per-echo noise on 47159 (motion-corrected); mask v3; 40240 excluded

**Per-echo noise from the residual (47159, mc echoes, 20000 voxels):** all frames [1, 0.86, 0.61, 0.77]; voxel halves [1, 0.77, 0.58, 0.68] / [1, 0.96, 0.64, 0.86]; frames nearest the median head position [1, 0.84, 0.63, 0.70]. Echo 1 is the noisiest by all three methods (MP-PCA, air, residual), so CLAUDE.md 5.3's equal-sigma assumption is not supported in this data. The size is uncertain: the halves differ by up to 0.2 (~0.05 in simulation) and echo 4 > echo 3 is physically odd. Likely cause: the real residual is not pure noise (it tracks head drift after motion correction; the mean decay is slightly concave brain-wide, median ln(mu/fit) per echo [-0.004, 0.011, 0.005, -0.017], as through-slice dephasing produces). Decision on the noise model pending (user).

**Mask v2 across 16 runs (mask QC):** volumes 1043-1600 ml (median 1292; pipeline median 1335). User: the outline variably excludes the edges of the cerebellum and orbitofrontal cortex, "important regions". Cause: the 25 ms decay limit also removes brain whose late echoes are lost to dephasing. Against the pipeline mask v2 lost a median 88 ml (max 165) at the edges and gained 64 ml (max 205) in the deep brain.

**Mask v3:** the decay limit now only decides what is connected to the brain; the core then regrows up to 4 voxels (~13 mm) into adjacent tissue bright at echo 1, whatever its decay. Phantom with a dephased lateral-inferior rim (T2* 15 ms) and an anatomical neck: rim kept 33 % (v2) -> 96 % (v3); neck muscle 1.4 % -> 8.9 % (a band around the cord); scalp 2 %; ventricles 100 %. 3 voxels: rim 83 %, muscle 6 %; 5 voxels: rim 100 %, muscle 12 %. Tissue with almost no signal even at echo 1 (severe dropout) stays outside: an EPI-only mask cannot recover it; an anatomical (T1-based) mask could.

**40240 excluded** (user): metal artefact or similar in the right cerebral hemisphere (also visible as an abnormal mask, 844 ml). 15 runs remain.

### 2026-11-03: stay on the gated route; mask frozen; noise survey next (user)

User: continue on the gated route; avoid getting sidetracked perfecting the mask.
- Mask v3 is frozen as the Phase 2 analysis mask. Any refinement (e.g. a T1-based mask) is deferred to Phase 5, which needs tissue masks anyway.
- Remaining Phase 2 blocker: the noise model. Next step: front end on the raw echoes of all 15 runs (`scripts/noise_survey_all.lsf`, no MP-PCA) and a table of per-echo noise ratios (`noise_survey.md`). Consistent ratios across runs would make the echo dependence a protocol/reconstruction property that can be set once.
- A `mejoint preprocess` step (mbme_tedana's pairing and motion correction without tedana) is deferred to Phase 6; Phases 3-5 use 47159, which already has motion-corrected echoes.
- Mask v3 contact sheet (15 runs, 2026-11-03): cerebellar hemispheres and orbitofrontal cortex inside the mask in every run; deep brain included. Volumes 1210-1789 ml (pipeline 963-1675). Small lateral islands in the lowest slice in a few runs (47527, 50666, 53870 at z = 3): inferior temporal/cerebellar tissue or some skull-base non-brain tissue; accepted for Phase 2, revisit with tissue masks at Phase 5.

### 2026-11-03: traveling waves on hold (user)

User added `docs/WAVES_BRIEF.md` and `docs/mockups/waves/`, and asked to hold all wave work until the decomposition of real data is reached. Nothing wave-related is built or branched before then.

### 2026-11-03: overflow warnings in the mono-exponential fit (noise survey log)

Cause: voxels with only one usable echo (< 1 % of the mask in 47159). Their Gauss-Newton system is singular; rounding left a tiny positive determinant and a runaway step. Their T2*/S0 were already NaN, and every voxel is fitted independently, so other voxels were unaffected (test: exact T2* in all well-posed voxels). The channels, sigma and per-echo noise ratios never use this fit, so the noise survey is unaffected.
Fix: fit only voxels with >= 2 usable echoes, skip ill-conditioned steps, bound the steps and ln S0. Regression test with one-echo, rising, flat, very fast and non-monotonic decays runs with warnings treated as errors.
Correction: T2* values near the 500 ms cap (e.g. a 90th percentile of 219 ms in the inferior third) come from rising or flat decays (CSF, inflow), not from this bug.

### 2026-11-03: noise survey (15 runs) and per-echo weighting adopted (user: option A)

Survey (raw echoes, residual-based per-echo ratios, frames nearest the median head position): echo 1 noisiest in 14 of 15 runs; median [1, 0.761, 0.671, 0.602], interquartile range [1, 0.72-0.82, 0.61-0.74, 0.52-0.66]; the 6 runs whose voxel halves agree within 0.15 give the same median ([1, 0.76, 0.63, 0.61]); MP-PCA on 47159 gave [1, 0.74, 0.60, 0.51]. Per-run estimates are unreliable (halves disagree by > 0.15 in 9 of 15 runs; residual tracks head drift), so one study-wide vector is used. Cause not established (unverified candidates: echo-1-specific reconstruction, a signal-proportional fluctuation largest at echo 1).

Decision (user, 2026-11-03): **option A**. `frontend.weighting: per_echo`, `frontend.echo_noise_ratios: [1.0, 0.761, 0.671, 0.602]`. Implementation: the noise-scaled mean mu_e / r_e replaces mu_e wherever the equal-sigma construction used it as the noise scale (sigma-free residual, trusted-voxel test, echo SNR and usability, whitening); the mono-exponential fit weights residuals by 1 / r_e; the MP-PCA check compares per echo with sigma_v r_e. `weighting: equal` is kept for the Phase 4 tedana-equivalence test and as a sensitivity analysis of every main result. CLAUDE.md 5.3 and 5.4 updated.

Evidence (simulation with these ratios, 28 x 28 x 16, 400 volumes): whitened channel/residual covariance within 0.9 % of I with per-echo weights; up to 65 % off with equal weights. Tests now simulate the configured ratios by default; properties defined under equal noise (test (c), the sigma bootstrap) run with equal data and equal weights. The floor case of test (e) was recalibrated from a six-seed spread (excess 1.080-1.105 included vs 1.023-1.044 excluded). 53 tests pass under six seeds.

Phase 2 gate still needs: 47159 rerun with per-echo weights (raw with the per-echo MP-PCA check; mc for the maps).

### 2026-11-03: Phase 2 gate run on 47159 with per-echo weights: MP-PCA check fails a second time (STOP)

Motion-corrected run: mask 1424 ml (40077 voxels); echo 4 usable in 79 % of voxels (36 % of the inferior third); residual excess 0.97 (0.94 inferior); sigma_v (echo-1 noise) median 439. This run's own per-echo estimate [1, 0.69, 0.54, 0.53] is lower than the study-wide ratios and its voxel halves agree within ~0.1; it is close to MP-PCA's ratios for 47159 ([1, 0.74, 0.60, 0.51]).
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

### 2026-11-03: differenced diagnostic on 47159: mixed result against the agreed rule (STOP, proposal to the user)

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
(c) Rerun 47159 (raw with MP-PCA, mc for maps). Rule fixed in advance: all four echoes within 20 % -> the MP-PCA check passes. Otherwise stop: no further tuning of the noise model to pass this check; the user decides whether the remaining gap is an accepted limitation.
Infrastructure added (no change to the analysis defaults): `frontend-all --force` / `noise_survey_all.lsf <OUT> force`; survey columns for differenced ratios, their halves and differenced / residual sigma; `sigma_plain.nii.gz` map.

### 2026-11-03: differenced sigma estimator and new study-wide ratios adopted (user approved (a)-(c))

Noise survey rerun (15 runs, raw echoes, all with `force`): differenced per-echo ratios median **[1, 0.721, 0.631, 0.560]**, interquartile range [1, 0.678, 0.608, 0.529] to [1, 0.768, 0.669, 0.584]. Voxel halves agree within 0.03 in 10 of 15 runs (41781, 43565, 50666, 53374 and 53870 are 0.05-0.17 apart). Non-differenced estimates in the same rerun remain unstable (e.g. 43565 [1, 1.37, 1.00, 1.15], 49236 [1, 1.38, 0.66, 1.04]). Differenced / residual sigma brain median 0.87-0.98 across runs (49236 lowest). The low-drift median moved from [1, 0.761, 0.671, 0.602] (first survey) to [1, 0.721, 0.617, 0.558] (this rerun); the voxel selection changed with the per-echo weights, another sign that the non-differenced estimate is sensitive to which voxels are used.

Decision (user, 2026-11-03):
(a) `frontend.sigma_estimator: differenced` (CLAUDE.md 5.4 step 4 updated); `residual` kept as a sensitivity analysis.
(b) `frontend.echo_noise_ratios: [1.0, 0.721, 0.631, 0.560]` (CLAUDE.md 5.3 updated).
(c) Rerun 47159 (raw with MP-PCA; mc for the maps). Rule fixed in advance: all four echoes within 20 % of MP-PCA (after the simulated 2 % bias) -> the MP-PCA check passes; otherwise stop, no further tuning of the noise model; the user decides whether the remaining gap is an accepted limitation.

Standard error of the differenced estimator: 1.103x (df 1) and 1.124x (df 2) the non-differenced one (simulation, white noise; unbiased); `sigma_rel_se(..., difference=True)` and test (d) use it. 57 tests pass under seeds 0, 1 and 2.

### 2026-11-03: MP-PCA check split into level and shape (user decision, before the 47159 rerun results)

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

### 2026-11-03: Phase 2 gate run on 47159 (raw, differenced estimator, ratios [1, 0.721, 0.631, 0.560]): shape passes, level misses by 0.004 (STOP, user decides)

MP-PCA check, level/shape rule:
- **Shape:** [1, 0.952, 1.024, 1.072], within the 10 % limit. If MP-PCA's echo ratios were the truth, channel noise variances would be off by up to 4.2 % and the c0-c1 noise correlation would be +0.057.
- **Level:** 1.204 after the simulated bias, against a limit of 1.20. Raw ratio 1.228; quiet window 1.218. Per-voxel differenced sigma alone (without the smooth-map fill in 30 % of voxels) gives 1.196 raw, 1.173 after the bias. The rule uses the sigma actually used for whitening, so the verdict is STOP.

Why the level rose from the predicted ~1.12: sigma_v (echo-1 noise) depends on the echo ratios through the residual basis. Lowering r for echoes 2-4 raised the differenced sigma_v from 397 to 422 (+6 %). The earlier rescaling had assumed this coupling was weak; it is not.

This run's own differenced ratios [1, 0.719, 0.603, 0.546] (halves within 0.02) agree with the study-wide vector within 0.03.

Non-differenced residual energy against the differenced sigma is now 1.18. This is expected: 1 / 0.92^2, the slow violations. It no longer means a model failure.

As agreed in advance, no further tuning of the noise model. The user decides whether the level gap is an accepted limitation.

### 2026-11-03: level gap accepted as a documented limitation (user: option A); Phase 2 gate evidence

Decision (user): the MP-PCA level gap (1.204 against a limit of 1.20; shape passes) is accepted as a limitation and is not treated as a model failure. It is reported as "thermal sigma level uncertain within about 20 % between two independent estimators (residual, MP-PCA)". The `snr_min` = 3 and `sigma_estimator: residual` sensitivity runs bracket its effect. Later, a spatial-noise-correlation measurement and an MP-PCA simulation will be added as a reported diagnostic only, not as a gate. They test whether MP-PCA reads low on correlated MB-6/GRAPPA noise and will inform which sigma level to trust for eta's noise correction.

Gate evidence, 47159 (raw and mc, differenced estimator, ratios [1, 0.721, 0.631, 0.560]):
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
- **Gaps** (e.g. 48246): segments come from volumes_used.tsv; filtering, frame-to-frame changes and FD never cross a gap; each segment gets its own constant.
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

### 2026-11-03: Phase 2b on 47159 (mc, 24 regressors): slow violations removed; projector removes half of c0's energy (to review)

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
- The colored-noise and projected-data cases are marked as known failures (strict xfail) until fixed.

**Gate not passed.** The overnight run on 47159 is exploratory: front end with the surrogate check, then all rank estimates and the real-data spectra. The spectra are required for the gate anyway and will show which estimator to fix.

Candidate fixes for the next session (not yet tested): AR(p) surrogates per row instead of phase randomization; deflation with the null's own top directions matched; or a noise reference from the residual channels after rescaling by their measured coloured spectrum.
