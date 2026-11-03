# CLAUDE.md: physics-constrained joint decomposition of multi-echo fMRI

This file is the standing brief for this repository. Read it fully before writing code, and re-read the relevant section before each phase. The literature review behind it is `docs/research_report.md`. This file turns that review into decisions. If you find a concrete error here, stop and tell the user rather than silently deviating.

---

## 1. Goal

Build a Python package that decomposes 4-echo multiband resting-state fMRI into components that each carry **two physical maps**: a ΔS0 map (TE-independent, non-BOLD-like) and a ΔR2* map (TE-dependent, BOLD-like). Both maps must be estimated **during** the decomposition, not by classifying ICA components afterwards as tedana/ME-ICA do with kappa/rho.

Deliverables:
1. A tested front end that turns the echoes into two noise-whitened physical channels plus a residual noise reference.
2. A calibrated dimensionality estimate.
3. A joint decomposition with robust restarts.
4. Per-component physical metrics with null distributions.
5. A head-to-head comparison with tedana on the same inputs.
6. An LSF batch interface.

Stretch goal: a structured factorization with group-sparse channel maps.

## 2. The user and how to work with them

- **User:** Stu McAfee, PhD (St. Jude, Diagnostic Imaging). He is expert in fMRI physiology and cerebellar/posterior fossa neuroscience.
- **Python level:** novice to intermediate. Explain code by what it does, not by language mechanics.
- **Writing:** keep scientific writing concise and defensible. Avoid em dashes in prose written for him.
- **Git:** all phases are developed on the single branch `claude/beautiful-hamilton-141ubf` (user decision, 2026-11-02; this replaces one branch per phase). Never commit directly to `main`; merge into `main` only when the user agrees. Whenever you create, switch or merge a branch, give the exact commands and one line on why:
  ```bash
  git pull origin claude/beautiful-hamilton-141ubf     # get the latest work on the project branch
  git add -A && git commit -m "..."                    # save progress
  git switch main && git merge claude/beautiful-hamilton-141ubf   # only when the user agrees
  ```
- **Stop and ask, do not guess,** when any of these happens:
  - an acquisition fact (section 4) is unknown and changes the design;
  - a gate (section 8) fails twice;
  - a choice would change the scientific claim (for example dropping an echo or changing the outcome);
  - results look too good.
- **Never fabricate** citations, numbers or tool behaviour. Check library APIs in the installed version (`help()`, source, `--help`). Label anything unverified.

## 3. Environment (HPC facts the user has confirmed)

- **Cluster and scheduler:** RHEL8 with **LSF** (`bsub`, `bjobs`, `bqueues`), not SLURM.
- **Queues:**
  - `standard`: CPU.
  - `gpu_short`: 30-minute limit.
  - `gpu_priority`: longer GPU jobs.
  - `gpu`: long backlog.
  - `gpu_interactive`.
- **Batch shells:** `bsub` shells do not define `module`. Source the modules init file first: `/usr/share/Modules/init/bash` on the login node, and `/etc/profile.d/modules.sh` or `/cm/local/apps/environment-modules/*/init/bash` on GPU nodes.
- **Job flags:**
  - `bsub -env` must be `"all, VAR=x"`, or the job gets a stripped environment.
  - `rusage[mem=]` is **per core**.
  - Multi-core jobs need `span[hosts=1]`.
  - Inside a job, use `$LSB_DJOB_NUMPROC` for worker counts, not `os.cpu_count()`.
- **Python:** conda lives in `~/miniconda3_new`. Project env: `~/miniconda3_new/envs/mejoint` (a clone of the tedana env: tedana 26.0.3, torch 2.6.0+cu124, dipy, pytest, pyyaml); always call `~/miniconda3_new/envs/mejoint/bin/python` by full path. Keep `~/miniconda3_new/envs/tedana` unchanged for the existing analysis. Job scripts use the code in `src/` via `PYTHONPATH` (no pip install).
- **Interactive sessions** started from Open OnDemand are LSF jobs whose `LS_SUBCWD` is the dashboard folder; scripts must not rely on it to find the repository.
- **LSF memory:** `rusage[mem=]` is in MB and enforced as a hard limit. Exported variables did not reliably reach jobs; pass paths as arguments or in list files. The cluster `python/3.11.0` module, which loads with `fsl/6`, shadows conda's python. Always call the env's python by full path in job scripts.
- **FSL:** `fsl/6` does not set `FSLOUTPUTTYPE`; export it in scripts.
- **GPU:** the driver is CUDA 12.4. Use a cu124 torch build; `torch==2.6.0+cu124` is known to work.
- **Packages:** install them on the login node.

## 4. Data and existing pipeline

**Known acquisition facts:**
- Resting-state multiband multi-echo EPI with 4 echoes.
- **TE = 16.20, 31.22, 46.24, 61.26 ms.**
- TR ≈ 0.75 s, so Nyquist is 0.667 Hz.
- 17 runs (one per subject) were audited in Phase 0. Most have 800 volumes per echo. Echo volume counts differ in 5 runs; the pipeline's data-driven pairing kept 797, 751, 777 and 797 volumes in four of them (run P10 has a 19-volume gap inside the run, so kept volumes are not always consecutive in time). **Subject P08** (267–278 volumes per echo) failed pairing in the pipeline and is **excluded** (user decision, 2026-11-02). **Subject P02** is **excluded** (metal artefact or similar in the right cerebral hemisphere; user decision, 2026-11-03). 15 runs remain.
- 52×64×48 matrix, **3.281 × 3.281 × 3.3 mm** voxels (no slice gap). 3T Siemens MAGNETOM Prisma, syngo MR XA60, CMRR `cmrr_mbep2d_bold`. DICOMs are enhanced (XA) multi-frame files.
- **MB 6** (derived from SliceTiming: 48 slices, 8 distinct times; `MultibandAccelerationFactor` is absent from the JSON). Slices acquired together are 8 slices (26.4 mm) apart. GRAPPA 2, partial Fourier 7/8, effective echo spacing 0.31 ms, total readout 15.8 ms.
- **Phase encoding is left-right** (`PhaseEncodingDirection` = i, DICOM `ROW`; the 52-line axis has the narrower 170 mm FOV). Distortion, ghosting and any CAIPI shift run left-right.
- HeadNeck_20 coil, elements HE1-4;NE1,2 (HE1-4 only, without neck elements, in P12, P06, P09 and P08); sum-of-squares coil combination, so low-SNR magnitude noise is noncentral-χ.
- Participants are children and young adults, including patients after posterior fossa tumor surgery (resection cavities, hemosiderin, clips).

**Established in Phase 0 (all 17 runs; details in `docs/decisions.md`):**
- Phase: **not saved**.
- Noise frames: **none**. Non-steady-state volumes: **none** (first-volume / median signal ratio about 1.00).
- Physiological recordings: **none recorded** (user). No RETROICOR/RVT; respiratory and cardiac effects must be handled from the data and reported as a limitation.
- Sequence fields: see the acquisition bullets above.
- Slice-stack spacing: 8 slices (26.4 mm). **CAIPI shift unknown** (no protocol PDF; not found in the DICOM sample).
- Tissue masks (grey matter, CSF/ventricles, vessels/sinuses) and cavity/peri-cavity masks: **no source exists** (user). They must be derived before Phase 5.

**Analysis mask and independence (user decisions, 2026-11-02):**
- The analysis always uses its own hole-filled brain mask, built from the input echoes by `mask.py` (head vs air from the measured noise level, coil-shading correction, brain vs scalp with a brain-like echo-1/echo-4 decay to exclude neck and face muscle, every enclosed hole filled, then regrowth into adjacent bright tissue so dephased cerebellar and orbitofrontal edges are kept). The user requires the cerebellum and orbitofrontal cortex to be included (2026-11-03). mbme_tedana's `brain_mask.nii.gz` (nilearn `compute_epi_mask`, which never fills holes) drops the deep brain in many participants; use it only to restrict comparisons with tedana to the voxels tedana saw.
- The analysis should run without tedana intermediates or outputs. The front end already runs from the raw echo files alone (`--echo1`, equal echo volume counts). Volume pairing and motion correction still come from mbme_tedana; how to remove that dependence is an open decision (see `docs/decisions.md`).

DICOM headers sit in the NIfTI folder or one level deeper.

**Existing pipeline: `mbme_tedana.py`** at `/research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/Monitoring Visit/mbme_tedana.py`, outputs in `.../Monitoring Visit/OUT/` (paths contain a space; always quote them). What it does:
- Finds the echoes from an echo-1 list.
- Pairs volumes across echoes when the echo counts differ, and writes `volumes_used.tsv`.
- Motion-corrects: mcflirt on echo 1, then one `flirt -applyxfm` resampling with the same transforms on every echo (spline interpolation by default; read it from `result.json` parameters). Motion parameters are written to `<run>/motion/mcflirt.par`.
- Builds a brain mask, runs tedana 26.0.3 and counts "Likely BOLD" components.
- Has a `--focal` option for focal kappa/rho.
- Keeps the motion-corrected echoes only with `--keep-intermediates`, at `<run>/work/echo-k_desc-mc_bold.nii.gz`.
- Writes tedana outputs to `<run>/tedana/`.
- The existing runs used the script's defaults `--ica-method robustica --n-robust-runs 30 --tedpca aic --tree tedana_orig --seed 42 --interp spline` (robustica, not tedana's own fastica default).
- TEs were passed with `--echo-times`: the script cannot read TEs from these enhanced DICOMs (JSON agrees).
- No run kept its motion-corrected echoes. A rerun of P01 with `--keep-intermediates` into `.../Monitoring Visit/OUT_keep/` provides them.
- 5 of the 16 completed runs hit tedana's "no provisional accept" fallback and report 0 "Likely BOLD" components (P12, P04, P10, P03, P07).

**Reuse it for preprocessing and as the tedana baseline. Do not reimplement volume pairing or motion correction.** Run tedana by full path from the pinned environment, and confirm flag spellings with `tedana --help`. The 26.0.3 source uses `--ica-method` and `--n-robust-runs`; some docs show underscores.

## 5. Core physics, front end, and key numbers

Label: everything in this section is **our own derivation, checked numerically twice** (it is not from the literature).

### 5.1 Signal model

For each voxel, echo e and volume t, with temporal-mean signal μ_e:

```
y_e(t) = ln(S_e(t)/μ_e) ≈ a(t) − TE_e·b(t) + ε_e(t)
a = Δln S0 (≈ ΔS0/S0, TE-independent), b = ΔR2* (TE-dependent),  Var(ε_e) ≈ σ_e²/μ_e²
```

- Use the log-ratio form. It is exactly linear in a and b.
- The linear ΔS/S form errs by about 3% at TE4 for ΔR2* = 1 s⁻¹.
- The linear form errs by about 9% for a 20% S0 spike (ln 1.2 vs 0.2), at any TE.

### 5.2 The cost of separating a and b voxel by voxel

With WLS weights μ_e²/σ² and T2* between 20 and 70 ms:
- The ΔR2* noise SD inflates **2.2–2.6×** compared with assuming ΔS0 = 0. That is 79–85% of the information.
- The â and b̂ estimates are correlated at r ≈ 0.89–0.92.
- Without the 61 ms echo, the inflation is 2.8–3.2× (against the 4-echo, ΔS0 = 0 baseline).

What follows from this:
- **Never decompose raw a and b maps.**
- **Never analyse the ΔR2* channel alone as if it were independent.**
- **Separate at the component level,** where each time course pools thousands of voxels.
- **Keep all 4 echoes.** Children's longer T2* (about 58 ms) puts more BOLD weight on the late echoes.

### 5.3 Front-end construction

**Per-echo noise (user decision, 2026-11-03, option A).** The noise survey (15 runs) found echo 1 noisiest in 14 of 15 runs, so σ is **not** equal across echoes in this protocol. The front end uses σ_ve = σ_v·r_e with one study-wide ratio vector **r = [1, 0.721, 0.631, 0.560]** (`frontend.echo_noise_ratios`; median over the 15 runs of the differenced-residual estimates, see 5.4 step 4; echo 1 = 1; user decision 2026-11-03, replacing the low-drift estimate [1, 0.761, 0.671, 0.602]). Everywhere below, replace the mean μ_e used as the noise scale by the noise-scaled mean μ_e/r_e: `Xw = (m / (sigma * r))[:, None] * D`, `yw = (m / (sigma * r))[:, None] * y`, and the σ-free residual basis is built from (μ/r)∘D. σ_v is then the echo-1 noise, estimated exactly as in 5.4. `frontend.weighting: equal` restores the original equal weights; use it for the Phase 4 tedana-equivalence test (property 1 below holds only with equal weights) and as a sensitivity analysis of every main result. On simulated data with these ratios, per-echo weights whiten to within 1 % of I; equal weights miss by up to 65 %.

Batched over voxels, written here for equal σ across echoes (with per-echo noise, use m / r as above):

```python
import numpy as np
# per voxel: m = mu (E,), sigma = scalar thermal SD, TE (E,) in ms, y = log-ratio data (E x T)
D = np.column_stack([-TE, np.ones(E), TE**2, (-1.0)**np.arange(E)])   # R2*, S0, curvature, alternating
Xw = (m / sigma)[:, None] * D
Q, R = np.linalg.qr(Xw, mode="complete")      # batch: arrays shaped (V, E, 4) work directly
sgn = np.sign(np.diagonal(R[:2, :2])); Q[:, :2] *= sgn; R[:2, :] *= sgn[:, None]   # diag(R) > 0 in EVERY voxel
yw = (m / sigma)[:, None] * y                 # whitened data (E x T)
c  = Q[:, :2].T @ yw                          # c[0]: OC-like channel, c[1]: depends ONLY on dS0
r  = Q[:, 2:E].T @ yw                         # residual: r[0] ~ curvature, r[1] ~ alternating/ghost
R00, R01, R11 = R[0, 0], R[0, 1], R[1, 1]     # R01 < 0 after the sign convention
```

How to use and read it:
- Apply the same sign convention in every voxel. Otherwise the spatial maps get sign-scrambled before stacking.
- With 3 usable echoes, use the columns [−TE, 1, TE²]: one residual channel.
- With 2 usable echoes there is no residual.
- With 1 usable echo, c0 is that echo whitened, c1 is undefined (exclude those rows; do not zero-fill), and s and q are NaN.

Properties (checked):
1. **c0 is OC-like.** c0 equals −k_v × (the OC deviation built with tedana's weights TE·exp(−TE/T2*)): proportional with a **negative** sign. This holds exactly only when σ is equal across echoes, μ_e is monoexponential with the same T2*, and the linear ΔS/μ form is used. For log-ratio data it holds to first order.
2. **Noise is white.** The noise in c0, c1 and r is white, unit-variance and uncorrelated, but only at high SNR. At the lowest-echo μ/σ = 3 (Rician), diagonal variances are 1.03–1.20 and off-diagonals up to 0.08. It fails near μ/σ ≈ 1.
3. **c1 isolates ΔS0.** A pure ΔR2* signal gives c1 = 0 and r = 0.
4. **Back-substitution recovers the physical maps.** For channel maps (m0, m1): `s = m1/R11` is the ΔS0/S0 map and `q = (m0 − R01·s)/R00` is the ΔR2* map. Exact for pure, mixed and opposite-sign sources. SE(s) ∝ 1/R11.
5. **tedana is a special case.** Decomposing c0 alone is, after per-voxel z-scoring, a tedana-style OC decomposition (see the Phase 4 equivalence test).

Do not add extra weighting by df or echo count. Whitening already gives unit noise, and fewer echoes show up as a smaller R. Extra weights would break the rank null.

### 5.4 Thermal σ, in this order (not circular)

1. Fit μ_e (and T2*, S0) nonlinearly on the temporal mean.
2. Echo usability: exclude echo e in a voxel if μ_e/σ_e < `snr_min`.
   - Config default 5; report sensitivity at 3.
   - Magnitude noise is Rician/noncentral-χ, and in the simulation echo-4 means were biased up to 1.6× near μ/σ ≈ 1.
   - To break the chicken-and-egg loop (σ is needed to choose echoes, and echoes are needed for σ), first estimate σ only in voxels with μ_min/σ_init > ~10, using σ_init from a robust background or MP-PCA estimate.
3. σ-free residual: build the residual basis from μ∘D only (with per-echo noise: from (μ/r)∘D). It is independent of σ_v. Then `Qperpᵀ((μ/r)∘y) = σ_v·r`.
4. Estimate σ per voxel robustly over time, from **frame-to-frame differences** of the residual, Δr_t = r_t − r_(t−1) (user decision 2026-11-03; `frontend.sigma_estimator: differenced`). White thermal noise has variance 2σ² after differencing; slow, drift-paced model violations cancel. On P04 they inflated the non-differenced σ by about 8 % (brain median).
   - with df = 2: `σ² = median(Σ Δr²) / (2 · 2 ln 2)`;
   - with df = 1: `σ² = median(Δr²) / (2 · 0.4549)`.
   - The standard error is 1.10× (df = 1) and 1.12× (df = 2) that of the non-differenced form (simulation). Fast violations (e.g. respiration-paced) do not cancel.
   - The non-differenced form (`median(Σ r²) / (2 ln 2)`, `median(r²) / 0.4549`; `sigma_estimator: residual`) is kept as a sensitivity analysis.
5. Fit a smooth σ map (g-factor varies smoothly) and use it in low-SNR voxels and df = 0 voxels.
6. Whiten using that σ.
7. **Never estimate σ from temporal variance of the data.** Physiological noise scales with signal and TE, so that would be circular.

**Interpolation caveat.**
- Motion correction resamples each volume. With trilinear interpolation, noise SD can drop to about 0.35× at half-voxel shifts. It varies frame by frame and becomes spatially correlated.
- The ratio across echoes stays equal (the same transform is used for all echoes), so the weights remain valid.
- Absolute σ and the unit-variance claims are approximate on motion-corrected data.
- Run the dipy MP-PCA σ cross-check (`mppca(..., return_sigma=True)`, per echo) on **raw, non-motion-corrected** echoes. Compare it per echo with the residual model σ_v·r_e. If they differ by more than 20%, stop and report. (Under the original equal-σ model this check failed on P04 with up to 70% disagreement; that led to the per-echo model above. With per-echo weights and the non-differenced σ it failed again at 33–53%; that led to the differenced estimator in step 4.)

## 6. Architecture (build in this order)

```
src/mejoint/
  io.py          # nibabel dataobj -> float32 masked (V x T); NIfTI/TSV/JSON writers; provenance
  inventory.py   # Phase 0 audit
  simulate.py    # ground-truth multi-echo simulator (Phase 1)
  mask.py        # robust brain mask from the input echoes (noise-referenced, shading-corrected, holes filled)
  frontend.py    # mu/T2* fit, sigma (5.4), echo usability, df map, log-ratio, QR whitening -> c0, c1, r
  nuisance.py    # one projector applied to y (log-ratio, all echoes) and to r (Phase 2b)
  rank.py        # spectra, thermal and temporal nulls, shrinkage (Phase 3)
  decompose.py   # joint spatial ICA on stacked [c0 | c1], robust restarts, back-substitution
  metrics.py     # eta (7.1), map norms with CIs, nulls, tissue-class summaries, tedana concordance
  gpu_ica.py     # optional batched FastICA in PyTorch
  structured.py  # stretch: group-sparse structured factorization
  cli.py         # `mejoint run --subject ... --config ...`, `mejoint summarize`
scripts/  submit_lsf.sh  (job array over subjects + dependent summary job)
tests/    pytest unit and synthetic integration tests
configs/  default.yaml   (every tunable; copied into every output folder)
docs/     research_report.md, decisions.md (gate results), design notes
```

- **Done markers:** every stage writes its outputs atomically, plus a `done.json` holding input hashes, config hash, git SHA, package versions and seeds.
- **Skipping:** a stage skips when its marker matches.
- **Caching:** cache the front end; do not recompute it for every decomposition setting.

## 7. Decomposition (build option 1 first)

1. **Joint spatial ICA on concatenated channels (the baseline method).**
   1. Stack c0 and c1 rows along the voxel axis (≤ 2V × T), demean over time.
   2. PCA via the T×T Gram matrix in float64.
   3. Reduce to k dims **before** ICA. sklearn `FastICA` may ignore `n_components` when `whiten=False` (unverified); reduce yourself.
   4. Whiten once.
   5. Run ≥ 30 FastICA restarts and cluster them to stable components (robustica-style).

   Each component then has a shared time course and maps (m0, m1), estimated as OLS betas of the channels on the time course, which give (s, q) by back-substitution.
2. **Structured factorization (stretch).** Minimize
   `‖C0 − B(R00∘Q + R01∘S)ᵀ‖² + ‖C1 − B(R11∘S)ᵀ‖² + λ_S Σ_r‖s_r‖₂ + λ_Q Σ_r‖q_r‖₂`, plus sparsity or independence on the maps.
   - Initialize from option 1.
   - Solve by proximal gradient or ALS in PyTorch, with the R×R solves in float64 plus a small ridge term.
   - The group penalties let each component become R2*-only, S0-only or mixed.
   - Choose λ by held-out reconstruction (split the timepoints) and report sensitivity.
3. **Not now (note in docs only):**
   - IVA.
   - TensorLy block-term/Tucker-2. Option 1 is already a constrained rank-(2,1,2) block-term model with the echo factor fixed to [1, −TE].
   - Variational Bayes with ARD priors.
4. **Optional add-on:** a low-rank + sparse (GODEC/RPCA) arm for the spatially diffuse global/respiratory signal that spatial ICA misses (Power et al. 2018). Report it separately, because it can remove neural global signal.

### 7.1 Per-component measurements (classification becomes measurement)

For a component with time course t, the OLS beta noise variance per voxel and channel is 1/‖t‖² (unit-noise channels). Define the **noise-debiased S0 energy fraction** from the two physical energies (voxels with c1 undefined are excluded):

```
E_S0 = Σ_v (R01² + R11²)·s_v²  −  Σ_v (R01² + R11²)/(R11²·‖t‖²)
E_R2 = Σ_v R00²·q_v²           −  Σ_v (1 + R01²/R11²)/‖t‖²
eta  = E_S0 / (E_S0 + E_R2)     (≈ 0 pure R2*, ≈ 1 pure S0, bounded and sign-symmetric for mixed)
```

A simulation check in the planning session gave η = 0.003 (pure R2*), 1.000 (pure S0) and 0.18–0.98 for mixed sources of either relative sign. For pure noise η is arbitrary, so **interpret η only when E_S0 + E_R2 exceeds its null** (random/phase-randomized time courses).

Do **not** use either of these instead:
- **The raw channel-energy fraction φ = ‖m1‖²/(‖m0‖² + ‖m1‖²).** A pure-S0 component gives only φ ≈ 0.15–0.21 at these TEs (lower in 3-echo voxels), while pure noise gives φ ≈ 0.5.
- **E_S0 divided by total channel energy.** It is unbounded for mixed sources, because ΔS0 and ΔR2* can cancel in c0.

**Reference bands.** Define η bands by simulation: pure-R2*-only, pure-S0-only and noise-only components at **matched component SNR**, run through the same pipeline. Real-data nulls come from random and phase-randomized time courses.

**Also report for each component:**
- ‖s‖ and ‖q‖ with bootstrap intervals;
- overlap with grey matter, CSF/ventricles and vessels/sinuses (and cavity classes);
- correlation with physiological and motion regressors;
- tedana's label for the best-matching tedana component.

**Do not hard-label ΔS0 as noise.** At 3T and short TR, neural signal has real ΔS0 parts (inflow, intravascular signal, CBV), so mixed components are expected.

## 8. Phases and gates

- Keep the task list current.
- Do not start a phase until the previous gate passes.
- Record every gate result in `docs/decisions.md`.
- Simulated signal amplitudes: ΔR2* about ±1 s⁻¹ (the BOLD scale reported for 3T), ΔS0/S0 about 0.5–2%, with tSNR stated in every test.

**Phase 0: audit.**
- Inventory one subject, then all: the section 4 unknowns plus echo files and JSON fields (`EchoTime`, `ImageType`, `MultibandAccelerationFactor`, `PartialFourier`, `PhaseEncodingDirection`, `ParallelReductionFactorInPlane`, coil-combination fields).
- Confirm the TEs.
- **Gate:** show the table to the user. If phase, physio or noise frames exist, ask how to proceed (they change sections 5.4 and 9).

**Phase 1: simulator and test harness.** `simulate.py` generates echoes at the real TEs from known (s, q) maps and time courses:
- focal and diffuse R2*-only, S0-only and mixed sources;
- T2* 20–70 ms;
- complex Gaussian noise, then magnitude (Rician floor);
- optional spatially varying (g-factor-like) noise;
- a smooth global respiratory R2* signal;
- motion spikes that violate the model;
- curvature near a synthetic "skull base".

**Gate:** tests run on CPU in under 2 minutes on small grids.

**Phase 2: front end (section 5).** Tests:
- (a) Channel and residual noise covariance within ±5% of I in voxels with μ_min/σ ≥ 5.
- (b) c1 ≈ 0 for pure-R2* sources.
- (c) c0 is proportional, with negative sign, to the OC deviation (compared up to sign and scale, linear form, equal σ).
- (d) σ recovered with brain-median error < 2%, and 95% of voxels within 3 SE.
- (e) Residual energy rises where curvature or floor violations were injected.

**Gate:** all tests pass, plus one real subject's maps of μ, T2*, σ, df and residual energy, including posterior fossa slices.

**Phase 2b: nuisance projection.**
- Build one projector from the filtered motion parameters (notch- or low-pass the respiratory band before computing FD in fast-TR pediatric data), plus RETROICOR/RVT and phase-derived ΔB0(t) if available.
- Apply it to y (log-ratio, all echoes) before the WLS, and identically to r.
- Reduce the temporal df used for σ and the rank null by p + 1.
- Test: projecting the echoes and then forming channels equals projecting the channels (identical regressors commute).
- Keep a no-regression variant for comparison.

**Phase 3: rank.**
- Compute eigen-spectra of the stacked whitened channels.
- Build two nulls:
  - **Thermal:** the residual channels (and noise frames if present). Match the matrix aspect ratio (rows/T) before comparing spectra.
  - **Temporal:** phase-randomized surrogates.
- Apply the same nuisance projection and demeaning to the nulls.
- **Caveat:** the residual sees thermal noise only, because physiological noise lies inside span{1, TE}. A residual-calibrated edge sits low and pushes k up. Report k from both nulls.
- The gated estimator is the hard threshold against the phase-randomized null. Optimal shrinkage is an option for weak components.
- Always also run a fixed sweep: k = MAPCA-style AIC/KIC/MDL estimates, 50, 70 and 90.

**Gate:** simulated true rank recovered within ±10%, and real-data spectra plotted.

**Phase 4: joint ICA (option 1) and metrics.**

Tests:
- Planted sources recovered with |r| > 0.9 at the stated SNR. Correlate q for R2*-only sources, s for S0-only sources, and both for mixed sources.
- η falls in the correct simulated band.
- Pure-noise data yields no components outside the noise band.
- No spurious mixed components from correlated noise.

**Equivalence test.** Build an internal "tedana-style" path from the same inputs:
- OC with the same T2* map;
- per-voxel z-scoring over time (tedana z-scores the reduced data per voxel before ICA; its AIC/KIC/MDL path uses MAPCA with internal normalization, per the 26.0.3 source);
- fixed k and the same FastICA seeds.

Run c0 (linear form) through the same per-voxel z-scoring, fixed k and seeds. **Pass if Hungarian-matched |r| > 0.99.** Then compare loosely, as a description only, against the actual tedana outputs.

**Gate:** all tests pass, with stability across ≥ 30 seeds reported (Hungarian matching on |corr|).

**Phase 5: real-data pilot (1–2 subjects).**

Arms:

| Arm | Description |
|---|---|
| A | tedana defaults |
| B | tedana + robustica (30 runs) with a fixed `--tedpca` sweep |
| C | Robust-tedana style: MP-PCA on **raw** echoes, then the same motion correction (separate inputs; label it), then robustica 50 runs, no PCA truncation, and its modified tree (figshare 25251433; 3 criteria bypassed) |
| D | OC only |
| E | OC + global/physio regression |
| F | this method, option 1 |
| G | option 2, when available |

Tests:
- **Injection:** add focal ΔR2*-only and ΔS0-only signals to the raw echoes before any processing. Amplitude, extent and channel assignment must survive.
- **Within-subject motion check:** frame-wise FD and DVARS against residual energy.
- **MB leakage:** ΔR2* correlations at alias positions versus control positions.
- **Split-half reliability.**
- **pBOLD / echo-pair FC geometry** as a decomposition-independent referee, **only after the user supplies the full preprint** (PubMed 41929219; DOI unverified). It is a preprint metric; label it as such.

**Gate:** results table and figures shown to the user before any batch run.

**Phase 6: batch run and summary.**
- LSF job array over subjects, plus a dependent summary job.
- Compute QC-FC and distance dependence here; they need many subjects.

**Phase 7 (stretch):** option 2, the thermal-denoising arm (section 9), and notes on a probabilistic version.

## 9. Signal-to-noise: do and don't

- **No spatial smoothing** before decomposition. It inflates MB-leakage false positives and merges small cerebellar foci with CSF and veins; tedana also advises against it.
- **Thermal denoising is optional and must be validated by injection.** If used:
  - Run it per echo, before motion correction. That requires adding a hook in `mbme_tedana.py`, not duplicating its motion correction.
  - Use complex NORDIC if phase exists. Otherwise use magnitude MP-PCA/NORDIC, accepting the Rician bias.
  - Use small patches; larger ones spread focal activation.
  - Afterwards σ is no longer equal across echoes and may not be white. Re-estimate σ **per echo** from an independent reference (noise frames or injection), use general per-echo weights (then c0 is no longer exactly OC), and re-validate the residual null. Do not reuse σ from before denoising.
  - Evidence: multi-echo + NORDIC improved tSNR by ~17–25% and split-half reliability by ~14–16% in adults (Moser et al., Imaging Neuroscience; **n = 7 total: 2 adults, 2 children, 3 neonates**). Denoising can lower percent signal change (Knudsen et al. 2025) and spread activation (Fernandes et al. 2022, preprint).
- **Weights come from thermal σ only** (5.4), never from tSNR.
- **Nuisance handling is one projector applied to y** (Phase 2b). The concern in Lindquist et al. (2019) is about sequential, non-orthogonal steps. Applying identical regressors to echoes or to channels gives the same result.
- **Censoring** at filtered FD > 0.2 mm, as a sensitivity analysis. For any count-based outcome, truncate every subject to a common number of volumes.
- **Recover the separation penalty through structure,** not voxelwise fitting: low rank, shared time courses, and (stretch) temporal smoothness or HRF priors.

## 10. Pitfalls (the posterior fossa concentrates all of them)

| Violation | How to detect | What to do |
|---|---|---|
| Correlated a/b estimation noise | off-diagonal of (XᵀWX)⁻¹; spurious mixed components in noise simulations | QR whitening (never raw a, b) |
| Late-echo noise floor (Rician) | μ/σ per echo; curvature residual r0 | echo exclusion (`snr_min`), df map |
| Through-slice dephasing near air/bone (petrous, mastoid, foramen magnum, sinuses) | poor static decay fit; residual curvature | down-weight or mask; report by region |
| Respiratory B0 and CO2 are **partly** TE-dependent (the PE-shift part looks like ΔS0 or motion; the field-gradient part looks like ΔR2*) | spectral peak at 0.25–0.5 Hz; RVT and global correlation | regressors in the model; global arm; never call R2*-dominant "neural" |
| Motion, spin history | per-frame residual energy spikes; ΔS0 edge maps | censoring, motion regressors. The residual is blind to TE-linear artifacts |
| Inflow, CSF pulsation (4th ventricle, edge slices, vertebrobasilar arteries, sinuses) | high-η components in vessels/CSF | tissue-class reporting; keep mixed components |
| Cardiac aliasing (1–2 Hz folds into 0–0.67 Hz) | loading on brainstem and cisterns | RETROICOR if physio exists; report |
| MB slice leakage (leaked BOLD looks like BOLD) | correlations at alias positions | report; exclude alias-driven components |
| Echo-specific ghosting | alternating residual r1; residual time courses outside the brain | QC flag |
| Interpolation-correlated noise | residual spatial autocorrelation; per-frame σ variation | one resampling only; native space; MP-PCA check on raw echoes |
| Resection cavity, hemosiderin, clips | dropout, df map, residual | separate cavity / peri-cavity / normal-tissue summaries |
| Dimensionality and seed dependence | sweeps, restarts | report stability; counts are not the primary outcome |

A small residual does not certify a clean ΔR2*. The residual only sees structure outside span{1, TE}.

## 11. Outcomes

**Primary:**
- variance fractions in c0, c1 and the residual, per tissue class;
- ΔR2* (q-attributable) variance remaining after physiological and global regressors;
- accepted-versus-rejected-style variance, with spread across seeds;
- split-half reliability.

**Secondary,** for continuity with the user's tedana analysis: the number of components whose η falls in the simulated R2*-only band, reported across the dimension sweep and at a common number of volumes, next to tedana's "Likely BOLD" count. State plainly that counts depend on the estimator: interpolation and filtering alone change ME-ICA counts (Olafsson et al. 2015).

## 12. Efficiency and reproducibility

- **Data size and PCA method:**
  - The data are small: one echo is about 320 MB in float32 (100k voxels × 800 volumes), and the stacked channels are twice the voxel rows.
  - Use exact methods: T×T Gram eigendecomposition in float64 after demeaning. Randomized SVD gains little and loses the full spectrum that rank calibration needs.
  - Fix component signs by a convention.
- **Memory:**
  - Load with `np.asanyarray(img.dataobj, dtype=np.float32)` and keep masked (V × T) arrays.
  - Chunk the front end over voxels if needed.
  - Budget 16 GB per subject job.
- **CPU ICA:**
  - Whiten once, then `FastICA(whiten=False)` on the reduced data.
  - Run restarts with joblib using `n_jobs = $LSB_DJOB_NUMPROC` and `inner_max_num_threads=1`.
  - Avoid BLAS oversubscription (threadpoolctl).
- **GPU ICA (optional):**
  - Batch the restarts as an (R, k, k) tensor with the data resident on the device.
  - Memory roughly doubles for the stacked 2V rows.
  - Fit within gpu_short's 30 minutes or checkpoint; otherwise use gpu_priority.
- **Seeds:**
  - Use `np.random.SeedSequence(PROJECT_SEED, spawn_key=(subject_idx, run_idx, stage_idx, restart))` and log the effective seeds.
  - For GPU determinism, set `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and TF32 off (verify in the installed torch).
  - Reproducibility means matched components with |r| > 0.999, not bitwise identity.
- **Versions:**
  - Pin versions in a lockfile, with tedana pinned to 26.0.3 to match the existing analysis. Same-data T2* differences between tedana versions have been reported.
  - Record `threadpoolctl.threadpool_info()`, the GPU model and the driver.
- **Tests:**
  - Use pytest. Synthetic tests must run on CPU in minutes.
  - Mark slow and real-data tests separately.

## 13. Literature status (do not over-claim)

- **No published method was found** that decomposes multi-echo fMRI into components with separate ΔS0 and ΔR2* maps during decomposition. **Before any novelty claim, the user must obtain:**
  - Uruñuela, Veganzones and Caballero-Gaudes, ISMRM 2023 abstract #2668, "Exploring Tensor Decomposition as an Alternative to ICA for Denoising Multi-Echo fMRI data" (paywalled);
  - Uruñuela's UPV/EHU thesis.
- **Closest relatives:**
  - ME-SPFM (Caballero-Gaudes et al., NeuroImage 2019;202:116081) models ΔS0 but estimates ΔR2* only.
  - MvME-SPFM (Uruñuela et al., preprint; final venue unverified; Python package `splora`).
  - Tensor PICA (Beckmann & Smith 2005).
  - Block-term decompositions for fMRI (Chatzichristos et al.).
- **tedana 26.0.3 facts verified from source:**
  - Kappa and rho are `np.average` of voxelwise F_T2/F_S0 maps, capped at 500 and weighted by the squared standardized betas of the OC data on the z-scored mixing matrix, over voxels with adaptive mask ≥ 3.
  - For AIC/KIC/MDL, PCA runs MAPCA on the OC data with internal normalization, and the reduced data are z-scored per voxel before ICA.
  - Defaults: `--tree tedana_orig`, `--tedpca aic`, `--ica-method fastica`, `--n-robust-runs 30`, `--seed 42`.
- **Citation flags:**
  - Dowdle et al. 2023 contains no multi-echo data; the multi-echo NORDIC evidence is Moser et al.
  - Ades-Aron et al. 2021 is in Radiology.
  - Speck & Hennig 1998 is MRM 40(2):243–248; its DOI suffix is unresolved.
  - pBOLD (Gonzalez-Castillo et al.) is a preprint.
  - Tensor MP-PCA (MRM, PMID 36219475) has no mature Python implementation.
