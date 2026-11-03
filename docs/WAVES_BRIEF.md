# WAVES_BRIEF.md: traveling-wave extension of the joint multi-echo decomposition

This brief extends `CLAUDE.md`. Read `CLAUDE.md` first: its rules on working with the user, the environment, stop-and-ask and reproducibility all apply here. If this brief conflicts with `CLAUDE.md` on the front end or the decomposition, `CLAUDE.md` wins; if it conflicts on waves, this file wins. The style reference for all figures is `docs/mockups/waves/` (three PNGs and `make_wave_figures.py`, made from simulated data in the planning session).

---

## 1. Goal

Add a traveling-wave analysis built on the same multi-echo physics. It has two parts.

**1. Individual stage (per participant).** It runs as an option when the decomposition is run, and also as a standalone rerun command. It:
- detects propagating modes with a complex, analytic-signal decomposition of the two whitened channels (c0, c1);
- gives each mode complex ΔR2\* and ΔS0 maps (amplitude and phase);
- computes wave measures;
- writes per-participant figures matching mock-up Figures 1 and 2.

**2. Group stage (separate command).** It runs after a batch, with a subject-label file. It:
- puts modes into correspondence across participants;
- computes subject-level measures;
- runs case–control tests;
- writes figures matching mock-up Figure 3.

**Why waves need this.** Spatial ICA, and the joint ICA of `CLAUDE.md` section 7, assume each component is one fixed map times one time course. A propagating wave has a phase that changes with location, so ICA splits it into several components. The analytic-signal decomposition represents a wave as a single complex mode.

## 2. The user, the branch, and dependencies

- **A fresh implementation (user decision, 2026-11-03).** The user has earlier complex PCA (CPCA) code. **Do not use, import, adapt or consult it.** Implement the analytic-signal decomposition from first principles in this repository.
  - The published group-concatenation CPCA framework (Bolt et al., 2022) may serve as a conceptual reference for what the method does. Its code is not a template.
  - Every convention is set explicitly in `configs/default.yaml` and justified in `docs/decisions.md`: filtering, normalization, Hilbert transform, rotation, number of modes and template matching.
- **Branch.** Create a new branch from the current project branch. Give the user the exact commands each time:
  ```bash
  git fetch origin
  git switch claude/beautiful-hamilton-141ubf && git pull     # start from the latest project work
  git switch -c claude/waves                                  # new branch for the wave extension
  git push -u origin claude/waves                             # publish it
  # later, to pick up front-end or decomposition fixes from the project branch:
  git fetch origin && git merge origin/claude/beautiful-hamilton-141ubf
  ```
  Never merge `claude/waves` into the project branch or `main` without the user's agreement.
- **What is built so far.** At the time of writing, the project branch has Phase 0–2 code: the inventory, the simulator, the front end and the robust analysis mask. Phase 2 is stopped on whether thermal σ differs between echoes (`frontend.per_echo_noise`).
- **Consequences:**
  - The wave code must take the whitening from the front-end outputs. That means per-voxel Q and R (or general per-echo weights), the mask and the df map. **Never assume equal σ across echoes.** Back-substitution uses whatever R the front end produced.
  - The joint ICA (`decompose`, Phase 4) does not exist yet. Build the wave stage as its own module with its own command now (section 6.1). When `decompose` is added on the project branch and merged in, wire `--waves` into it (section 6.2).
  - Use motion-corrected echoes (`--source mc`) for waves. Raw echoes carry motion-driven residual energy (decisions.md, 2026-11-03).

## 3. Phases and gates (record each in `docs/decisions.md`)

| Phase | Content | Gate |
|---|---|---|
| W0 | Read `CLAUDE.md`, this brief and the mock-ups; ask the open questions in section 10 | User answers recorded |
| W1 | Simulator extension (section 8.1) | Tests pass on CPU in < 2 min |
| W2 | Individual stage on simulated data (sections 4 and 5) | All section 8.2 criteria pass |
| W3 | Individual stage and figures on one real participant | User reviews the figures |
| W4 | Group stage on a simulated cohort (sections 7 and 8.3) | Null false-positive rate and power criteria pass |
| W5 | Batch run plus group stage on real data | User reviews the tables and Figure 3 |

Stop and ask if a gate fails twice, or if results look too good.

## 4. Individual stage: method

### 4.1 Inputs

From a front-end output folder:
- c0 and c1 per voxel and volume (whitened, unit noise);
- the residual;
- per-voxel R00, R01, R11;
- the analysis mask and df map;
- echo times;
- `volumes_used.tsv`;
- slice timing from the JSON sidecar (`SliceTiming`).

Voxels where c1 is undefined contribute c0 only, and their s and q are NaN (CLAUDE.md 5.3).

### 4.2 Contiguous time

Hilbert and FFT steps need evenly spaced, contiguous volumes, but some runs have pairing gaps (for example, 48246 has a 19-volume gap).
- Use the longest contiguous block.
- Interpolate gaps of ≤ 2 volumes and record it.
- If the longest block is shorter than `waves.min_contiguous_volumes` (default 400), skip waves for that run and record why.

### 4.3 Slice-timing correction

With MB 6, 48 slices are acquired in 8 groups across one TR. Uncorrected, this creates an artificial inferior–superior lag gradient. Correct each voxel's channels **in the Fourier domain**: multiply the FFT by exp(−i 2π f δ_slice), where δ_slice comes from `SliceTiming`, before band-pass and Hilbert. This is an exact fractional delay. Taper or pad the ends to limit wrap-around, and verify the sign convention with the simulator (section 8.2).

Motion correction moves voxels slightly across slices. Use the slice index in the motion-corrected grid and note the approximation.

### 4.4 Band-pass and analytic signal

- Default band 0.01–0.10 Hz (`waves.band_hz`).
- Apply the same zero-phase filter to c0, c1 and the residual.
- Take the analytic signal of each with `scipy.signal.hilbert`.

### 4.5 Normalization

**Do not z-score c0 and c1 separately.** That destroys the physical scaling that back-substitution needs.
- **Default:** keep the noise-whitened units.
- **Bolt-style option** (`waves.voxel_scaling: zscore_c0`): divide both channels of a voxel by the SD of that voxel's c0. This matches z-scored CPCA on the optimally combined data while keeping the c1/c0 ratio.

Record which was used.

### 4.6 Complex decomposition

- Stack the complex channels [c0 | c1] along the voxel axis (≤ 2V × T, complex64).
- Compute complex PCA from the T × T Hermitian Gram matrix in complex128, using `numpy.linalg.eigh`.
- Keep `waves.n_modes` (default 10) and report all of them.
- No rotation by default. If a rotation of the complex loadings is added later, it must be optional, justified in decisions.md, and tested in simulation.
- Each mode k has:
  - one complex time course **z**ₖ(t);
  - complex channel maps m0, m1, the least-squares betas of each channel on **z**ₖ;
  - physical maps by complex back-substitution: `s = m1/R11`, `q = (m0 − R01·s)/R00`. This is linear, so it is exact in complex arithmetic.
- **Fix the arbitrary global phase of each mode** with a stated convention, for example the phase of the amplitude-weighted mean of q in gray matter is set to 0. Store the convention.
- Mode centre frequency f_c is the power-weighted mean frequency of **z**ₖ.
- **c0-only equivalence test.** Write a separate minimal reference: the SVD of the band-passed analytic signal of the optimally combined data, z-scored per voxel, written independently of the main code path. The main code run on c0 alone, with the zscore_c0 option, must reproduce it. Pass: Hungarian-matched |complex correlation| > 0.99 for the top modes.

### 4.7 Per-mode measures

All in native space. Write each one to `modes.tsv`.

1. **Variance fraction** of the mode in c0 and in the stacked channels.
2. **Phase and lag maps.** Phase = arg(q) after the global-phase convention. Lag (s) = phase / (2π f_c). Store both, and state that lag is defined within one cycle.
3. **Local propagation direction and speed:**
   - Lag gradient from the complex map, with no unwrapping: ∇τ = −Im(conj(z)·∇z) / (|z|²·2π f_c).
   - Smooth the complex map first with `waves.gradient_smooth_mm` (default 4 mm FWHM).
   - Compute in 3D, in mm, using the voxel sizes.
   - Local speed = 1/|∇τ|. Report median and IQR over high-amplitude gray-matter voxels.
4. **Directional coherence R** = |amplitude-weighted mean of the unit gradient vectors| (3D).
5. **Global wave-vector fit (no unwrapping).** Maximize |Σᵥ wᵥ zᵥ exp(−i k·xᵥ)| over wave vectors k on a grid, then refine.
   - Outputs: speed = 2π f_c / |k|, direction = k / |k|, and fit quality = |Σ| / Σ|wᵥ zᵥ|.
   - A standing (non-propagating) mode has its best k near 0.
6. **η per mode and per voxel** (CLAUDE.md 7.1, with |·|²).
   - Noise debiasing must be calibrated for complex betas. Use the residual or surrogate expectation and **check it in simulation**; do not reuse the real-valued constant without checking.
   - Interpret η only where E_S0 + E_R2 exceeds its null.
7. **TE-dependent lag:**
   - Per voxel, the echo-domain complex amplitude is zₑ = s − TEₑ·q (log-ratio units).
   - Lag per echo = −arg(zₑ) / (2π f_c), unwrapped across echoes.
   - Report TE4 − TE1 lag per voxel; per mode, the median and the fraction of voxels beyond the null.
   - A single ΔR2\* source gives ≈ 0. Mixed S0 and ΔR2\* parts with different timing give a non-zero value.
8. **Echo phasors** for 3 representative voxels:
   - the highest-amplitude gray-matter voxel;
   - the highest-η voxel among those above the null;
   - a posterior fossa voxel.
9. **Path profiles.** Project voxels onto the fitted propagation axis, bin by distance (`waves.path_bin_mm`, default 6 mm), and compute per bin:
   - median lag per echo;
   - η;
   - the space–time matrix Re(mode × time course).

**Uncertainty and nulls.**
- **Circular block bootstrap over time.** Use blocks of at least one cycle of f_c, 200 resamples by default. Rerun the decomposition, match modes by |complex correlation|, and get CIs for every scalar and for per-voxel TE-lag.
- **Phase-randomized surrogates.** Randomize phase independently per voxel, keeping each power spectrum. These give the mode-energy null and the η-energy null.
- Seeds come from the project SeedSequence: (subject, run, stage = waves, resample).

### 4.8 Tissue and region labels for the measures

- **Native-space tissue classes:** gray matter, white matter, CSF/ventricles, large vessels and sinuses, and for patients cavity and peri-cavity. They do not exist yet (CLAUDE.md 4). Until they do, use the analysis mask, a "posterior fossa" box from the inferior third, and an "edge" label (voxels next to air/bone, from the mask).
- **Parcels for the group stage** (section 7.2): warp atlases to native space; never warp the complex maps for the group statistics.

## 5. Individual figures (match the mock-ups)

Write per participant and per selected mode, in `<subject>/waves/figures/`. Select modes as the top `waves.plot_modes` (default 3) by variance, or by template match once a template exists. Use the mock-up styling and keep every panel's meaning:

**`wave_mode_k.png` (mock-up Figure 1, adapted from the flat sheet to volumes):**
- A: q as an axial montage that includes posterior fossa slices. Lag on the cyclic `twilight` map, faded by amplitude, with in-plane gradient arrows on a sparse grid.
- B: s, same encoding and same amplitude scale as the 1% S0 reference.
- C: η map.
- D: cycle snapshots at 0°, 90°, 180° and 270° on one axial and one sagittal slice (BOLD-positive real part).
- E: space–time plot along the fitted propagation axis, with the fitted speed as a dashed line and the speed with its CI as text.

**`wave_diagnostics_k.png` (mock-up Figure 2):**
- A: lag along the axis, per echo (binned medians with bootstrap bands, TE ramp colours).
- B: TE4 − TE1 lag distribution against its null.
- C: η along the axis.
- D: direction histogram (polar, projected on the plane of the fitted direction) with R.
- E: echo phasors for the representative voxels.
- F: power spectrum of the mode time course, with f_c marked.

**Captions** state: simulated or real data, participant ID, mode number, f_c, n volumes used, whether slice-timing correction was applied, and the normalization.

**Volume images:** write `q_amp`, `q_phase`, `s_amp`, `s_phase`, `eta` and `lag` NIfTIs for viewing.

Cerebellar flatmaps or cortical surface views are an optional later addition. Ask the user before adding normalization dependencies.

## 6. Command-line interface

### 6.1 Now (standalone)

```
mejoint waves --frontend-dir <subject front-end output> --out <dir> [--config ...]
              [--band 0.01 0.10] [--n-modes 10] [--plot-modes 3] [--no-figures]
              [--voxel-scaling whitened|zscore_c0] [--no-slice-timing] [--bootstrap 200]
```

Writes `<out>/waves/`:
- `modes.npz`: complex maps m0, m1, s, q; time courses; f_c; phase convention;
- `modes.tsv`: one row per mode, all section 4.7 scalars with CIs;
- `maps/*.nii.gz`;
- `figures/`;
- `waves_summary.md`;
- `done.json`, with hashes, config, git SHA and seeds.

### 6.2 When `decompose` exists

`mejoint decompose ... --waves [--wave-figures]` runs the joint ICA and then the wave stage on the same cached front-end outputs, writing `waves/` beside the ICA outputs. The flags pass through to the same function as `mejoint waves`, so there is one code path.

The batch script (`scripts/*.lsf`) gains a `WAVES=1` switch, passed as an argument, not an exported variable (CLAUDE.md 3).

### 6.3 Group stage (run separately after a batch)

```
mejoint waves-group --runs-root <folder with one subject folder each> --labels labels.tsv --out <dir>
                    [--template-from all|controls] [--modes 1 2 3] [--covariates age fd_mean n_volumes coil]
                    [--n-perm 10000] [--atlas schaefer400_7net ren2019_100] [--hierarchy-order <parcel order file>]
```

`labels.tsv` (tab-separated, header required):

| subject | group | age | sex | ... |
|---|---|---|---|---|
| 47159 | patient | 12.4 | F | ... |

- `group` has exactly two levels; the reference level comes from `waves.group.reference` (default `control`).
- Covariate columns are optional.
- Subjects in the labels file without a completed `waves/done.json` are listed and the command stops, unless `--allow-missing` is given.
- Subjects with outputs but no label are listed and ignored.

## 7. Group stage: method

### 7.1 Covariates and harmonization

Always tabulate, and offer as covariates:
- number of volumes used (also truncate every subject to a common N, `waves.group.common_volumes`, as the primary analysis);
- mean filtered FD;
- coil configuration (four runs had head elements only; CLAUDE.md 4);
- pairing gaps;
- age.

### 7.2 Mode correspondence (in parcel space, no warping of complex maps)

1. **Register to template.** Register each subject's mean echo-1 image to the template with ANTs. On this cluster: `module load ANTs/...` (case-sensitive), and set `ANTSPATH=/hpcf/authorized_apps/rhel7_apps/ANTs/ea8af5f/bin/`, which the module does not set. Warp the atlas labels to native space with nearest-neighbour interpolation.
   - **Atlases (user decision, 2026-11-03):**
     - **Cerebral cortex:** Schaefer-400, with 7-network labels.
     - **Cerebellar cortex:** the 100-parcel connectivity-based cerebellar atlas of Ren, Guo and Guo (2019), *Scientific Reports* 9, 9115, doi:10.1038/s41598-019-45670-6.
       - It was derived from within-cerebellum resting-state connectivity in 57 HCP participants. Its parcels therefore do not depend on cerebral connectivity, which suits cerebellar–cortical phase measures.
       - The paper gives www.neuroguo.com/resources/ as the download site.
       - **Before use, check:** the template space of the files (the paper does not name it explicitly; likely MNI152 via the HCP pipeline), the license, the label table, and that the files open and cover the whole cerebellar cortex. Record all of this in decisions.md.
       - Do not substitute another atlas without asking.
     - Brainstem and deep cerebellar nuclei are not covered by either atlas. Ask the user before adding a third.
   - Register to each atlas's own template space, and check that the warped labels sit on the subject's cortex and cerebellum on a contact sheet.
   - Store the transforms. Do not resample q, s or channels.
2. **Parcel loadings.** Per subject and mode, take the amplitude-weighted complex mean of q (and s) in each parcel. This gives a complex parcel vector per mode.
3. **Template.** Run a group-concatenated complex PCA of parcel-averaged analytic signals, with subjects concatenated in time and the same conventions as the individual stage. Use `--template-from all` by default, with `controls` as a sensitivity analysis.
4. **Matching.**
   - Match each subject's modes to template modes by Hungarian assignment on |complex correlation|.
   - Then rotate each matched subject mode by the global phase that maximizes the real part of its correlation with the template.
   - Store match quality per subject and mode, and include it as a covariate. A worse match in patients is itself a finding and must be reported.

### 7.3 Subject-level measures per template mode

Store all of these in `subject_measures.tsv`, long format with subject, group, mode, measure, value, CI:

- variance fraction;
- wave-vector speed and fit quality;
- directional coherence R;
- mode η;
- median TE4 − TE1 lag;
- match quality;
- **cerebellar–cortical phase offset:** arg(Σ_cerebellar parcels q · conj(Σ_cortical parcels q)), amplitude-weighted, after the global-phase rotation, in degrees and in s at f_c;
- **lag profile along an ordered set of parcels.** The order comes from `--hierarchy-order` (a file listing parcels or networks in order). The user's normative cerebellar gradient work may supply one; ask.

### 7.4 Statistics

- **Scalars:**
  - Permutation test of the group difference (group labels shuffled, 10,000 permutations).
  - With covariates, use Freedman–Lane permutation on the linear model.
  - Effect sizes with bootstrap 95% CIs.
- **Angles (phase offset):**
  - Circular permutation test on the difference of circular means.
  - A separate permutation test on the difference in mean resultant length R.
  - Watson–Williams as a secondary test, noting its equal-concentration assumption.
  - Never apply linear statistics to angles.
- **Lag profile:**
  - A mixed model with region × group and a random intercept per subject (statsmodels `MixedLM`), plus a permutation test on the per-subject profile slope.
- **Multiplicity:**
  - Pre-specify the primary measures with the user before looking at real results; record them in decisions.md.
  - Apply FDR (Benjamini–Hochberg) across the remaining modes × measures.
  - Report everything, labelled primary or exploratory.

### 7.5 Group figure (`waves_group_mode_k.png`, mock-up Figure 3)

- **A–D: strip plots.** Panels are variance fraction, speed, R and η. Show every subject as a dot and the group mean with 95% CI. Print the permutation p; use "p < 0.001", never "p = 0.000".
- **E: polar plot of the cerebellar–cortical offset.** One dot per subject, circular-mean arrows, and a text block with means, R and the tests.
- **F: lag profile.** Group means with CIs, labelled directly.
- **Also:**
  - a match-quality strip plot;
  - a table figure or TSV of all tests.
- **Data labels:** captions state real or simulated, n per group, the common N of volumes, the covariates used, and primary versus exploratory.

## 8. Tests

### 8.1 Simulator extension (`simulate.py`, Phase W1)

Add to the existing simulator, at the real TEs, TR and slice timing:
1. a neural ΔR2\* wave crossing gray-matter-like voxels in 3D (planar and radial variants, speeds 2–20 mm/s, band 0.01–0.1 Hz);
2. a vascular wave: fast ΔS0 "inflow" along a vessel-like path plus delayed tissue ΔR2\*;
3. a **standing** oscillation (same phase everywhere), the negative control for propagation;
4. acquisition at the real slice times, with MB 6 groups, so uncorrected data contain the slice-timing artifact;
5. optionally, cross-talk between slices 8 apart (slice-leakage proxy).

### 8.2 Individual-stage criteria (W2)

| Test | Pass |
|---|---|
| Recovered speed of a neural wave | within 10% |
| Direction error | < 15° |
| R for the neural wave | > 0.8 |
| Standing wave after slice-timing correction | wave-vector speed > 100 mm/s or fit quality at the null level (no spurious propagation) |
| Standing wave without slice-timing correction | shows the inferior–superior artifact (proves the correction is needed and its sign is right) |
| η, neural wave | within the pure-R2\* band |
| η, vascular mode on the vessel | above the null |
| TE4 − TE1 lag | neural CI includes 0; vascular CI excludes 0 |
| c0-only equivalence with CPCA | \|r\| > 0.99 |
| Bootstrap CI coverage of true speed | 90–98% over 100 simulated runs (nominal 95%) |
| Pure noise | no mode above the energy null in > 5% of runs |

### 8.3 Group-stage criteria (W4)

Simulate cohorts (for example 20 + 20) from the W1 simulator with subject-level variability.
- **Null:** no group difference, 200 cohorts. The false-positive rate of each test must be within 3–7% at α = 0.05, circular tests included.
- **Power:** a known difference in speed and phase offset. Report power at n = 20 per group and at the real sample sizes.
- **Matching:** with modes randomly reordered and phase-rotated per subject, matching must recover the true correspondence in > 95% of subjects at realistic SNR.

## 9. Pitfalls specific to waves

- **Slice timing** (section 4.3). Without correction the inferior–superior lag gradient is guaranteed.
- **Multiband leakage.** It couples slices 26.4 mm apart at zero lag and can imprint a standing pattern. Report the alias-position check from CLAUDE.md section 8 per mode.
- **Hemodynamic delay variation.** It differs by seconds across regions, so lag maps mix neural and vascular timing. η and TE-dependent lag separate only the part with an S0 signature.
- **CO2- and arousal-linked global waves.** They are genuinely ΔR2\*. No physiological recordings exist (CLAUDE.md 4), so never call a wave neural on multi-echo evidence alone; state this in every summary.
- **The ΔS0 channel's phase is noisy per voxel** (2.2–2.6× penalty). Interpret S0 phase only at mode or parcel level, or where amplitude exceeds the null.
- **Lag is within one cycle.** Waves spanning more than a cycle wrap. Speeds come from gradients or the wave-vector fit, never from raw lag differences across a wrap.
- **Native-space comparisons.** All group comparisons go through parcels. Voxelwise group maps need warping of the complex maps (real and imaginary parts separately, after phase alignment); that is optional and must be labelled as such.

## 10. Questions to ask the user in W0

1. The hierarchy ordering for the lag profile: networks, a gradient, or a parcel list. It may span Schaefer networks and Ren parcels.
2. Primary measures and the case–control labels: which subjects are patients and which are controls, and which covariates are available.
3. Whether brainstem or deep cerebellar nuclei should be added, given that neither atlas covers them.
4. Whether cerebellar flatmaps or surface views are wanted now or later.

Already decided (2026-11-03): a fresh implementation, with no prior CPCA code; Schaefer-400 for cerebral cortex; Ren et al. (2019) 100-parcel atlas for cerebellar cortex.

## 11. Configuration (add to `configs/default.yaml` under `waves:`)

```yaml
waves:
  band_hz: [0.01, 0.10]
  min_contiguous_volumes: 400
  max_interp_gap: 2
  slice_timing_correction: true
  voxel_scaling: whitened        # or zscore_c0 (Bolt-style)
  n_modes: 10
  plot_modes: 3
  gradient_smooth_mm: 4
  path_bin_mm: 6
  bootstrap: 200
  surrogates: 200
  group:
    reference: control
    template_from: all
    common_volumes: null         # set after the batch, from the shortest usable run
    n_perm: 10000
    atlases: [schaefer400_7net, ren2019_100]   # cerebral cortex, cerebellar cortex
```
