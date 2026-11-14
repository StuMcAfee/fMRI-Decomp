# Proposed CLAUDE.md edits (decision D6, for the user's approval)

**Status: approved and applied to CLAUDE.md on 2026-11-09 (all ten items, E1-E10).**

Drafted 2026-11-09. Each item gives the section, what changes, and the exact new text. Sections not listed (5 core
physics, 12 efficiency, 13 literature) stay as they are. Nothing is applied until the user approves; approval can be
for all items or item by item.

---

## E1. Title and section 1 (Goal): route B becomes the main analysis

**Title, replace with:**

> # CLAUDE.md: physics-based separation of BOLD-like and S0-like signal in multi-echo fMRI

**Section 1, replace the whole section with:**

> ## 1. Goal
>
> **Scientific aim (user, 2026-11-06 / 08).** The participants are survivors of childhood medulloblastoma, at least five years after diagnosis, surgery and treatment; adolescents and adults. Prediction: after childhood cerebellar injury, intrinsic BOLD activity in the cerebral cortex is less integrated. Healthy age- and sex-matched controls will be added later. The cerebellum is not an outcome region: its anatomy, damage and radiation exposure vary too much to interpret BOLD there; those factors are estimated in separate efforts and enter as independent variables for their remote effects on the cortex. Network detection is developed from the single acquisition's physics; clinical variables and reference anatomy are not used to define it.
>
> **Main analysis: route B (`docs/ROUTE_B_PLAN.md`, approved 2026-11-09).** Average the four echoes within each cerebral region, then separate each region's signal into an S0-like part (ΔS0, TE-independent) and an R2*-like part (ΔR2*, TE-dependent, BOLD-like), with the thermal noise removed. No decomposition is involved, so there is no detection limit.
>
> **Descriptive analysis: the joint decomposition (section 7).** Components that each carry a ΔS0 map and a ΔR2* map, both estimated during the decomposition (not by classifying ICA components afterwards, as tedana / ME-ICA do with kappa and rho). The injection tests (2026-11-07 / 08) showed that whole-brain spatial ICA, ours and tedana's, isolates only signals that are both large and strong; it describes the dominant large-scale components and is not the primary outcome.
>
> Deliverables:
> 1. A tested front end: the echoes as noise-whitened physical channels plus leftover channels (done).
> 2. Anatomy in each participant's own EPI space: cortical parcels, thalamus and striatum (route B step 3).
> 3. Region signals, the split, and the region noise from the leftover channels (route B step 2, done in simulation).
> 4. Region shares (BOLD-like, S0-like, leftover), BOLD-like connectivity and integration measures, with circular-shift nulls.
> 5. Validation by injection of known signals into the raw echoes, and each participant's measured sensitivity.
> 6. The joint decomposition with background-corrected η (descriptive) and the comparison with tedana.
> 7. An LSF batch interface; participant- and region-level tables for the joint analysis with the cerebellar factors and, later, the controls.
>
> Stretch goal: a structured factorization with group-sparse channel maps.

## E2. Section 2: the working branch

**Replace the Git bullet and its code block with:**

> - **Git:** all work is developed on `claude/decomposition` (user permission 2026-11-06; it carries all earlier work). Kept for reference, not developed: `claude/beautiful-hamilton-141ubf` (original project branch, Phases 0-4), `claude/waves` (traveling waves, paused), `claude/two-stage` (dropped design), `claude/pre-two-stage` (return point; it stands in for a tag, because tags cannot be pushed from this session or the cluster). Never commit directly to `main`; merge into `main` only when the user agrees. Whenever you create, switch or merge a branch, give the exact commands and one line on why:
>   ```bash
>   git pull origin claude/decomposition     # get the latest work on the working branch
>   git add -A && git commit -m "..."        # save progress
>   git switch main && git merge claude/decomposition   # only when the user agrees
>   ```

## E3. Section 3: output folder

**Add this bullet at the end of section 3:**

> - **Outputs:** the repository's `outputs/` is a link to `/research_jude/rgs01_jude/dept/DI/tir/vision_lab/stu/CSIMemphis/Analysis/MEJOINT_QC/` (user, 2026-11-09; `scripts/use_output_root.sh`), so every job writes there and earlier results stay available as inputs. Job logs stay in the repository's `logs/`. FSL 6 (`module load fsl/6`) provides the registrations.

## E4. Section 4: participants, T1 and tissue masks

**Add after the "Participants are children and young adults..." bullet:**

> - The patients are medulloblastoma survivors (section 1). Craniospinal irradiation is standard treatment, so cortical vasculature may be affected: report S0-like and leftover shares next to BOLD-like ones.

**Replace the bullet "Tissue masks (grey matter, CSF/ventricles, vessels/sinuses) and cavity/peri-cavity masks: no ready-made source exists..." with:**

> - Tissue masks: no ready-made source exists (user). Each session has a T1w MPRAGE (`.../CSIMemphis/Data/Rad/<subject>/<date>/.../*SAG_3D_T1_MPRAGE*.nii.gz`; list file from the user, 2026-11-09). The T1 option is built for route B (`src/mejoint/anatomy.py`, `scripts/anatomy_run.lsf`: FSL bet, fast, epi_reg, flirt, fnirt; Schaefer 2018 parcels bundled in `atlases/`; Harvard-Oxford subcortical from FSL). The data-only option (`src/mejoint/tissue.py`: classes from the multi-echo data alone) remains, and its CSF and dropout classes are excluded from regions. Cavity / peri-cavity masks: not built.

## E5. Section 6: architecture

**Replace the code block with:**

> ```
> src/mejoint/
>   io.py            # nibabel dataobj -> float32 masked (V x T); NIfTI/TSV/JSON writers; provenance
>   inventory.py     # Phase 0 audit
>   simulate.py      # ground-truth multi-echo simulator (Phase 1)
>   mask.py          # brain mask from the input echoes (noise-referenced, shading-corrected, holes filled)
>   frontend.py      # mu/T2* fit, sigma (5.4), echo usability, df map, log-ratio, QR whitening -> c0, c1, r
>   nuisance.py      # one projector applied to y (log-ratio, all echoes) and to r (Phase 2b)
>   tissue.py        # data-only tissue classes and partial-volume coupling
>   rank.py          # spectra, nulls, MAPCA estimates (Phase 3)
>   anatomy.py       # route B: T1 matching, atlas tables, region labels in EPI space, QC figures
>   regions.py       # route B: region signals, split, region noise, shares, coupling, dimensionality, circular-shift null
>   inject.py        # known sources added to the raw echoes; recovery against the truth; detectability
>   decompose.py     # joint spatial ICA on stacked [c0 | c1] (descriptive), robust restarts, back-substitution
>   metrics.py       # eta (7.1), map norms with CIs, nulls, tissue-class summaries
>   eta_background.py  # background-corrected eta (circular-shift and phase surrogates; whole brain and support)
>   eta_recheck.py, te_check.py, sweep_batch.py   # post hoc checks and summaries of the decomposition
>   waves.py, wave_figures.py, waves_batch.py     # traveling waves (paused; docs/WAVES_BRIEF.md)
>   cli.py           # `mejoint <command>`
> atlases/  schaefer2018/  (bundled parcellations, MIT license)
> scripts/  LSF job and submit scripts (pipeline, injection, anatomy, eta-background, output folder)
> tests/    pytest unit and synthetic integration tests
> configs/  default.yaml   (every tunable; copied into every output folder)
> docs/     research_report.md, decisions.md (gate results), ROUTE_B_PLAN.md, WAVES_BRIEF.md, design notes
> ```

## E6. Section 7: the decomposition is descriptive; background-corrected eta; working k

**Replace the section heading and add a paragraph before item 1:**

> ## 7. Decomposition (descriptive)
>
> Since 2026-11-09 the joint decomposition describes the dominant large-scale components; route B is the main analysis. Working k = 150 (user, 2026-11-08): injected network-sized sources separate only around k = 150 (focal and resting-strength sources are not isolated at any tested k). Results at other k stay reported.

**Add at the end of section 7.1:**

> **Background-corrected eta (user decision 2026-11-08; `eta_background.py`).** The thermal debiasing above does not remove the energy that any slow time course picks up from the data's own slow fluctuations over ~40,000 voxels; that background has eta ~0.25-0.45 and set every native component's reported eta (~0.42). Remove it: for each component, 50 surrogates of its own time course (primary: circular shifts by at least 30 s, the end wrapping to the beginning, user's choice; secondary: phase randomization), their mean E_S0 and E_R2 subtracted from the component's own, over the whole brain (with a 5-95 % interval) and over the component's support (voxels chosen in one half of the run, measured in the other). The surrogate p is descriptive only: a data-derived component beats shifted copies of itself even on noise; which components count stays with the pipeline null.

## E7. Section 8: route B phases and gates

**Add after the Phase 5 block:**

> **Phase 5 outcome (2026-11-07 / 08):** injection into the raw echoes, full pipeline. The front end measures ΔS0 vs ΔR2* correctly when the time course is known (ratio, amplitude within ~10 %, no added curvature); processing does not create the native curvature; whole-brain ICA (ours and tedana's) finds no focal source and finds network-sized sources only when strong and at k ~150; the reported component eta measured the background (section 7.1).
>
> **Route B (`docs/ROUTE_B_PLAN.md`; approved 2026-11-09).**
> - **R1, simulation tests** (done 2026-11-09): averaging keeps the model linear; region noise from the leftover channels; shares; dimensionality and the circular-shift null; segregation.
> - **R2, anatomy** for the pilots. **Gate:** the user checks every registration (QC figures).
> - **R3, injection validation:** shares, coupling and dimensionality recovered from injected region-wide signals at resting strength (share tolerance 0.1, set in advance); smallest usable region fixed; per-participant sensitivity. **Gate:** results shown to the user.
> - **R4, pilot** (P04 and a second participant): maps, matrices, spectra, integration measures. **Gate:** user review before any batch run.
> - **R5, batch** over all participants: participant- and region-level tables; QC-FC and distance dependence (they need many subjects).
>
> Traveling waves: paused (user, 2026-11-06).

**Replace "Phase 6: batch run and summary" with:** "Phase 6: replaced by R5."

## E8. Section 9: region averaging is not smoothing

**Add after the "No spatial smoothing" bullet:**

> - Route B's region averaging is not smoothing: whole parcels are pooled in native space, no signal is blurred across region borders, and the functional data are never resampled.

## E9. Section 10: three new pitfalls

**Add these rows to the table:**

> | Background energy in whole-brain component metrics | eta uniform (~0.4) across components and participants; a known pure source reads mixed | background-corrected eta (7.1); region-level measures (route B) |
> | Detection limit of whole-brain spatial ICA | injected sources not recovered; share of a source's time course in the top-k PCA dimensions | route B; report k; injection-measured sensitivity per participant |
> | Echo profiles from noise-selected voxels | zero crossing near the noise-weighted mean TE (29-37 ms) and a ~32 ms ratio even on pure noise | profiles at voxels fixed independently (injection truth, regions) |

## E10. Section 11: outcomes

**Replace the whole section with:**

> ## 11. Outcomes
>
> Cerebral regions only (section 1): Schaefer 200 parcels / 7 networks (400 / 17 as sensitivity) and, as secondary regions, left and right thalamus and striatum.
>
> **Primary:** effective dimensionality (participation ratio) of cortical BOLD-like activity (noise-corrected correlations of the regions' R2*-like time courses), against its circular-shift null, at a common number of volumes, reported with each participant's injection-measured sensitivity. Less integration means higher dimensionality.
>
> **Secondary:**
> - system segregation of the BOLD-like correlations; the number of eigenvalues above the circular-shift null;
> - region BOLD-like, S0-like and leftover shares, by network;
> - BOLD-like versus S0-like versus ordinary connectivity; global BOLD-like share;
> - thalamic and striatal BOLD-like coupling with the cortical networks.
>
> **Descriptive:** joint-decomposition components across k with background-corrected eta; the comparison with tedana's components and "Likely BOLD" counts. Counts depend on the estimator, k, scan length and signal strength (interpolation and filtering alone change ME-ICA counts, Olafsson et al. 2015); they are not outcomes.
>
> Sensitivity analyses: motion handling (24-term regression primary, band-passed motion projection), censoring at filtered FD > 0.2 mm, global regression, equal versus per-echo weighting, parcel resolution, band limits, grey-matter threshold. "BOLD-like" is not "neural": breathing-related CO2 and field changes are partly BOLD-like and global.
