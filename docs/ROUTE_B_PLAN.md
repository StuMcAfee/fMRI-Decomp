# Route B: BOLD-like variance and connectivity in the cerebrum (plan for approval)

Status: draft for the user's evaluation, 2026-11-08. Nothing here is built. Decisions needing approval are
collected in section 9.

## 1. Aim and scope

**Question.** After childhood cerebellar injury and its treatment, is intrinsic BOLD activity in the cerebral cortex
less integrated? Route B measures this from the physics of the four echoes, region by region, without relying on
ICA to find networks first.

**What is measured** (per participant):
1. For each cortical region: the share of its slow signal variance that is BOLD-like (a change in R2\*), the share
   that is S0-like (a change in signal size independent of echo time: motion, inflow, fluid shifts), and the share
   left unexplained by both (the "leftover", e.g. the curvature found earlier).
2. BOLD-like connectivity between cortical regions, next to S0-like connectivity and ordinary connectivity.
3. Integration: how many independent patterns the BOLD-like activity of the cortex needs (effective
   dimensionality), and how segregated the large-scale networks are.

**Scope (user decision, 2026-11-08).** BOLD outcomes come from the cerebrum only. The cerebellum is not an outcome
region: its anatomy, damage and radiation exposure vary too much between participants to interpret BOLD signals
there. Those factors are estimated in separate efforts and enter later as independent variables that may explain
remote effects on the cortex. This plan only has to deliver clean participant-level and region-level tables they
can be joined to.

**Why route B.** The injection tests showed that whole-brain ICA, ours and tedana's, isolates only signals that
are both large and strong; network-sized activity at resting strength sits near or below that limit. Averaging the
signal within a region first does the pooling ICA was supposed to do, needs no detection step, and lets the
thermal noise be removed exactly, because it is known for every voxel and echo. The physics separation itself was
shown to be correct on real data when the time course is known (injections, 2026-11-07 / 08).

## 2. What is reused

- Volume pairing and motion correction from mbme_tedana (unchanged).
- The front end: per-voxel mean decay, thermal noise (sigma) with the per-echo ratios, usable echoes per voxel,
  log-ratio signals, the 24-term motion nuisance projector.
- The injection framework (known signals added to the raw echoes, full pipeline, recovery against the truth).
- The circular-shift control (user, 2026-11-08), here as the null for coupling between regions.
- The joint ICA stays as a description of the dominant large-scale components; it is not used by route B.

## 3. Step 1: anatomy (regions in each participant's own EPI space)

1. **EPI to T1.** Register the mean echo-1 image to the participant's T1 with boundary-based registration
   (FSL `epi_reg`, which needs a white-matter segmentation from FSL FAST). Phase encoding is left-right and no field
   map exists, so distortion is not corrected; every registration is checked by eye (overlay figure per
   participant).
2. **T1 to template.** Nonlinear registration to MNI space (FSL FNIRT; ANTs is an alternative but is not verified
   on this cluster). Lesions, shunts and enlarged ventricles are masked from the cost function where present.
3. **Cortical parcellation.** Schaefer et al. (2018) parcels in MNI space, with their 7- or 17-network labels
   (Yeo et al. 2011), brought into each participant's EPI space with the inverse transforms (nearest neighbour).
   The functional data are never resampled again (one resampling only, native space; CLAUDE.md section 10).
4. **Grey-matter restriction.** Within each parcel, keep voxels with grey-matter probability above a threshold (T1
   segmentation in EPI space) that are not CSF-dominated in our data-only tissue classes. This limits the CSF
   partial volume that makes BOLD look partly S0-like.
5. **Optional non-cortical cerebral regions** (decision D1): thalamus and striatum (they relay
   cerebello-thalamo-cortical signals), from a subcortical atlas. Brainstem and cerebellum excluded.
6. **Region quality table:** voxel count, share with all four echoes usable, mean signal-to-noise per echo, CSF
   fraction, distance to susceptibility regions (orbitofrontal, temporal poles), overlap with multiband alias
   positions of other regions (slices 8 apart, 26.4 mm; the CAIPI shift is unknown).

## 4. Step 2: region signals and the BOLD-like / S0-like split

1. **Average within each region, per echo.** For each frame, average the log-ratio signal of the region's voxels,
   echo by echo, weighting each voxel and echo by its inverse thermal noise variance (zero weight where an echo is
   unusable). Because the log-ratio model is exactly linear, the region average still follows it: its S0-like and
   R2\*-like parts are weighted averages of the voxels'.
2. **Separate.** Apply the same per-voxel construction as the front end to the region's four-echo signal and its
   noise: two channels (BOLD+S0, S0 only) and two leftover channels, then back to an S0-like time course a(t) and an
   R2\*-like time course b(t) for the region.
3. **Region noise from the region's own leftover channels.** Motion-correction interpolation makes neighbouring
   voxels' noise correlated, so the noise of a region average is larger than independent voxels would give. The
   leftover channels hold thermal noise only (when the model holds), so the region's noise level is measured from
   them (frame-to-frame differences, as for sigma), not assumed.
4. **Same nuisance projection** as the front end (it commutes with averaging), restricted to the slow band
   (0.01 to 0.1 Hz) for the main measures; a frequency-resolved version is kept for the breathing band and the
   folded-down heartbeat band.

## 5. Step 3: metrics

**Per region** (each with a block-bootstrap interval over time, blocks of about 20 s):
- BOLD-like share = R2\*-like energy / (R2\*-like + S0-like energy), with the thermal noise removed (this is 1 - eta
  at region level, where the background problem of whole-brain components does not arise: the region's own
  variance is measured, no time course is regressed across the brain).
- S0-like share and leftover share (leftover energy above noise / total).
- The same split by frequency band.
- Also in percent signal change at a reference echo time (about 30 ms), for intuition.

**Between regions** (cortex; plus D1 regions if approved):
- BOLD-like connectivity: correlations between regions' R2\*-like time courses b(t); also S0-like a(t) and ordinary
  (OC-like) connectivity for comparison.
- Noise handling: thermal noise does not correlate between regions (except through multiband leakage and
  interpolation at shared borders), but it lowers correlations; covariances are used with the known noise removed
  from the diagonal, and correlations are reported against the null below.
- **Null: circular shifts.** Each region's time course shifted by its own random lag (at least 30 s, wrapping
  around): every region keeps its own dynamics, their alignment is destroyed. 1,000 shifts give the chance level
  of every edge and of every summary measure.

**Integration** (candidates for the primary outcome; decision D4):
- **Effective dimensionality** of the cortical BOLD-like covariance: the participation ratio of its eigenvalues
  (sum of eigenvalues squared over the sum of squared eigenvalues), with noise removed, against its
  circular-shift null. Less integration means more independent patterns, so a higher value. Also reported: the
  number of eigenvalues above the circular-shift null (a count of independent patterns at region level, the
  route-B counterpart of the ICA count).
- **System segregation** (Chan et al. 2014): mean within-network minus mean between-network BOLD-like coupling,
  divided by within-network coupling, on the Schaefer network labels.
- **Global BOLD-like signal share:** how much of the cortex's BOLD-like variance is shared by all regions
  (breathing-related carbon dioxide changes are BOLD-like and global; see risks).

Dimensionality depends on the number of frames and on signal-to-noise, so every participant is truncated to a
common number of volumes for the integration measures, and the participant's measured sensitivity (Step 5) is
reported with them.

## 6. Step 4: validation by injection (before any real-data interpretation)

Known signals added to the raw echoes of real runs, full pipeline, measured against the truth:
1. **Shares:** region-wide signals with set BOLD-like / S0-like mixtures at resting strength; the measured shares
   must fall within a tolerance set in advance (proposed: 0.1). This also fixes the **smallest usable region**
   (decision D2: parcel resolution).
2. **Coupling:** two injected networks with set correlations between and within them; recovered against the
   circular-shift null.
3. **Dimensionality:** the same regions driven by 4 independent time courses versus 4 copies of one; the
   participation ratio must change in the right direction and by the right amount.
4. **Leakage check:** an injected signal in one slice group must not create BOLD-like coupling with regions at its
   multiband alias positions (it cannot, since injection is after reconstruction; the real-data check is the
   comparison of alias-position edges with control edges at the same distance).
5. **Per-participant sensitivity:** the injection repeated in every participant gives the size of signal each one
   can detect, so a group difference in integration can be checked against a difference in sensitivity.

Unit tests on simulated data (CPU, minutes) come first: region averaging keeps the model linear; the split returns
the injected shares; the noise estimate from the leftover channels matches the truth with correlated voxel noise;
the circular-shift null is flat for independent regions.

## 7. Step 5 onward: pilot, then batch

- **Pilot:** P04 and P12. Show the registration overlays, region quality, share maps, matrices, eigenvalue
  spectra and the integration measures. Gate: the user reviews before any batch run.
- **Batch:** all 15 participants (LSF job array, as now). Output: one tidy table per participant (integration
  measures, global share, sensitivity, motion summaries, volumes used) and one per region (shares with intervals,
  quality flags), ready to be joined with the separately estimated cerebellar factors and, later, the control
  group.

**Sensitivity analyses:** motion handling (24-term regression, band-passed motion projected out, none); censoring at
filtered FD > 0.2 mm; global signal regression; equal versus per-echo weighting; parcel resolution; band limits;
grey-matter threshold.

## 8. Figures (cerebrum only)

- Cortical surface maps (inflated template surface) of BOLD-like share, S0-like share and leftover; per participant
  and, later, group means. Volume maps are projected to the surface for display only (nilearn, to be verified in
  the environment; FreeSurfer surfaces are optional).
- Per-participant overlays on their own T1 (registration check; regions near lesions, shunts or enlarged
  ventricles visible).
- Stacked bars by network (7 or 17): BOLD-like, S0-like and leftover shares.
- Paired connectivity matrices ordered by network (BOLD-like, S0-like, ordinary, and BOLD-like minus ordinary),
  with edges beyond the circular-shift null marked.
- Eigenvalue spectra against the circular-shift null, with the participation ratio; segregation by network.
- A quality page per participant (registration, region coverage, echo usability, alias-position overlap).

## 9. Decisions for the user

- **D1 Regions beyond cortex:** cortex only, or cortex plus thalamus and striatum as secondary regions?
  Recommendation: add thalamus and striatum as secondary (relays of cerebellar output to cortex); brainstem and
  cerebellum excluded.
- **D2 Parcel resolution:** Schaefer 200 parcels / 7 networks as primary, 400 / 17 as sensitivity. With 3.3 mm
  voxels a 400-parcel region holds roughly 40 grey-matter voxels, a 200-parcel region roughly 80 (estimates; the
  injection test sets the minimum).
- **D3 Registration tools:** FSL (`epi_reg`, FNIRT), which is available; ANTs and FreeSurfer only if verified on
  the cluster.
- **D4 Primary outcome:** effective dimensionality (participation ratio) of cortical BOLD-like activity as the
  primary integration measure; system segregation and the count above the circular-shift null as secondary;
  region shares descriptive. Added after R3 (user, 2026-11-10): the participation ratio without the first
  (brain-wide) pattern as a secondary measure, reported next to the global share.
  Added after R4 (user, 2026-11-10): thalamus and striatum without voxels of CSF probability > 0.3; a reliability
  floor of 0.5 (deep-region coupling below it reported as not measurable; cortex sensitivity set without regions
  below it); cortical CSF exclusion (> 0.3) as a sensitivity analysis of the integration measures; a deflated
  eigenvalue count under test on the R3 injections.
- **D5 Motion handling for the primary analysis:** 24-term motion regression (the current front end default) or
  band-passed motion projected out (the waves default).
- **D6 CLAUDE.md edits** (proposed text to follow on approval): section 1 (route B as the main analysis; ICA
  descriptive), section 6 (new modules: anatomy / regions, region signals, connectivity), section 8 (new phases and
  gates), section 11 (outcomes: region shares, BOLD-like connectivity and dimensionality; cerebellum excluded as an
  outcome region), and section 2 (the working branch).

## 10. Risks and how they are handled

| Risk | Handling |
|---|---|
| Registration errors (left-right distortion, lesions, shunts, enlarged ventricles) | boundary-based registration, lesion masking, a per-participant check by eye, regions with poor fit flagged |
| Breathing-related carbon dioxide changes are BOLD-like and global; they raise apparent integration | global share reported; global regression and breathing-band analyses as sensitivity; no physiological recordings, so this is estimated from the data and stated as a limitation |
| CSF partial volume makes BOLD look partly S0-like | grey-matter restriction; CSF fraction per region reported |
| Multiband leakage creates BOLD-like coupling between regions 8 slices apart | alias-position edges compared with control edges at the same distance; flagged |
| Dimensionality depends on scan length and signal-to-noise | common number of volumes; per-participant sensitivity from injection reported with every result |
| Whole-brain radiation (craniospinal irradiation is standard for medulloblastoma) may alter cortical signal and vasculature, changing BOLD-like shares for vascular, not neural, reasons | region-level S0/BOLD shares and leftover reported, so vascular changes are visible; dose to cortex can enter as an independent variable from the separate efforts |
| "BOLD-like" is not "neural" | wording throughout: R2\*-like / BOLD-like; neural interpretation only with the sensitivity analyses |

## 11. Order of work and gates

1. Approve this plan (decisions D1 to D6) and the CLAUDE.md edits.
2. Simulation tests for region averaging, the split, the region noise estimate and the circular-shift null.
3. Anatomy for the two pilot participants (registration overlays shown to the user).
4. Injection validation of shares, coupling and dimensionality; parcel resolution fixed. Gate: results shown.
5. Pilot results on P04 and P12. Gate: user review.
6. Batch over all 15 participants; tables for the joint analysis with the cerebellar factors.

Compute is light compared with the ICA work: registration takes minutes per participant, region signals and
connectivity seconds to minutes, and the injection runs reuse the existing pipeline.
