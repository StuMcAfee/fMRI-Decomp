# ICA on parcel BOLD-like signals (branch `claude/parcel-ica`; user request 2026-11-18)

Exploratory. Nothing here changes the primary outcome or any route B measure unless the user adopts it after the
validation gates below; until then denoised results are reported next to the undenoised ones.

## 1. Aims

1. **Physiological denoising**: identify and remove cardiac- and respiratory-related structure that remains in the
   parcels' BOLD-like (dR2*) signals.
2. **Individual network maps**: network components on the 1876 icosahedral parcels (fs_LR 32k) in each participant,
   in the format the cortico-cerebellar models use, ready for later outcome tests.

## 2. What the current analysis already handles, and what is left

- Route B's coupling and integration use 0.01-0.1 Hz, which excludes the respiratory fundamental (typically
  0.2-0.35 Hz; the motion-parameter filter finds each participant's breathing peak, e.g. 0.29 Hz in the pilot).
- The 24-term motion regression removes motion-locked signal; removing the first (brain-wide) pattern took motion
  dependence of connectivity to chance (3.4 % of edges, R5).
- Left in the band: slow physiological signals (breathing depth and rate changes, CO2, heart-rate variability),
  which are BOLD-like by the echo-time test (CLAUDE.md section 10: "never call R2*-dominant neural"), partly global
  but with regionally varying amplitude; cardiac pulsation aliased by the 0.75 s TR (1-2 Hz folds to unknown
  frequencies below 0.67 Hz, possibly into the band); vein- and CSF-adjacent signal at parcel edges.
- No physiological recordings exist (CLAUDE.md section 4), so components can only be judged from the data.

## 3. Design (proposed)

- **Input per participant**: parcel BOLD-like time courses b (parcels x frames, all 797 common frames, full band),
  the S0-like time courses a, and per-parcel thermal noise; developed first on the existing Schaefer 200 route B
  outputs (available now), then on the 1876 icosahedral parcels once the surface pipeline has run.
- **Noise whitening**: each parcel's b divided by its thermal noise SD (from the leftover echo channels), so that
  thermal noise has unit variance in every parcel. Parcels hold about 13 voxels on average and differ widely in
  noise; without whitening, PCA and ICA are dominated by the noisiest parcels.
- **Dimensionality**: eigenvalues of the whitened data against (a) the Marchenko-Pastur edge for unit noise of this
  matrix shape and (b) an empirical null from the parcels' leftover-channel time courses (pure noise through the same
  steps). Phase 3 found no defensible voxel-level k because the noise is coloured and nonstationary; at parcel level
  the noise level is measured per parcel, but both estimates are reported, together with a fixed sweep of k.
- **Spatial ICA** (maps over parcels as the independent sources), at least 30 restarts clustered into stable
  components (`decompose.cluster_restarts`). Temporal ICA needs far more frames than one 10-minute run.
- **Component measurements** (classification becomes measurement, as in CLAUDE.md section 7.1):
  - spectrum: share of power above 0.1 Hz, power at the participant's breathing frequency and its first harmonic,
    narrow peaks above 0.1 Hz (candidate cardiac aliases);
  - S0 coupling: noise-debiased energy of the parcels' S0-like signals on the component time course, relative to the
    BOLD-like energy (a parcel-level analogue of eta: inflow, pulsation and motion load on S0);
  - motion: correlation with filtered FD and with the respiratory-band motion trace;
  - global: correlation with the mean BOLD-like time course;
  - spatial: loading on parcels with high CSF fraction or next to the large veins and sinuses, map smoothness.
- **Removal rule**: a pre-specified rule on these measurements, fixed by simulation and injection before any real
  data are denoised; components removed by projecting their time courses out of every parcel (one projector, as for
  the motion regressors), and the degrees of freedom reduced accordingly.
- **Network maps**: the components that pass are matched to the Yeo-7 networks on fs_LR 32k by spatial correlation
  (Hungarian matching); stability across restarts and between the two halves of the run is reported per participant.

## 4. Validation and gates

| step | what | gate |
|---|---|---|
| P1 simulation | synthetic parcel data: network sources, physiological sources (respiratory-band dR2* with vein-weighted maps, slow global breathing-related signal with regional amplitude, aliased cardiac peaks), measured per-parcel noise levels | true dimensionality and sources recovered; physiological components flagged, network components kept |
| P2 real pilot | Schaefer 200, then icosahedral parcels, in P04 and the healthy volunteer | user review of components, spectra and maps |
| P3 injection | network and physiological-like signals added to the raw echoes (route B injection framework) | recovery and correct flagging at resting strength, tolerance set in advance |
| P4 effect on measures | primary and secondary route B measures with and without denoising | user decision on adoption |

## 5. Risks

- Whole-brain spatial ICA isolated only large, strong injected sources at voxel level (Phase 5). Parcel pooling
  raises the signal-to-noise ratio per sample, but the detection limit must be measured again (P3).
- Breathing-related CO2 and blood-pressure signals are partly neural-coupled and global; removing them can remove
  neural signal (Power et al. 2018). Removal stays a sensitivity analysis until P4.
- Without recordings, cardiac components can be identified only indirectly (spectra, S0 coupling, location).
- Individual ICA on one 10-minute run may be unstable; stability is reported, not assumed.

## 6. Decisions (user, 2026-11-18: all four defaults accepted)

1. Scope: denoising as a sensitivity analysis until P4 [yes].
2. Spatial ICA on noise-whitened parcel signals [yes].
3. Full-band input for ICA (needed to see respiratory and cardiac signatures), measures still in 0.01-0.1 Hz [yes].
4. Individual ICA with Yeo-7 template matching first; group-informed methods (group ICA with dual regression) later,
   once controls exist, to avoid patient-derived spatial priors [yes].
