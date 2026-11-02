# Whitened two-channel decomposition can supersede kappa-rho classification

**Bottom line.** No published method decomposes multi-echo fMRI into components that carry separate ΔS0 and ΔR2* maps during the decomposition. The nearest prior work is a paywalled ISMRM 2023 abstract on tensor decomposition for multi-echo denoising and the deconvolution line (ME-SPFM, MvME-SPFM), which models ΔR2* only. The method is therefore novel. It is also simple to build. A per-voxel noise-whitened projection of the 4 echoes onto span{1, −TE} gives two decorrelated channels: one is the optimally combined (OC) signal and the other carries only ΔS0. A joint spatial ICA or structured factorization on those channels, with shared time courses, gives each component a ΔS0 map and a ΔR2* map. The main cost is statistical. In a per-voxel, per-volume fit, estimating ΔS0 jointly inflates the ΔR2* noise SD by 2.2 to 2.6× compared with assuming ΔS0 = 0, roughly an 80% information loss. That is why the separation belongs at the component level, where each time course pools many voxels, and not in voxelwise maps. The biggest scientific risks are not algorithmic. Respiratory CO2 and B0 effects are truly TE-dependent and survive any TE-based separation. Neural signal at 3T carries some ΔS0 from inflow and blood volume. Posterior fossa voxels near the skull base, and around resection cavities, break the monoexponential, Gaussian-noise model most often. The workflow is small by HPC standards: about 1.3 GB per subject in float32, exact Gram-matrix PCA, and FastICA restarts batched on the GPU. Reproducibility comes from careful precision handling, seeds derived from (subject, run, stage, restart), pinned environments, and a synthetic-data test suite.

**How to read the flags.** This report goes into a Claude Code implementation guide, so every non-trivial claim carries a status tag:

| Tag | Meaning |
|---|---|
| **[fetched]** | A researcher opened the source and checked the content |
| **[title-verified]** | The title or venue was seen in search results; the body was not read |
| **[prior knowledge]** | Given with a DOI that was not resolved this session; spot-check before citing |
| **[preprint]** | Not peer reviewed |
| **[coordinator-verified]** | Read directly from the tedana 26.0.3 source code |
| **[own calc]** | Arithmetic done for this report |
| **[own inference]** | The researchers' reasoning, not a published finding |

## tedana adds physics only after a physics-blind decomposition

tedana and ME-ICA reduce the echo dimension before decomposing. They run PCA and then spatial ICA on the optimally combined data, regress each component time course back onto every echo, and only then score TE dependence with voxelwise F-statistics. Kappa scores the fit to the R2* model and rho the fit to the S0 model, and a decision tree uses both ([tedana approach docs](https://tedana.readthedocs.io/en/stable/approach.html)) **[fetched]**. OC weights each echo by TE·exp(−TE/T2*), normalized across echoes. The default T2*/S0 fit is log-linear on log(|S|+1). `--fittype curvefit` does a nonlinear fit and `--fitmode ts` gives volume-wise T2*/S0 estimates. ICA and the fit maps need at least 3 good echoes in the adaptive mask, while OC and the denoised output need only 1 ([tedana approach docs](https://tedana.readthedocs.io/en/stable/approach.html)) **[fetched]**.

The coordinator checked the details the notes could not confirm against the tedana 26.0.3 source ([tedana repository](https://github.com/ME-ICA/tedana)) **[coordinator-verified]**:

- **Kappa and rho are `np.average` of the voxelwise F_T2 and F_S0 maps.** The weights are the squared standardized betas from regressing the OC data on the z-scored mixing matrix (`calculate_weights`). The average runs over voxels with adaptive-mask value ≥ 3, and the F maps are capped at 500. The notes' recalled formula, which weighted by the squared component z-map clipped at 8, does not describe the current code and should not be used to reimplement kappa.
- **Defaults:** `--tree tedana_orig`, `--tedpca aic`, `--ica-method fastica`, `--n-robust-runs 30`, `--seed 42`.
- **Classification tags** include 'Likely BOLD', 'Unlikely BOLD', 'Less likely BOLD', 'Accept borderline', 'Low variance' and 'No provisional accept'.

The fetched usage page spells the flags `--ica_method` and `--n_robust_runs` ([tedana 26.0.3 usage](https://tedana.readthedocs.io/en/stable/usage.html)), so the implementation session should run `tedana --help` in the pinned environment and record the exact invocation. The `tedana_orig` tree matches `meica` through node 21 and then rejects more. `minimal` is described as simpler and more stable but not extensively validated ([tedana decision trees](https://tedana.readthedocs.io/en/stable/included_decision_trees.html)) **[fetched]**.

The documented weaknesses all follow from that two-stage design:

- **PCA dimensionality is unstable.** The tedana FAQ says "there is no consistent number of components" and that all estimators sometimes fail ([tedana FAQ](https://tedana.readthedocs.io/en/stable/faq.html)) **[fetched]**.
- **Results depend on the seed.** Robust-tedana found that activation volume changed by more than an order of magnitude across ICA seeds in some subjects. Its fix (MP-PCA before fMRIPrep, no PCA truncation, 50 robustica runs, three tree criteria bypassed) beat standard tedana in about 80% of 250 epilepsy patients, at 27 min versus 4 min runtime ([Robust-tedana bioRxiv](https://www.biorxiv.org/content/10.1101/2025.06.16.660050.full.pdf)) **[fetched, preprint]**.
- **Short TR hurts.** On short-TR SMS rest data, ICA failed to converge in 3 of 12 subjects and 6 of 12 had no BOLD ICs ([Olafsson, Kundu, Liu ISMRM 2015](https://cds.ismrm.org/protected/15MProceedings/PDFfiles/2053.pdf)) **[fetched]**.
- **Spatially diffuse noise escapes.** Spatial ICA cannot capture it, so tedana adds GSR and minimum-image regression ([tedana approach docs](https://tedana.readthedocs.io/en/stable/approach.html)) **[fetched]**.
- **Mixed components get one label.** A component that loads on both S0 and R2* (for example motion-coupled BOLD) is accepted or rejected as a whole **[own inference]**.

Voxelwise ΔS0/ΔR2* estimation has a long history and every report finds a noise cost:

- Speck and Hennig (1998) fitted I0 and T2* from multi-echo EPI. They separated inflow-driven I0 increases, seen only at short TR, from true T2* changes ([Scilit record](https://scilit.com/publications/70fa69efe4f03cc0d102288165ba9c0d)) **[fetched]**. The two notes give conflicting DOIs (10.1002/mrm.1910400210 and …0211); resolve this before citing.
- Heunis et al. (2021) found that the time-resolved T2*FIT series gave the largest effect sizes and real-time contrasts but lower tSNR ([DOAJ abstract](https://doaj.org/article/44a87b9e0a9f4abe8c7cffe4ad46e06d)) **[fetched abstract]**.
- Michálek and Mikl (2025) measured a **median tSNR of 19.2 for raw dynamic T2*, against 69.9 for OC and 76.1 for tedana**. TV regularization restored it to 68.8 with the best CNR (0.86) ([Frontiers 2025](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1544748/full)) **[fetched]**.

The deconvolution line deliberately avoids this cost. ME-SPFM (NeuroImage 2019;202:116081, not HBM as the original brief stated) writes the same intercept-plus-slope model. It then deconvolves the R2*-only term, assuming ΔS0 is removed in preprocessing, and notes that it "can be modified straightforwardly to estimate both" with separate regularization ([EHU record](https://addi.ehu.es/handle/10810/38274?show=full); [bioRxiv 558288](https://www.biorxiv.org/content/10.1101/558288v1)) **[fetched]**. MvME-SPFM stacks echoes with TE-scaled HRF blocks, solves an ℓ1 + ℓ2,1 problem with FISTA and stability selection, and again leaves ΔS0 unestimated ([bioRxiv](https://www.biorxiv.org/content/biorxiv/early/2022/10/13/2022.09.30.510299.full.pdf)) **[fetched; final venue unverified]**. Its Python package, splora, provides reusable FISTA and ℓ2,1 code.

The nearest prior art for treating echo as a decomposition mode is the Uruñuela, Veganzones and Caballero-Gaudes abstract "Exploring Tensor Decomposition as an Alternative to ICA for Denoising Multi-Echo fMRI data" ([ISMRM 2023 #2668](https://archive.ismrm.org/2023/2668.html)) **[title-verified; body paywalled]**. **Get its full text, and check Uruñuela's UPV/EHU thesis, before claiming novelty.** The general machinery is established:

- Tensor PICA, a voxel × time × session CP model with spatial independence, available in MELODIC ([Beckmann & Smith 2005](https://www.researchgate.net/publication/8000208_Tensorial_extensions_of_independent_component_analysis_for_multisubject_FMRI_analysis)) **[title-verified]**.
- Block-term decompositions (BTD and BTD2), which held up under strong noise and spatial overlap ([Chatzichristos et al., PubMed 30553751](https://www.pubmed.gov/30553751/)) **[title-verified]**.

Probabilistic and deep-learning work is thin. It amounts to Bayesian shrinkage of voxelwise R2* ([Xu et al. ISMRM 2014](https://archive.ismrm.org/2014/4127.html)), "Bayesian denoising" listed as in development by tedana ([tedana multi-echo docs](https://tedana.readthedocs.io/en/stable/multi-echo.html)), and DUNE, a U-Net denoiser comparable to MEICA with no S0/R2* separation ([NeuroImage 2026](https://www.sciencedirect.com/science/article/pii/S1053811926000674)) **[title/abstract-verified]**. None of them does physics-constrained source separation.

One evaluation tool stands out. pBOLD scores a scan by where echo-pair FC edges fall: S0-dominated edges lie on the identity line and BOLD edges on a TE-determined slope. GSR raised tSNR but lowered pBOLD, and higher pBOLD went with better phenotype prediction (N = 439 scans) ([PubMed 41929219](https://pubmed.ncbi.nlm.nih.gov/41929219/)) **[fetched abstract, preprint]**. pBOLD is independent of any decomposition, so it can referee the new method against tedana.

## The joint fit costs 80% of voxelwise information, which argues for component-level separation

### Signal model and the noise penalty

The per-voxel model is `y_e(t) = a(t) − TE_e·b(t) + ε_e(t)`, where `y_e = ln(S_e(t)/μ_e)` (or ΔS_e/μ_e), `a = ΔS0/S0`, `b = ΔR2*`, and `μ_e = S0·exp(−TE_e/T2*)` is the temporal-mean signal. With thermal noise σ roughly equal across echoes, `Var(y_e) ≈ σ²/μ_e²`, so the WLS weights are `w_e = μ_e²/σ²` ([tedana approach docs](https://tedana.readthedocs.io/en/stable/approach.html) for the signal model; weights **[own inference]**). The log-ratio form makes the model exactly linear. The linear ΔS/S form errs by 3% at TE4 for ΔR2* = 1 s⁻¹ but by 9% for a 20% motion spike **[own calc]**.

The cost of estimating `a` jointly with `b`, against constraining a = 0, is fixed by the TE design. The coordinator's figures were checked here **[own calc, reproduced]**:

| T2* (ms) | SD inflation of ΔR2*, joint vs ΔS0 = 0, 4 echoes | corr(â, b̂) | Weighted mean TE (ms) | 3 echoes (61 ms echo dropped), vs 4-echo ΔS0 = 0 baseline | 3 echoes, vs 3-echo ΔS0 = 0 baseline |
|---|---|---|---|---|---|
| 20 | 2.57× | 0.92 | 20.4 | 2.86× | 2.76× |
| 30 | 2.25× | 0.90 | 23.8 | 2.76× | 2.52× |
| 45 | 2.19× | 0.89 | 27.6 | 2.91× | 2.48× |
| 60 | 2.21× | 0.89 | 29.9 | 3.07× | 2.50× |
| 70 | 2.23× | 0.89 | 31.1 | 3.16× | 2.51× |

Across T2* = 20 to 70 ms, the inflation is **2.2 to 2.6× in SD, an information loss of about 79 to 85%** per voxel-volume, with **estimate correlation 0.89 to 0.92**. Dropping the 61 ms echo raises the inflation to **about 2.8 to 3.2× relative to the full 4-echo ΔS0 = 0 estimator** (the coordinator's baseline). Relative to a 3-echo ΔS0 = 0 estimator the figure is 2.5 to 2.8×. At T2* = 45 ms, in units of σ relative to S0, sd(ΔS0/S0) = 2.27σ and sd(ΔR2*) = 73σ s⁻¹ jointly, against 33σ s⁻¹ with ΔS0 = 0 **[own calc]**. Unweighted OLS (the analog of tedana's loglin fit) is worse: an SD ratio of about 2.5, or 6.3 in variance **[own calc]**.

Two corrections to the notes follow:

- **The intercept can be decorrelated from ΔR2*, but only at the WLS-weighted mean TE**, about 20 to 31 ms depending on T2*. The arithmetic mean TE of 38.7 ms suggested in one note does not do it **[own calc]**.
- **Re-centring does not reduce the ΔR2* variance at all.** It only removes the covariance term, which is why whitening is still required.

### Whitening that makes the two channels physically interpretable

The following construction **[own inference/derivation]** should be the front end. Whiten the echo vector per voxel, `ỹ_e = (μ_e/σ_e)·y_e`, so the noise covariance becomes I₄. Then take a thin QR of the whitened design with the R2* column first: `W^½[−TE, 1] = Q·R`, where Q is 4 × 2 orthonormal and R = [[R₀₀, R₀₁], [0, R₁₁]] is upper triangular. The channel coordinates `c = Qᵀỹ = R·[b, a]ᵀ + n` have white, unit-variance, mutually uncorrelated noise. They also carry clean meanings:

- **c₀** weights raw ΔS_e by TE_e·μ_e, which is exactly tedana's OC weighting. It is therefore the maximum-likelihood ΔR2* estimate under ΔS0 = 0, and it responds to both sources.
- **c₁ = R₁₁·a + noise depends only on ΔS0.**
- **The 2-D complement `Q⊥ᵀỹ`** is pure thermal noise plus model violation under the model.

A pure-BOLD component has a zero c₁ map. A pure-S0 component appears in both channels with a known per-voxel ratio R₀₁/R₁₁. Any component's physical maps follow by back-substitution: `s = m₁/R₁₁` and `q = (m₀ − R₀₁·s)/R₀₀`. The scheme also contains tedana as a special case. Decomposing c₀ alone is a noise-whitened OC decomposition, so the new method adds the c₁ channel and loses nothing at the decomposition stage. The 80% penalty applies only to per-voxel, per-frame ΔS0/ΔR2* maps. A component time course pools thousands of voxels, so its S0-versus-R2* ratio is estimated from spatially pooled evidence, which is the same logic that makes kappa and rho work **[own inference]**.

Where fewer echoes are good, Q comes from the good echoes. 3 echoes leave 1 residual df and 2 echoes leave 0. With 1 good echo, c₁ is undefined, so give that voxel zero weight in channel 1 and store a per-voxel df map ([tedana approach docs](https://tedana.readthedocs.io/en/stable/approach.html) for the echo thresholds; handling **[own inference]**).

### Candidate decompositions, in build order

The table below summarizes the five options; the paragraphs after it give details.

| Option | Method | Strength | Main risk | Code base |
|---|---|---|---|---|
| 1 (build first) | Spatially concatenated joint ICA on [c₀ \| c₁] (2V × T), shared mixing | Reuses tedana-style PCA and robustica; direct baseline | Assumes both channels share spatial statistics | sklearn FastICA (whiten=False), robustica |
| 2 | Structured factorization with group-sparse channel maps | Separates sources inside the objective; handles mixed components | Choosing λ; non-convex | PyTorch, proximal gradient or ALS |
| 3 | IVA on two datasets | Allows lagged or absent time courses per channel | Loses the strict shared-time constraint | No Python fMRI IVA confirmed |
| 4 | Constrained BTD/Tucker-2 in echo space, echo factor fixed to [1, −TE] | Strong uniqueness, no independence assumption | Equivalent to option 2 when weighted; CP alone forces rank-1 echo signatures | TensorLy (PyTorch backend) or custom |
| 5 (add-on) | Low-rank + sparse (GODEC/RPCA) on the R2* channel | Captures the diffuse global signal spatial ICA misses | Can remove neural global signal | Power 2018 code |

**Option 1** stacks the two whitened channels along the voxel axis and runs temporal-mixing spatial ICA. Each component gets a shared time course and a [m₀ | m₁] map pair. This is the fastest path to a working baseline.

**Option 2** is the formulation that actually separates sources during the decomposition. It minimizes `‖C₀ − B(R₀₀∘Q + R₀₁∘S)ᵀ‖² + ‖C₁ − B(R₁₁∘S)ᵀ‖² + λ_S Σ_r‖s_r‖₂ + λ_Q Σ_r‖q_r‖₂`, plus a sparsity or independence term on the maps. Under the group penalty each component self-selects as R2*-only, S0-only or mixed. Option 1 provides the initialization. The structure follows ME-SPFM's suggestion of separate regularization per term ([bioRxiv 558288](https://www.biorxiv.org/content/10.1101/558288v1)) **[own inference for the formulation]**.

**Option 4** relies on an equivalence. Each component's echo-space contribution, `b_r ⊗ (s_r·1ᵀ − q_r·TEᵀ)`, is a multilinear rank-(2,1,2) block term with a fixed echo factor matrix. Concatenated joint ICA is therefore a constrained Tucker-2/BTD **[own inference]**. That makes the Chatzichristos BTD work and the ISMRM 2023 abstract the algorithmic ancestors.

A fully probabilistic version also appears unclaimed **[own inference]**: a heteroscedastic Gaussian likelihood in echo space, with ARD priors on s_r and q_r, fitted by variational Bayes. It would address dimensionality choice and seed instability together.

Classification still happens, but it becomes a physical measurement rather than a heuristic tree:

- For each component, report the c₁ energy fraction `φ_r = ‖m₁‖²/(‖m₀‖² + ‖m₁‖²)` against its null distribution from the residual channels.
- Report the back-substituted ‖q‖ and ‖s‖ with bootstrap intervals.
- Report the spatial overlap with grey matter, vessels and CSF.
- Report fits to external physiological regressors.

Mixed components are expected (next section), so avoid a hard ΔS0 = noise rule **[own inference]**.

## Posterior fossa anatomy concentrates every model violation

### Motion and B0 load both channels in different spatial patterns

Power et al. found that motion-related S0 variance is "mostly focal, and thus distance-dependent". Motion-related R2* variance is "mostly global". ME-ICA "unmasks a strong global motion-associated influence in R2* data": mean QC:RSFC rose from 0.08 in OC data to 0.21 after ME-ICA. Censoring at FD > 0.2 mm reduced the residual influence to negligible levels ([Power et al. 2018, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5834724)) **[fetched]**.

Respiration-driven B0 change has three distinct effects **[own inference]**:

- a TE-linear phase, which leaves the magnitude unchanged;
- a phase-encode image shift that is about equal across echoes, so it looks like ΔS0 and like apparent motion;
- a change in the within-voxel field gradient, which grows with TE and so looks like ΔR2*. ME-ICA accepts this as BOLD.

The third effect is largest near the petrous bone, mastoid air cells and foramen magnum. Respiratory B0 fluctuation is spatially non-uniform and larger inferiorly ([Van de Moortele et al. 2002](https://doi.org/10.1002/mrm.10145)) **[prior knowledge]**. Respiration also contaminates motion estimates in fast-TR pediatric data, so motion parameters should be notch- or low-pass-filtered before computing FD ([Fair et al. 2020](https://doi.org/10.1016/j.neuroimage.2019.116400); [Gratton et al. 2020](https://doi.org/10.1016/j.neuroimage.2020.116866)) **[prior knowledge]**.

**The residual is blind to any artifact whose TE profile is linear.** The residual is W-orthogonal to [1, −TE], so dephasing that grows roughly linearly over 16 to 61 ms, and spin history, are absorbed into a and b. A small residual therefore does not certify a clean ΔR2* **[own inference]**.

### Much of the TE-dependent variance is physiological, not neural

Respiratory variability correlated with global-signal variability at r = 0.64 in fitted R2* and 0.59 in ME-ICA-denoised data. Neural signal was estimated at only about 5 to 20% of the recorded fMRI signal ([Power et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5834724)) **[fetched]**. Components near the kappa threshold tend to correlate with low-frequency RVT ([Handwerker 2024 slides](https://fmrif.nimh.nih.gov/images/3/3a/12_MultiEcho_Summer_Handwerker2024.pdf)) **[fetched]**. CO2-driven signal is BOLD by the TE criterion, yet ME-ICA splits or rejects it inconsistently ([Moia et al. 2021](https://doi.org/10.1016/j.neuroimage.2021.117914)) **[prior knowledge]**.

The sampling numbers below are **[own calc]**:

- At TR 0.75 s the Nyquist frequency is 0.667 Hz.
- Pediatric respiration (about 0.25 to 0.5 Hz) is sampled directly and appears as a resolvable spectral peak in both channels.
- Cardiac frequencies of about 1 to 2 Hz alias into the resting-state band (1.5 Hz aliases to about 0.17 Hz).

Physiological recordings are close to mandatory. Without them, the only way to remove respiratory global fluctuations "is to remove all global fluctuations" ([Power et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5834724)) **[fetched]**. Cardiac pulsatility concentrates in the brainstem, basal cisterns and ventricles ([Dagli et al. 1999](https://doi.org/10.1006/nimg.1998.0424)) **[prior knowledge]**. At fast TR, CSF inflow into the fourth ventricle at the edge slice produces large oscillations anti-phased with cortical BOLD ([Fultz et al. 2019](https://doi.org/10.1126/science.aax5440)) **[prior knowledge]**.

### Neural signal is not purely ΔR2*

Inflow can produce activation in large vessels "distant from initial neuronal activity" ([Handwerker 2024](https://fmrif.nimh.nih.gov/images/3/3a/12_MultiEcho_Summer_Handwerker2024.pdf)) **[fetched]**. Speck and Hennig saw I0 rises only at short TR ([Scilit](https://scilit.com/publications/70fa69efe4f03cc0d102288165ba9c0d)) **[fetched]**. At 3T, intravascular signal with its own S0 and short T2* is a substantial part of gradient-echo BOLD ([Uludağ et al. 2009](https://doi.org/10.1016/j.neuroimage.2009.05.051)) **[prior knowledge]**. Changes in blood volume shift compartment fractions and so the intercept ([Lu & van Zijl 2005](https://doi.org/10.1002/mrm.20379)) **[prior knowledge]**. tedana's own FAQ admits that a clear V1 checkerboard response was "sometimes rejected" ([tedana FAQ](https://tedana.readthedocs.io/en/stable/faq.html)) **[fetched]**. At TR 0.75 s, inflow in the vertebrobasilar arteries, transverse and sigmoid sinuses and fourth ventricle will be large and TE-independent ([Gao & Liu 2012](https://doi.org/10.1016/j.neuroimage.2011.09.088) for the short-TR mechanism) **[prior knowledge; site-specific claim is own inference]**. The decomposition must therefore allow mixed (s, q) loadings.

### The noise model fails at late echoes and under multiband

Single-coil magnitude noise is Rician, with a positive bias at low SNR ([Gudbjartsson & Patz 1995](https://doi.org/10.1002/mrm.1910340618)). Sum-of-squares coil combination gives noncentral-χ noise ([Constantinides et al. 1997](https://doi.org/10.1002/mrm.1910380524)) **[prior knowledge]**. Near the noise floor the consequences are **[own inference]**:

- late-echo fluctuations are compressed;
- μ₄ is biased upward, which biases T2* long and ΔR2* low;
- a negative-curvature residual appears that looks like a model violation.

Where T2* is about 20 ms, TE4 keeps only about 5% of S0 **[own calc]**.

Multiband slice-GRAPPA leakage reached 26.1 (out of an input of 100) in aliased regions. It produced matched-alias false positives in 81 of 98 HCP subjects (p < 1e-9) and "can produce clusters of induced correlations in resting-state fMRI". Smoothing inflated the effect ([bioRxiv 243782](https://www.biorxiv.org/content/10.1101/243782v1.full.pdf)) **[fetched; authorship unconfirmed]**. Leaked signal keeps its source slice's TE dependence, so leaked cortical BOLD looks like BOLD in posterior fossa voxels **[own inference]**. tedana's FAQ warns that slice leakage violates the iid assumption behind its PCA estimators ([tedana FAQ](https://tedana.readthedocs.io/en/stable/faq.html)) **[fetched]**.

### "Number of BOLD components" measures the estimator, not the brain

Interpolating data to more timepoints adds no information. Yet it raised ME-ICA BOLD-component counts (t = 4.77 and 6.16), and low-pass filtering at 0.19 Hz raised them too (t = 3.43) ([Olafsson et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4408238)) **[fetched]**. robustica "almost always results in fewer components than initially prescripted". In one FAQ example, 60 PCA components gave 46 stable ones ([tedana FAQ 26.0.2](https://tedana.readthedocs.io/en/26.0.2/faq.html)) **[fetched]**. The developers suggest a fixed, slightly high `--tedpca` (for example 70) with robustica ([tedana newsletter 2025](https://groups.google.com/g/tedana-newsletter/c/j9ttG2aWDLI)) **[fetched]**. The joint decomposition inherits the same sensitivity, so counts must be reported across a dimension sweep. Variance- and reliability-based outcomes should replace counts as primary endpoints **[own inference]**.

### Pitfall table for the implementation guide

| Violation | Detection | Mitigation |
|---|---|---|
| Correlated estimation noise between a and b (r ≈ 0.9) | Off-diagonal of (XᵀWX)⁻¹; spurious mixed components in pure-noise simulations | QR whitening into c₀, c₁ (never decompose raw a, b) **[own calc/inference]** |
| Noise floor at late echoes | Per-voxel S̄_e/σ; curvature residual map | Exclude echoes with S̄/σ < 3, or Rician/noncentral-χ likelihood, or quadrature floor subtraction **[own inference]** |
| Through-slice sinc dephasing, effective-TE shifts | R² of static decay fit; curvature index against a field-gradient map | Mask or down-weight; model sinc(γ·Gz·Δz·TE/2) if a field map exists ([Fernández-Seara & Wehrli 2000](https://doi.org/10.1002/1522-2594(200009)44:3%3C358::AID-MRM3%3E3.0.CO;2-I)) **[prior knowledge]** |
| Respiratory B0 and CO2 in ΔR2* | Spectral peak at 0.25 to 0.5 Hz; regression on RVT and filtered motion | RVT/RETROICOR inside the model; global-signal arm; censoring at FD > 0.2 mm |
| Motion and spin history | Residual-energy spikes per frame; ΔS0 edge maps | Censoring; motion regressors inside the model |
| Inflow and CSF pulsation | High-φ components in vessels, fourth ventricle, edge slices | Tissue-class reporting; do not auto-reject mixed components |
| MB slice leakage | ΔR2* correlation at MB alias positions vs non-alias controls | Report; exclude alias-driven components |
| Echo-specific ghosts | Alternating-pattern residual vector; out-of-brain residual time courses | QC flag |
| Interpolation-correlated noise | Residual spatial autocorrelation | Fit and decompose in native space, unsmoothed, after one resampling |
| Dimensionality and seed dependence | Sweep AIC/KIC/MDL and fixed 50/70/90; ≥ 30 restarts | Report stability index; fixed N volumes per subject |
| Resection cavity, hemosiderin, clips | Individual dropout masks; df map | Separate cavity, peri-cavity and normal-tissue classes |

No literature covers ME-ICA or multi-echo behaviour in post-operative posterior fossa patients specifically. That gap is both a risk and a publishable contribution **[gap noted in all four notes]**.

## Thermal denoising and correct weights are the defensible SNR levers

Low-rank thermal denoising gives the largest documented gains. Moser et al. denoised each echo separately, run by run, with magnitude plus phase, before any other preprocessing. In multi-echo data they reported **tSNR gains of about 17 to 25% in adults and 16 to 18% in children**, and **split-half reliability gains of 14 to 16%** in adults. 10 to 15 min of multi-echo + NORDIC data matched the reliability of 70 min of single-echo data, and FWHM barely changed (3.47 to 3.70 mm). The benefit was smaller in motion-compromised data. They recommend saving phase and noise frames ([Moser et al., Imaging Neuroscience](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00426/125637/Multi-echo-acquisition-and-thermal-denoising)) **[fetched; sample of 7]**.

**Correction:** Dowdle et al. (2023) contain no multi-echo data. They do show that NORDIC raised target t-values from 6.16 to 9.9, about as much as 1.5-voxel smoothing (10.1). Global smoothness rose 3.7% versus 142.8% for smoothing, and NORDIC preserved beta amplitudes that smoothing altered ([Dowdle et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10234612)) **[fetched]**. Ades-Aron et al. reported MP-PCA language-mapping gains (median z from 3.8 to 4.5) in Radiology, not NeuroImage ([RSNA](https://pubs.rsna.org/doi/10.1148/radiol.2020200822)) **[fetched]**.

The risks bear directly on weak, focal components:

- **Magnitude input loses noise and adds bias.** Magnitude-only NORDIC gave about +50% tSNR against more than +100% for complex input. Every variant reduced percent signal change, and magnitude data inflated tSNR outside the brain because the Rician noise breaks the Gaussian assumption ([Knudsen et al. 2025](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1499762/full)) **[fetched]**.
- **Larger windows spread activation.** MP-PCA and NORDIC both cause activation "spreading", and larger windows spread more ([Fernandes et al. arXiv:2211.15401](https://arxiv.org/abs/2211.15401)) **[preprint]**.
- **Joint denoising cost signal.** Denoising two contrasts jointly raised tSNR but lowered t-values ([Knudsen et al. 2025](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1499762/full)).

**Default:** denoise per echo, before the fit, so per-echo noise stays independent and the residual subspace remains a valid noise reference. Tensor MP-PCA, with echo as its own mode, is the principled joint alternative ([PubMed 36219475](https://www.pubmed.gov/36219475/)) **[title-verified]**, but no mature Python implementation exists. The Brainhack ME-mppca project was at the "concept" stage in November 2025 ([Brainhack 2025](https://brainhack.org/global2025/project/project_21/)) **[fetched]**.

**Weights must come from thermal σ, not tSNR.** Physiological noise scales with signal, and the BOLD-like part scales with TE ([Krüger & Glover 2001](https://doi.org/10.1002/mrm.1240)) **[prior knowledge]**. Weights from total temporal variance would therefore down-weight the most BOLD-sensitive echoes, which is circular **[own inference]**. Estimate σ_e from noise frames, background, or the MP-PCA sigma map (dipy `mppca(..., return_sigma=True)`; [dipy reference](https://docs.dipy.org/dev/reference/dipy.denoise.html)). Fit μ_e, S0 and T2* nonlinearly on the high-SNR temporal mean.

Children's T2* (about 58 to 59 ms, against 49 to 51 ms in adults) moves BOLD weight toward echoes 3 and 4 ([Moser et al.](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00426/125637/Multi-echo-acquisition-and-thermal-denoising)) **[fetched]**. That raises the value of the 61 ms echo, which the noise table shows is also the echo that holds down the joint-fit penalty.

**Rank selection should use a calibrated null, not the asymptotic Marchenko-Pastur edge.** MP assumes iid homoscedastic noise, but multiband g-factor and sub-second-TR autocorrelation both break that. The general-noise random-matrix literature models the noise and replaces hard truncation with optimal shrinkage ([Cordero-Grande et al.](https://ar5iv.labs.arxiv.org/html/1812.05954)) **[fetched preprint]**. The null can come from noise frames, from the residual channels `Q⊥ᵀỹ` (thermal plus violation only), or from phase-randomized surrogates (temporal structure only). Shrinkage keeps weak, near-edge components at reduced weight instead of discarding them. The residual should calibrate thermal-noise eigenvalues only: physiological noise lies inside span{1, TE}, so the residual underestimates total noise **[own inference; no published use of the ME residual this way was found]**.

**Phase data, if saved, adds a physically meaningful regressor.** Fitting the phase across 4 echoes gives ΔB0(t) directly for every frame, a regressor for respiration-induced field change ([Visser et al. 2012](https://doi.org/10.1002/mrm.24119)) **[prior knowledge]**. Phase also enables complex NORDIC. To check whether phase was saved, look for dcm2niix `_ph` files or `ImageType` containing "P" in the JSON sidecars **[prior knowledge]**. Macrovascular phase regression is validated at 0.8 mm and 7T ([Stanley et al. 2021](https://pesquisa.bvsalud.org/portal/resource/en/mdl-33316391)) **[fetched]**, so it is low priority at 3.5 mm.

**Put nuisance regressors in one model.** Modular sequential regression can reintroduce artifacts ([Lindquist et al. 2019](https://www.pubmed.gov/30666750/)) **[title-verified]**. Project RETROICOR, RVT, filtered motion and ΔB0(t) out of all echoes together before the WLS, or include them as a nuisance block in the temporal design, never channel by channel afterwards **[own inference]**.

**Do not smooth.** Spatial smoothing goes against tedana's guidance ([tedana approach docs](https://tedana.readthedocs.io/en/stable/approach.html)), inflates leakage false positives, and merges small cerebellar foci with CSF and veins.

**Recover the joint-fit penalty with temporal priors.** The 800 frames at TR 0.75 s support HRF or smoothness priors on component time courses (MvME-SPFM style) and weaker shrinkage on ΔS0 where it is plausible **[own inference]**.

Recommended order **[own inference from the cited constraints]**:

1. Drop non-steady-state volumes.
2. Thermal denoising per echo (complex if phase exists), validated by injection.
3. Estimate motion on echo 1 and apply one resampling to all echoes ([Moser et al.](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00426/125637/Multi-echo-acquisition-and-thermal-denoising)).
4. Nonlinear baseline fit; σ_e and S̄/σ echo mask; df map.
5. Joint nuisance projection.
6. Log-ratio WLS and QR whitening into c₀, c₁ and the residual.
7. Calibrated rank, then joint decomposition and back-substitution.

## Exact linear algebra on GPU-resident data keeps runs cheap and reproducible

One echo at 100k voxels × 800 volumes is 320 MB in float32, and all four echoes are 1.28 GB. The full problem is "tall and skinny", so exact methods are cheap and randomized SVD buys roughly 1 to 3× at most while adding seed dependence **[own calc]**. Marchenko-Pastur thresholding also needs the full eigenvalue spectrum, which randomized methods do not give. Form the T × T Gram matrix in float64 after demeaning, because squaring the condition number in float32 corrupts exactly the small noise eigenvalues that thresholding depends on. Fix component signs with an svd_flip convention **[own inference]**.

On the CPU, sklearn FastICA accepts `whiten=False`, so PCA and whitening can be done once and the cached subspace fed to every restart ([scikit-learn FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)) **[fetched]**. Running too many BLAS threads per worker is a common slowdown. tedana's docs warn that more threads "tend to slow down performance" ([tedana usage](https://tedana.readthedocs.io/en/stable/usage.html)). Pin joblib workers with `parallel_config(inner_max_num_threads=1)` ([joblib](https://joblib.readthedocs.io/en/stable/parallel.html)) **[fetched]**. robustica uses DBSCAN by default and returns fewer components than requested. Its benchmark identifies distance computation as the slowest step and centroids as the most memory-hungry ([robustica paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-05043-9)) **[fetched]**.

On the GPU, no maintained FastICA was found. A small PyTorch version is practical **[own inference]**:

- Hold W as an (R, k, k) tensor for R restarts and keep Z resident on the device.
- Each iteration is a batched matmul, a tanh, and symmetric decorrelation via `torch.linalg.eigh`.
- Memory is about 1.2 GB for R = 30, k = 100, V = 1e5.
- The expected speedup is 10 to 100× over single-process sklearn.

TensorLy runs CP and Tucker on CUDA after `tl.set_backend('pytorch')` ([TensorLy backend guide](https://tensorly.org/stable/user_guide/backend.html)) **[fetched]**. The physics-constrained option 2 is better written as custom ALS or proximal-gradient code, with the R × R solves in float64 and a small ridge term **[own inference]**.

| Stage | Default | Approx. cost (own calc) | Device |
|---|---|---|---|
| I/O | `img.get_fdata(dtype=np.float32, caching='unchanged')` or `np.asanyarray(img.dataobj, dtype=np.float32)`; convert to uncompressed .npy for memory mapping ([nibabel](https://nipy.org/nibabel/images_and_memory.html)) | seconds | CPU |
| Thermal denoise | dipy `mppca` or patch-denoise; stride ≥ 2; batched eigh. dipy warns when the patch (125 voxels) is smaller than the number of volumes (800) ([dipy](https://docs.dipy.org/dev/reference/dipy.denoise.html); [patch-denoise](https://pypi.org/project/patch-denoise/1.4.3/)) | ~1.5e12 flops per echo at 5³ patches; minutes to hours in a Python loop | GPU batched eigh |
| Baseline and WLS | Closed-form vectorized 2 × 2 normal equations, float64 accumulation, chunks of 50 to 100 frames | ~6e9 flops | CPU |
| PCA | Gram eigh in float64 or `scipy.linalg.svd(..., lapack_driver='gesdd')`, falling back to `gesvd` on convergence error | ~6e10 flops | CPU or GPU |
| ICA | 30 to 100 restarts, `max_iter=500`, `tol=1e-4`, seeds from SeedSequence | ~1e12 flops per restart on CPU | GPU batched |
| Memory | 16 GB per subject job; memory-map arrays shared with workers | 1.3 GB raw + 1.3 GB denoised | |

Reproducibility rests on five practices:

- **Seeds.** Derive every seed from `np.random.SeedSequence(PROJECT_SEED, spawn_key=(subject, run, stage))` and log the effective seed. tedana itself changes the seed on non-convergence, up to `--maxrestart 10` ([tedana usage](https://tedana.readthedocs.io/en/stable/usage.html)) **[fetched]**.
- **Pinned versions.** Same-data differences between tedana 24.0.1 and 26.0.1 T2* outputs have been reported ([Neurostars](https://neurostars.org/t/tedana-26-0-1-t2-maps-with-extreme-outliers-compared-to-version-24-0-1/35893)) **[fetched; cause unknown]**. Pin versions with a lockfile and an Apptainer container, and record `threadpoolctl.threadpool_info()`, the CUDA version, the GPU model and the driver in provenance JSON.
- **Avoid mixed OpenMP runtimes in one process** ([threadpoolctl](https://github.com/joblib/threadpoolctl)) **[fetched]**.
- **GPU determinism settings.** Use `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and TF32 off, and the torch build for CUDA 12.4 (for example `torch==2.6.0+cu124`) **[prior knowledge; the PyTorch docs could not be fetched]**. Define reproducibility as matched components with |r| > 0.999, not bitwise identity.
- **Package and job layout.** Build a `src/` package whose CLI entry point processes one subject, cache each stage under a key of input hash + config hash + code version, and write outputs atomically with done markers. Map an LSF job array index (`$LSB_JOBINDEX`) to subject, and add a dependent group step. GPU jobs should fit within gpu_short's 30-minute limit using checkpoints **[prior knowledge for LSF syntax; check local docs]**.

## Six tests decide whether the method beats tedana

Comparisons must be fair. The new method should be benchmarked against tedana as currently strengthened, not against default single-run FastICA alone ([Robust-tedana](https://www.biorxiv.org/content/10.1101/2025.06.16.660050.full.pdf)). The benchmark arms are:

- A: tedana defaults;
- B: tedana with robustica at 30 runs and a fixed `--tedpca` sweep;
- C: Robust-tedana-style MP-PCA + 50 robustica runs + no PCA truncation;
- D: OC only;
- E: OC + RVT/GSR;
- F: the new method.

Map tedana's classification tags onto the physics labels component by component as a concordance check.

| Test | Construction | Pass criterion (proposed, own inference) |
|---|---|---|
| 1. Synthetic recovery | Echoes from known s, q maps and time courses at the real TEs, T2* 20 to 70 ms, complex noise then magnitude; include pure-noise and mixed sources | Pure-noise data yields no components with φ outside the null; known (s, q) recovered with map |r| > 0.9; no spurious mixed components |
| 2. Injection | Add focal ΔR2*-only and ΔS0-only signals to real raw echoes before denoising | Amplitude, extent and channel assignment preserved through the full pipeline (guards against spreading and shrinkage) |
| 3. Stability | ≥ 30 seeds × dimensions {AIC, KIC, MDL, 50, 70, 90} × channel weightings | Stability index reported; component sets matched by Hungarian assignment on |corr| |
| 4. Physics referee | pBOLD and echo-pair FC geometry ([PubMed 41929219](https://pubmed.ncbi.nlm.nih.gov/41929219/)) | pBOLD ≥ the tedana arms at equal or better tSNR |
| 5. Artifact control | QC-FC and distance dependence ([Power 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5834724)); fraction of ΔR2* variance explained by RVT/RETROICOR; MB alias test | Lower QC-FC than tedana; physiological variance reported separately from residual "neural" variance |
| 6. Equivalence sanity | Decompose c₀ alone | Reproduces a tedana-like OC decomposition (validates the whitening code) |

Primary outcomes should be the variance fractions in c₀, c₁ and the residual per tissue class, the ΔR2* variance left after physiological and global regressors, accepted-versus-rejected variance with seed spread, and split-half reliability. Counts of BOLD components should not be a primary outcome, because Olafsson et al. showed that counts track timepoints and filtering ([Olafsson et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4408238)). Keep a constant number of volumes per participant after censoring, by truncating to a common N **[own inference]**.

Corrections and open checks the guide must carry:

- ME-SPFM is NeuroImage 2019, not HBM.
- Dowdle 2023 has no multi-echo data; Moser et al. is the multi-echo NORDIC evidence.
- Ades-Aron 2021 is Radiology.
- The Speck and Hennig DOI suffix (…0210 vs …0211) conflicts between notes.
- The kappa/rho weighting is the coordinator-verified OC-beta form, not the recalled z-map form.
- The decorrelating reference TE is the weighted mean (20 to 31 ms), not 38.7 ms.
- The "2.8 to 3.2×" three-echo figure is relative to the four-echo ΔS0 = 0 baseline.
- Still unverified: the MvME-SPFM final venue, the tensor MP-PCA author list and code, the NORDIC patch ratio, the PyTorch determinism flags, and the CMRR per-echo slice-GRAPPA kernel behaviour.
- Still unknown: the CSIMemphis acquisition details (phase saved, noise frames, partial Fourier, PE direction, MB factor).

## Conclusion

Reframing the problem changes how it should be judged. tedana's kappa and rho already estimate each component's S0/R2* character from pooled voxels. The real novelty is giving the decomposition a correctly whitened ΔS0-only channel, so that mixed sources get two maps and an uncertainty instead of one binary label. The construction contains tedana as a special case, and the 2.2 to 2.6× voxelwise penalty becomes an argument for the design rather than against it: it shows why the separation has to happen at the component level. It also argues for keeping the 61 ms echo, especially in children, whose longer T2* puts more BOLD weight on late echoes.

The harder, more publishable question is what the separated channels reveal in the posterior fossa. That region combines inflow, CSF pulsation, skull-base field gradients, MB leakage and post-surgical cavities, and none of these has been characterized for multi-echo methods. Collecting physiological recordings and checking whether phase and noise frames exist are the two acquisition-side steps that most determine whether the method can make claims about neural signal or only about TE dependence. A component with a clean ΔR2*-only map can still be respiratory CO2. Physics separation narrows the interpretation; it does not settle it.
