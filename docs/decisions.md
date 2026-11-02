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
