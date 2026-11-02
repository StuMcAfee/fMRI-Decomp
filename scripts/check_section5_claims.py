"""Independent numerical re-check of CLAUDE.md section 5 (inflation table, properties 1, 3, 4,
whitened-noise covariance vs SNR, sigma estimator). Run: python scripts/check_section5_claims.py"""
import numpy as np
rng = np.random.default_rng(0)
TE = np.array([16.20, 31.22, 46.24, 61.26]); E = 4

def frontend(mu, sigma):
    D = np.column_stack([-TE, np.ones(E), TE**2, (-1.0)**np.arange(E)])
    Xw = (mu/sigma)[:, None]*D
    Q, R = np.linalg.qr(Xw, mode="complete")
    sgn = np.sign(np.diag(R)[:2]); Q[:, :2] *= sgn; R[:2, :] *= sgn[:, None]
    return Q, R

print("== 5.2 inflation / corr ==")
for T2 in [20, 30, 45, 60, 70]:
    mu = np.exp(-TE/T2); W = np.diag(mu**2)
    X = np.column_stack([-TE, np.ones(E)])
    C = np.linalg.inv(X.T@W@X); base = 1/(TE**2@mu**2)
    X3 = X[:3]; C3 = np.linalg.inv(X3.T@np.diag(mu[:3]**2)@X3)
    print(T2, "infl %.2f" % np.sqrt(C[0,0]/base), "corr %.2f" % (C[0,1]/np.sqrt(C[0,0]*C[1,1])),
          "3echo %.2f" % np.sqrt(C3[0,0]/base))

print("== property 1, 3, 4 ==")
T2 = 45.; S0 = 1000.; sigma = 10.
mu = S0*np.exp(-TE/T2); Q, R = frontend(mu, sigma)
print("R01<0:", R[0,1] < 0)
b = 0.001; a = 0.0
y = a - TE*b; yw = mu/sigma*y
c = Q[:, :2].T@yw; r = Q[:, 2:].T@yw
print("pure R2*: c1=%.2e r=%s" % (c[1], r))
w = TE*np.exp(-TE/T2)
dS = mu*(rng.normal(size=E))   # arbitrary linear-form deviation
c0 = Q[:, 0]@((mu/sigma)*(dS/mu)); oc = (w@dS)/w.sum()
print("c0/OC ratio (should be negative constant):", c0/oc, " second draw:",
      (lambda d: (Q[:,0]@((mu/sigma)*(d/mu)))/((w@d)/w.sum()))(mu*rng.normal(size=E)))
for a, b in [(0.01, 0.0), (0.0, 0.001), (0.01, 0.001), (0.01, -0.001)]:
    c = Q[:, :2].T@(mu/sigma*(a - TE*b)); s = c[1]/R[1,1]; q = (c[0]-R[0,1]*s)/R[0,0]
    print("a,b=", a, b, "-> s,q=", round(s, 6), round(q, 7))

print("== whitened noise cov vs SNR of last echo (Rician, log-ratio, true mu & sigma) ==")
T = 200000
for T2 in [20, 45, 70]:
    for snr4 in [3, 5, 10, 20]:
        sigma = 1.0; S0 = snr4*sigma*np.exp(TE[-1]/T2); mu_true = S0*np.exp(-TE/T2)
        Sc = mu_true[:, None] + sigma*(rng.normal(size=(E, T)) + 1j*rng.normal(size=(E, T)))
        S = np.abs(Sc)
        mu_emp = S.mean(1)               # empirical temporal mean (Rician-biased)
        Q, R = frontend(mu_emp, sigma)
        y = np.log(S/mu_emp[:, None]); yw = (mu_emp/sigma)[:, None]*y
        z = Q.T@yw; z -= z.mean(1, keepdims=True)
        Cz = np.cov(z)
        # sigma estimate: log-form vs linear-form residual
        Qp = np.linalg.qr(mu_emp[:, None]*np.column_stack([-TE, np.ones(E), TE**2, (-1.0)**np.arange(E)]), mode="complete")[0][:, 2:]
        ulog = Qp.T@(mu_emp[:, None]*y); ulog -= ulog.mean(1, keepdims=True)
        ulin = Qp.T@(S - mu_emp[:, None])
        s_log = np.sqrt(np.median((ulog**2).sum(0))/(2*np.log(2)))
        s_lin = np.sqrt(np.median((ulin**2).sum(0))/(2*np.log(2)))
        print(f"T2*={T2} mu4/sig={snr4:>2}: diag={np.round(np.diag(Cz),3)} max|offdiag|={np.abs(Cz-np.diag(np.diag(Cz))).max():.3f}  sigma_hat log={s_log:.3f} lin={s_lin:.3f}")
