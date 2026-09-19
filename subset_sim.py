"""
Subset Simulation (Au & Beck) — rzadkie awarie R_k < V.

Przestrzeń u ~ N(0,I); φ′, γ, (c′) przez transformację lognormalną.
MCMC: Modified Metropolis (MMA), warunek g = R_k − V ≤ b_i.
p_f = ∏ P(F_i | F_{i−1}),  β = −Φ⁻¹(p_f).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from bearing import resistance_Rk
from distributions import (
    beta_from_pf,
    interpret_beta,
    normal_moments_to_lognormal,
    recommend_n,
)


def _pack_ln(phi_mu, phi_sd, gamma_mu, gamma_sd, c_mu, c_sd):
    phi_ln = normal_moments_to_lognormal(phi_mu, phi_sd)
    gamma_ln = normal_moments_to_lognormal(gamma_mu, max(gamma_sd, 0.0))
    c_ln = (
        None
        if c_mu <= 0
        else normal_moments_to_lognormal(c_mu, max(c_sd, 0.0))
    )
    return phi_ln, gamma_ln, c_ln


def _u_to_physical(u: np.ndarray, phi_ln, gamma_ln, c_ln) -> tuple[float, float, float]:
    phi = float(
        np.clip(math.exp(phi_ln.mu_ln + phi_ln.sigma_ln * float(u[0])), 1.0, 55.0)
    )
    gam = float(
        np.clip(math.exp(gamma_ln.mu_ln + gamma_ln.sigma_ln * float(u[1])), 10.0, 30.0)
    )
    if c_ln is None:
        c = 0.0
    else:
        c = float(max(0.0, math.exp(c_ln.mu_ln + c_ln.sigma_ln * float(u[2]))))
    return phi, gam, c


def _eval_g(u: np.ndarray, *, phi_ln, gamma_ln, c_ln, B, L, Df, V, q_N, T, use_depth):
    """(g = R−V, R). Błąd → g = +∞."""
    phi, gam, c = _u_to_physical(u, phi_ln, gamma_ln, c_ln)
    try:
        R, _ = resistance_Rk(
            phi_deg=phi,
            c_prime=c,
            gamma=gam,
            B=B,
            L=L,
            Df=Df,
            V=V,
            q_N=q_N,
            T=T,
            use_depth=use_depth,
        )
        if not math.isfinite(R) or R <= 0:
            return float("inf"), float("nan")
        return float(R - V), float(R)
    except Exception:
        return float("inf"), float("nan")


def _mma_step(u, g, R, b, rng, prop_sd, eval_kw):
    """Modified Metropolis w u; kandydat tylko gdy g' ≤ b."""
    dim = u.size
    u_cand = u.copy()
    for j in range(dim):
        uj = float(u[j])
        prop = uj + prop_sd * float(rng.standard_normal())
        log_a = -0.5 * (prop * prop - uj * uj)
        if log_a >= 0.0 or math.log(rng.random()) < log_a:
            u_cand[j] = prop

    if np.allclose(u_cand, u):
        return u, g, R, False

    g_new, R_new = _eval_g(u_cand, **eval_kw)
    if g_new <= b:
        return u_cand, g_new, R_new, True
    return u, g, R, False


def run_subset_simulation(
    *,
    B: float,
    L: float,
    Df: float,
    V: float,
    gamma_mu: float,
    gamma_sd: float,
    q_N: float,
    phi_mu: float,
    phi_sd: float,
    c_mu: float,
    c_sd: float,
    n: int = 2000,
    p0: float = 0.1,
    seed: int = 42,
    use_depth: bool = True,
    T: float = 0.0,
    max_levels: int = 15,
    prop_sd: float = 1.0,
) -> dict:
    rng = np.random.default_rng(int(seed))
    n = int(max(200, n))
    p0 = float(min(max(p0, 0.05), 0.5))
    n_seed = max(1, int(round(p0 * n)))
    while n % n_seed != 0 and n_seed > 1:
        n_seed -= 1
    chain_len = n // n_seed

    phi_ln, gamma_ln, c_ln = _pack_ln(phi_mu, phi_sd, gamma_mu, gamma_sd, c_mu, c_sd)
    dim = 2 if c_ln is None else 3
    eval_kw = dict(
        phi_ln=phi_ln,
        gamma_ln=gamma_ln,
        c_ln=c_ln,
        B=B,
        L=L,
        Df=Df,
        V=float(V),
        q_N=q_N,
        T=T,
        use_depth=use_depth,
    )

    R_det, f_det = resistance_Rk(
        phi_deg=phi_mu,
        c_prime=max(c_mu, 0.0),
        gamma=gamma_mu,
        B=B,
        L=L,
        Df=Df,
        V=V,
        q_N=q_N,
        T=T,
        use_depth=use_depth,
    )

    U = rng.standard_normal((n, dim))
    G = np.empty(n)
    Rv = np.empty(n)
    for i in range(n):
        G[i], Rv[i] = _eval_g(U[i], **eval_kw)

    levels: list[dict] = []
    p_conds: list[float] = []
    thresholds: list[float] = []
    n_eval = n
    n_accept = 0
    n_prop = 0

    level = 0
    finished = False
    while level < max_levels and not finished:
        order = np.argsort(G)
        G_s, U_s, R_s = G[order], U[order], Rv[order]
        n_fail_here = int(np.sum(G_s <= 0.0))

        if n_fail_here >= n_seed:
            p_last = n_fail_here / float(n)
            p_conds.append(p_last)
            thresholds.append(0.0)
            levels.append(
                {
                    "level": level,
                    "b": 0.0,
                    "p_cond": p_last,
                    "n_fail": n_fail_here,
                    "final": True,
                }
            )
            U, G, Rv = U_s, G_s, R_s
            finished = True
            break

        b = float(G_s[n_seed - 1])
        if b <= 0.0:
            p_last = max(n_fail_here / float(n), 1.0 / (n + 1))
            p_conds.append(p_last)
            thresholds.append(0.0)
            levels.append(
                {
                    "level": level,
                    "b": 0.0,
                    "p_cond": p_last,
                    "n_fail": n_fail_here,
                    "final": True,
                }
            )
            U, G, Rv = U_s, G_s, R_s
            finished = True
            break

        p_conds.append(p0)
        thresholds.append(b)
        levels.append(
            {
                "level": level,
                "b": b,
                "p_cond": p0,
                "n_seeds": n_seed,
                "final": False,
            }
        )

        seeds_U = U_s[:n_seed].copy()
        seeds_G = G_s[:n_seed].copy()
        seeds_R = R_s[:n_seed].copy()

        U_new = np.empty((n, dim))
        G_new = np.empty(n)
        R_new = np.empty(n)
        idx = 0
        for s in range(n_seed):
            u_cur = seeds_U[s].copy()
            g_cur = float(seeds_G[s])
            r_cur = float(seeds_R[s])
            for _ in range(chain_len):
                u_cur, g_cur, r_cur, acc = _mma_step(
                    u_cur, g_cur, r_cur, b, rng, prop_sd, eval_kw
                )
                n_prop += 1
                n_accept += int(acc)
                n_eval += 1
                U_new[idx] = u_cur
                G_new[idx] = g_cur
                R_new[idx] = r_cur
                idx += 1

        U, G, Rv = U_new, G_new, R_new
        level += 1

    if not finished:
        n_fail_here = int(np.sum(G <= 0.0))
        p_last = max(n_fail_here / float(n), 1.0 / (n + 1))
        p_conds.append(p_last)
        thresholds.append(0.0)
        levels.append(
            {
                "level": level,
                "b": 0.0,
                "p_cond": p_last,
                "n_fail": n_fail_here,
                "final": True,
                "truncated": True,
            }
        )

    p_f = float(np.prod(p_conds))
    p_f = min(max(p_f, 1e-16), 1.0 - 1e-16)
    beta = beta_from_pf(p_f)
    n_levels = len(p_conds)
    beta_kind = "from_pf"
    beta_note = (
        f"Subset Simulation: p_f = ∏ p_i = {p_f:.4e} "
        f"({n_levels} poziom(ów), p0={p0:g}, N/poziom={n:,}). "
        f"β = −Φ⁻¹(p_f)."
    )

    var_terms = [(1.0 - pi) / (pi * n) for pi in p_conds if pi > 0]
    cov_pf = math.sqrt(sum(var_terms)) if var_terms else float("nan")
    se_pf = cov_pf * p_f if math.isfinite(cov_pf) else float("nan")
    dens = float(stats.norm.pdf(-beta))
    se_beta = (se_pf / dens) if dens > 1e-16 and math.isfinite(se_pf) else float("nan")

    R_ok = Rv[np.isfinite(Rv) & (Rv > 0)]
    if R_ok.size < 10:
        raise RuntimeError("Subset Simulation: za mało poprawnych R_k — sprawdź dane.")

    G_ok = R_ok - float(V)
    mu_G = float(np.mean(G_ok))
    sd_G = float(np.std(G_ok, ddof=1)) if R_ok.size > 1 else float("nan")
    beta_moments = mu_G / sd_G if sd_G and sd_G > 1e-12 else float("inf")

    n_fail_last = int(np.sum(G <= 0.0))
    interpret = interpret_beta(beta, beta_kind, n_fail_last, n)
    interpret.insert(
        0,
        f"**Metoda: Subset Simulation** — {n_levels} poziom(ów), "
        f"łącznie ~{n_eval:,} wywołań nośności (p0={p0:g}).",
    )

    pctl = {
        "p5": float(np.percentile(R_ok, 5)),
        "p50": float(np.percentile(R_ok, 50)),
        "p95": float(np.percentile(R_ok, 95)),
    }
    accept_rate = n_accept / n_prop if n_prop else float("nan")

    return {
        "method": "SS",
        "method_label": "Subset Simulation",
        "n": n,
        "n_ok": int(R_ok.size),
        "n_fail": n_fail_last,
        "n_eval": n_eval,
        "R": R_ok,
        "phi_ln": phi_ln,
        "c_ln": c_ln,
        "gamma_ln": gamma_ln,
        "R_det": R_det,
        "f_det": f_det,
        "R_mean": float(np.mean(R_ok)),
        "R_std": float(np.std(R_ok, ddof=1)) if R_ok.size > 1 else 0.0,
        "R_p5": pctl["p5"],
        "R_p50": pctl["p50"],
        "R_p95": pctl["p95"],
        "V": float(V),
        "p_f": p_f,
        "se_pf": se_pf,
        "cov_pf": cov_pf,
        "beta": beta,
        "beta_kind": beta_kind,
        "beta_note": beta_note,
        "beta_moments": beta_moments,
        "se_beta": se_beta,
        "FS_det": float(R_det / V) if V > 0 else float("inf"),
        "FS_p5": float(pctl["p5"] / V) if V > 0 else float("inf"),
        "interpret": interpret,
        "recommend": recommend_n(3.8, 30),
        "ss_p0": p0,
        "ss_levels": levels,
        "ss_p_conds": p_conds,
        "ss_thresholds": thresholds,
        "ss_n_seed": n_seed,
        "ss_accept_rate": accept_rate,
        "ss_prop_sd": prop_sd,
    }
