"""
μ, σ (opis „normalny”) → parametry lognormalne + Monte Carlo → β.

    δ = σ/μ
    σ_ln = √ln(1+δ²)
    μ_ln = ln(μ) − ½ σ_ln²

Zmienne losowe (niezależne): φ′, c′, γ — lognormalne.
β = −Φ⁻¹(p_f),  p_f = P(R_k < V)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from bearing import resistance_Rk


@dataclass(frozen=True)
class LogNormalParams:
    mu: float
    sigma: float
    mu_ln: float
    sigma_ln: float
    cov: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.sigma_ln <= 0:
            return np.full(n, self.mu)
        return rng.lognormal(self.mu_ln, self.sigma_ln, size=n)


def normal_moments_to_lognormal(mu: float, sigma: float) -> LogNormalParams:
    mu = float(mu)
    sigma = float(sigma)
    if mu <= 0:
        raise ValueError("Dla lognormalnej średnia μ musi być > 0.")
    if sigma < 0:
        raise ValueError("σ ≥ 0")
    if sigma == 0:
        return LogNormalParams(mu, 0.0, math.log(mu), 0.0, 0.0)
    cov = sigma / mu
    sigma_ln = math.sqrt(math.log(1.0 + cov * cov))
    mu_ln = math.log(mu) - 0.5 * sigma_ln * sigma_ln
    return LogNormalParams(mu, sigma, mu_ln, sigma_ln, cov)


def recommend_n(beta_target: float = 3.8, n_fail_min: int = 30) -> dict:
    pf = float(stats.norm.cdf(-beta_target))
    n_sug = int(math.ceil(n_fail_min / max(pf, 1e-16)))
    return {
        "beta_target": beta_target,
        "p_f_target": pf,
        "n_fail_min": n_fail_min,
        "n_suggested": n_sug,
        "note": (
            f"Dla β≈{beta_target:.1f} → p_f≈{pf:.2e}. "
            f"Aby mieć ~{n_fail_min} awarii: N ≳ {n_sug:,}. "
            "Zgrubnie (β±0,2): 5–20 tys. "
            "Do ogona przy β>3,5: dziesiątki–setki tysięcy (lub FORM/IS)."
        ),
    }


def beta_from_pf(p_f: float) -> float:
    p = min(max(float(p_f), 1e-16), 1.0 - 1e-16)
    return float(-stats.norm.ppf(p))


def interpret_beta(beta: float, beta_kind: str, n_fail: int, n_ok: int) -> list[str]:
    """Krótkie, ludzkie wyjaśnienie wyniku β / p_f."""
    lines: list[str] = []
    if beta_kind == "lower_bound":
        lines.append(
            f"**Brak awarii** w {n_ok:,} próbach → nie znamy dokładnego β, tylko "
            f"**dolną granicę** β ≳ {beta:.2f} (z 1/(N+1))."
        )
        lines.append(
            "Żeby zobaczyć β „z ogona”, zwiększ N albo zwiększ zmienność / obciążenie V "
            "(albo użyj Subset Simulation)."
        )
        return lines

    pf = float(stats.norm.cdf(-beta))
    lines.append(
        f"**β = {beta:.2f}** ↔ **p_f ≈ {pf:.2e}** "
        f"(β = −Φ⁻¹(p_f); większe β = rzadsza awaria)."
    )
    if beta < 2.0:
        lines.append("β < 2 — **bardzo niska** niezawodność w tym modelu.")
    elif beta < 3.0:
        lines.append("β ≈ 2…3 — umiarkowana; często za mało na typowe wymagania.")
    elif beta < 3.5:
        lines.append("β ≈ 3…3,5 — dolny zakres często spotykany w praktyce.")
    elif beta < 4.2:
        lines.append(
            "β ≈ 3,5…4,2 — typowy zakres docelowy "
            "(orientacyjnie rząd β≈3,8 przy niektórych założeniach CC2 / 50 lat)."
        )
    else:
        lines.append(
            "β > 4,2 — **wysoka** niezawodność w tym modelu "
            "(albo za mało awarii w próbie przy CMC)."
        )
    lines.append(
        f"W tej próbie: **{n_fail} awarii / {n_ok:,}** poprawnych realizacji."
    )
    return lines


def run_monte_carlo(
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
    n: int = 20_000,
    seed: int = 42,
    use_depth: bool = True,
    T: float = 0.0,
) -> dict:
    rng = np.random.default_rng(int(seed))
    n = int(max(200, n))

    phi_ln = normal_moments_to_lognormal(phi_mu, phi_sd)
    phi_s = np.clip(phi_ln.sample(n, rng), 1.0, 55.0)

    gamma_ln = normal_moments_to_lognormal(gamma_mu, max(gamma_sd, 0.0))
    gamma_s = np.clip(gamma_ln.sample(n, rng), 10.0, 30.0)

    if c_mu <= 0:
        c_s = np.zeros(n)
        c_ln = None
    else:
        c_ln = normal_moments_to_lognormal(c_mu, max(c_sd, 0.0))
        c_s = np.clip(c_ln.sample(n, rng), 0.0, None)

    R = np.empty(n, dtype=float)
    for i in range(n):
        try:
            R[i], _ = resistance_Rk(
                phi_deg=float(phi_s[i]),
                c_prime=float(c_s[i]),
                gamma=float(gamma_s[i]),
                B=B,
                L=L,
                Df=Df,
                V=V,
                q_N=q_N,
                T=T,
                use_depth=use_depth,
            )
        except Exception:
            R[i] = np.nan

    ok = np.isfinite(R) & (R > 0)
    R_ok = R[ok]
    if R_ok.size < 50:
        raise RuntimeError("Za mało poprawnych realizacji — sprawdź dane.")

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

    G = R_ok - float(V)
    n_fail = int(np.sum(G < 0))
    p_f = n_fail / float(R_ok.size)

    if n_fail == 0:
        beta = beta_from_pf(1.0 / (R_ok.size + 1))
        beta_kind = "lower_bound"
        beta_note = (
            f"Brak awarii w próbie → nie znamy dokładnego β, tylko dolną granicę "
            f"β ≳ {beta:.2f} (z 1/(N+1))."
        )
    else:
        beta = beta_from_pf(p_f)
        beta_kind = "from_pf"
        beta_note = (
            f"β policzone z częstości awarii: β = −Φ⁻¹(p_f), "
            f"p_f = {n_fail}/{R_ok.size} = {p_f:.4e}."
        )

    mu_G = float(np.mean(G))
    sd_G = float(np.std(G, ddof=1))
    beta_moments = mu_G / sd_G if sd_G > 1e-12 else float("inf")

    se_pf = math.sqrt(p_f * (1.0 - p_f) / R_ok.size)
    if n_fail > 0 and math.isfinite(beta):
        dens = float(stats.norm.pdf(-beta))
        se_beta = se_pf / dens if dens > 1e-16 else float("nan")
    else:
        se_beta = float("nan")

    pctl = {
        "p5": float(np.percentile(R_ok, 5)),
        "p50": float(np.percentile(R_ok, 50)),
        "p95": float(np.percentile(R_ok, 95)),
    }

    interpret = interpret_beta(beta, beta_kind, n_fail, int(R_ok.size))
    interpret.insert(
        0,
        f"**Metoda: CMC** (klasyczne Monte Carlo) — {int(R_ok.size):,} realizacji.",
    )

    return {
        "method": "CMC",
        "method_label": "CMC (Crude Monte Carlo)",
        "n": n,
        "n_ok": int(R_ok.size),
        "n_fail": n_fail,
        "n_eval": int(R_ok.size),
        "R": R_ok,
        "phi_ln": phi_ln,
        "c_ln": c_ln,
        "gamma_ln": gamma_ln,
        "R_det": R_det,
        "f_det": f_det,
        "R_mean": float(np.mean(R_ok)),
        "R_std": float(np.std(R_ok, ddof=1)),
        "R_p5": pctl["p5"],
        "R_p50": pctl["p50"],
        "R_p95": pctl["p95"],
        "V": float(V),
        "p_f": p_f,
        "se_pf": se_pf,
        "cov_pf": float("nan"),
        "beta": beta,
        "beta_kind": beta_kind,
        "beta_note": beta_note,
        "beta_moments": beta_moments,
        "se_beta": se_beta,
        "FS_det": float(R_det / V) if V > 0 else float("inf"),
        "FS_p5": float(pctl["p5"] / V) if V > 0 else float("inf"),
        "interpret": interpret,
        "recommend": recommend_n(3.8, 30),
        "ss_p0": None,
        "ss_levels": None,
        "ss_p_conds": None,
        "ss_thresholds": None,
        "ss_n_seed": None,
        "ss_accept_rate": None,
        "ss_prop_sd": None,
    }
