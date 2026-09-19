"""Nośność stopy — wzory jak SYNAJv2 (drained)."""

from __future__ import annotations

import math


def bearing_factors(phi_deg: float) -> tuple[float, float, float]:
    fi = math.radians(float(phi_deg))
    if abs(math.tan(fi)) < 1e-12:
        raise ValueError("φ zbyt małe")
    N_q = math.exp(math.pi * math.tan(fi)) * (math.tan(math.pi / 4 + fi / 2)) ** 2
    N_c = (N_q - 1.0) / math.tan(fi)
    N_gamma = 2.0 * (N_q + 1.0) * math.tan(fi)
    return N_c, N_q, N_gamma


def shape_depth_inclination(
    phi_deg: float,
    B_prime: float,
    L_prime: float,
    Df: float,
    N: float,
    T: float = 0.0,
    use_depth: bool = True,
) -> dict:
    fi = math.radians(phi_deg)
    N_c, N_q, N_gamma = bearing_factors(phi_deg)

    s_q = 1.0 + (B_prime / L_prime) * math.sin(fi)
    s_gamma = 1.0 - 0.3 * (B_prime / L_prime)
    s_c = (s_q * N_q - 1.0) / (N_q - 1.0) if N_q != 1.0 else 1.0

    if use_depth and B_prime > 0:
        ratio = Df / B_prime
        if ratio <= 1.0:
            d_q = 1.0 + 2.0 * math.tan(fi) * (1.0 - math.sin(fi)) ** 2 * ratio
        else:
            d_q = 1.0 + 2.0 * math.tan(fi) * (1.0 - math.sin(fi)) ** 2 * math.atan(ratio)
        d_gamma = 1.0
        d_c = (
            d_q - (1.0 - d_q) / (N_c * math.tan(fi))
            if abs(N_c * math.tan(fi)) > 1e-12
            else 1.0
        )
    else:
        d_c = d_q = d_gamma = 1.0

    m = (2.0 + B_prime / L_prime) / (1.0 + B_prime / L_prime)
    if N <= 0:
        i_q = i_gamma = i_c = 0.0
    else:
        ratio_tn = max(0.0, min(1.0, T / N))
        i_q = (1.0 - ratio_tn) ** m
        i_gamma = (1.0 - ratio_tn) ** (m + 1.0)
        i_c = i_q - (1.0 - i_q) / (N_q - 1.0) if N_q != 1.0 else i_q

    return {
        "N_c": N_c,
        "N_q": N_q,
        "N_gamma": N_gamma,
        "s_c": s_c,
        "s_q": s_q,
        "s_gamma": s_gamma,
        "d_c": d_c,
        "d_q": d_q,
        "d_gamma": d_gamma,
        "i_c": i_c,
        "i_q": i_q,
        "i_gamma": i_gamma,
        "b_c": 1.0,
        "b_q": 1.0,
        "b_gamma": 1.0,
        "g_c": 1.0,
        "g_q": 1.0,
        "g_gamma": 1.0,
        "m": m,
    }


def resistance_Rk(
    *,
    phi_deg: float,
    c_prime: float,
    gamma: float,
    B: float,
    L: float,
    Df: float,
    V: float,
    q_N: float = 0.0,
    T: float = 0.0,
    use_depth: bool = True,
) -> tuple[float, dict]:
    """R_k [kN], z odpływem; B'=B, L'=L (MVP bez mimośrodu)."""
    B_p, L_p = float(B), float(L)
    q = float(q_N) + float(gamma) * float(Df)
    f = shape_depth_inclination(phi_deg, B_p, L_p, Df, V, T, use_depth)
    A = B_p * L_p
    q_R = (
        c_prime * f["N_c"] * f["s_c"] * f["i_c"] * f["d_c"]
        + q * f["N_q"] * f["s_q"] * f["i_q"] * f["d_q"]
        + 0.5 * gamma * B_p * f["N_gamma"] * f["s_gamma"] * f["i_gamma"] * f["d_gamma"]
    )
    R_k = A * q_R
    f.update({"q_R": q_R, "A_prime": A, "R_k": R_k, "q": q})
    return R_k, f
