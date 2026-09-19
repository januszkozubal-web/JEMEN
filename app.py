# JEMEN — stopa + CMC / Subset Simulation → β
# Author: jvk · MIT

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from bearing import resistance_Rk
from distributions import normal_moments_to_lognormal, recommend_n, run_monte_carlo
from subset_sim import run_subset_simulation

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.1rem; max-width: 1100px; }
.beta-big { font-size: 2.8rem; font-weight: 700; color: #5BA3D9; margin: 0.15rem 0 0.4rem 0; }
.box-ok { background:#1a2e24; border-left:4px solid #3c9; padding:0.75rem 1rem; border-radius:4px; }
.box-warn { background:#2e2a1a; border-left:4px solid #e8a838; padding:0.75rem 1rem; border-radius:4px; }
</style>
"""


def _ln_line(name: str, p) -> str:
    return (
        f"**{name}:** μ={p.mu:.3g}, SD={p.sigma:.3g} → "
        f"CoV={p.cov:.3f}, μ_ln={p.mu_ln:.4f}, σ_ln={p.sigma_ln:.4f}"
    )


def _ln_md_row(name: str, p) -> str:
    if p is None:
        return f"| {name} | — | — | — | — | — |"
    return (
        f"| {name} | {p.mu:.6g} | {p.sigma:.6g} | {p.cov:.4f} | "
        f"{p.mu_ln:.6f} | {p.sigma_ln:.6f} |"
    )


def results_to_markdown(res: dict, inp: dict) -> str:
    se_b = res["se_beta"]
    se_b_txt = f"{se_b:.4f}" if np.isfinite(se_b) else "—"
    interpret = "\n".join(f"- {x}" for x in res["interpret"])
    if res["c_ln"] is not None:
        c_row = _ln_md_row("c′", res["c_ln"])
    else:
        c_row = (
            f"| c′ | {inp['c_mu']:.6g} | {inp['c_sd']:.6g} | (stałe 0) | — | — |"
        )

    method = res.get("method_label", res.get("method", "?"))
    ss_block = ""
    if res.get("method") == "SS" and res.get("ss_levels"):
        rows = []
        for lv in res["ss_levels"]:
            rows.append(
                f"| {lv['level']} | {lv['b']:.4g} | {lv['p_cond']:.4g} | "
                f"{'tak' if lv.get('final') else 'nie'} |"
            )
        acc = res.get("ss_accept_rate", float("nan"))
        acc_txt = f"{acc:.3f}" if np.isfinite(acc) else "—"
        n_ev = res.get("n_eval", 0)
        ss_block = f"""
## Subset Simulation — poziomy

| poziom | próg b (g=R−V) | p_cond | finał |
|--------|----------------|--------|-------|
{chr(10).join(rows)}

- p0 = {res.get('ss_p0')}
- akceptacja MMA ≈ {acc_txt}
- wywołań nośności ≈ {n_ev:,}
"""

    return f"""# JEMEN — zestaw wyników

Wersja testowa — nie do dokumentacji projektowej.

**Metoda:** {method}

## Dane wejściowe

| Wielkość | Wartość |
|----------|---------|
| B [m] | {inp['B']:.4g} |
| L [m] | {inp['L']:.4g} |
| D_f [m] | {inp['Df']:.4g} |
| V [kN] (stałe) | {inp['V']:.4g} |
| q_N [kPa] | {inp['q_N']:.4g} |
| współczynnik głębokości d | {"tak" if inp['use_depth'] else "nie"} |
| N (CMC) / N na poziom (SS) | {inp['n']:,} |
| ziarno RNG | {inp['seed']} |
| p0 (SS) | {inp.get('p0', '—')} |

### Grunt — μ, SD → lognormal

| Zmienna | μ | SD | CoV | μ_ln | σ_ln |
|---------|---|----|-----|------|------|
{_ln_md_row("γ", res["gamma_ln"])}
{_ln_md_row("φ′", res["phi_ln"])}
{c_row}

## 1) Indeks niezawodności β

- **β = {res['beta']:.4f}** ({res['beta_kind']})
- {res['beta_note']}

### Interpretacja

{interpret}
{ss_block}
## 2) Prawdopodobieństwo awarii

| Wielkość | Wartość |
|----------|---------|
| p_f = P(R_k < V) | {res['p_f']:.6e} |
| awarie (ostatni poziom / CMC) | {res['n_fail']} / {res['n_ok']:,} |
| SE(p_f) | {res['se_pf']:.4e} |
| SE(β) | {se_b_txt} |
| β (momenty G = R−V) | {res['beta_moments']:.4f} |

## 3) Nośność R_k

| Wielkość | [kN] |
|----------|------|
| V | {res['V']:.2f} |
| R_k na średnich | {res['R_det']:.2f} |
| średnia R_k | {res['R_mean']:.2f} |
| odchylenie std R_k | {res['R_std']:.2f} |
| 5. percentyl R_k | {res['R_p5']:.2f} |
| mediana R_k | {res['R_p50']:.2f} |
| 95. percentyl R_k | {res['R_p95']:.2f} |
| FS = R(μ)/V | {res['FS_det']:.3f} |
| FS₅ = R₅/V | {res['FS_p5']:.3f} |

## Uwagi

- Awaria w modelu: **R_k < V**.
- β = −Φ⁻¹(p_f).
- CMC: klasyczne MC. SS: Subset Simulation (Au & Beck) z MMA.

---
JEMEN · jvk · MIT
"""


def main() -> None:
    st.set_page_config(page_title="JEMEN — stopa + β", page_icon="🎲", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("JEMEN")
    st.caption(
        "Nośność stopy (SYNAJv2) + **CMC** albo **Subset Simulation** · "
        "**φ′, c′, γ** lognormalne · wynik: **β** · **jvk** · MIT"
    )
    st.warning("Wersja testowa — nie do dokumentacji projektowej.")

    tip35 = recommend_n(3.5, 30)
    tip38 = recommend_n(3.8, 30)
    with st.expander("Ile obliczeń?", expanded=False):
        st.markdown(
            f"""
| Cel | CMC (N) | Subset Simulation |
|-----|---------|-------------------|
| Szybki rzut | **5–20 tys.** | N/poziom **1–2 tys.**, p0=0,1 |
| β ≈ 3,5 (~30 awarii CMC) | **≳ {tip35['n_suggested']:,}** | zwykle **kilka poziomów × 2 tys.** |
| β ≈ 3,8 | **≳ {tip38['n_suggested']:,}** | SS zwykle tańsze przy rzadkich awariach |

{tip35['note']}
            """
        )

    t1, t2, t3 = st.tabs(["1. Dane", "2. Obliczenia → β", "3. Pomoc"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Stopa i obciążenie (stałe)")
            st.number_input("B [m]", 0.5, 20.0, 2.0, 0.1, key="B")
            st.number_input("L [m]", 0.5, 50.0, 3.0, 0.1, key="L")
            st.number_input("D_f [m]", 0.0, 10.0, 1.0, 0.1, key="Df")
            st.number_input(
                "V — obciążenie pionowe [kN] (stałe)",
                10.0,
                50_000.0,
                2220.0,
                10.0,
                key="V",
            )
            st.number_input("q_N [kPa]", 0.0, 200.0, 0.0, 1.0, key="qN")
            st.checkbox("Współczynnik głębokości d", True, key="depth")

        with c2:
            st.subheader("Grunt — średnia i SD (→ lognormal)")
            st.caption("Podajesz μ i SD jak z opisu; losowanie jest lognormalne.")
            st.number_input("γ — średnia [kN/m³]", 14.0, 24.0, 18.0, 0.5, key="g_mu")
            st.number_input("γ — SD [kN/m³]", 0.0, 5.0, 0.90, 0.05, key="g_sd")
            st.number_input("φ′ — średnia [°]", 5.0, 50.0, 30.0, 0.5, key="phi_mu")
            st.number_input("φ′ — SD [°]", 0.0, 15.0, 1.8, 0.01, key="phi_sd")
            st.number_input("c′ — średnia [kPa]", 0.0, 200.0, 0.0, 1.0, key="c_mu")
            st.number_input("c′ — SD [kPa]", 0.0, 100.0, 0.0, 1.0, key="c_sd")

            st.markdown("**Parametry lognormalne (po przeliczeniu)**")
            try:
                gln = normal_moments_to_lognormal(
                    float(st.session_state["g_mu"]), float(st.session_state["g_sd"])
                )
                pln = normal_moments_to_lognormal(
                    float(st.session_state["phi_mu"]), float(st.session_state["phi_sd"])
                )
                st.markdown(_ln_line("γ", gln))
                st.markdown(_ln_line("φ′", pln))
                if float(st.session_state["c_mu"]) > 0:
                    cln = normal_moments_to_lognormal(
                        float(st.session_state["c_mu"]), float(st.session_state["c_sd"])
                    )
                    st.markdown(_ln_line("c′", cln))
                else:
                    st.caption("c′ = 0 → w obliczeniach stałe zero.")
            except Exception as exc:
                st.error(str(exc))

        R0, f0 = resistance_Rk(
            phi_deg=float(st.session_state["phi_mu"]),
            c_prime=float(st.session_state["c_mu"]),
            gamma=float(st.session_state["g_mu"]),
            B=float(st.session_state["B"]),
            L=float(st.session_state["L"]),
            Df=float(st.session_state["Df"]),
            V=float(st.session_state["V"]),
            q_N=float(st.session_state["qN"]),
            use_depth=bool(st.session_state["depth"]),
        )
        V = float(st.session_state["V"])
        st.info(
            f"**Krok kontrolny:** na średnich R_k = **{R0:.1f} kN**, "
            f"FS = **{R0/V:.2f}** "
            f"(N_c={f0['N_c']:.2f}, N_q={f0['N_q']:.2f}, N_γ={f0['N_gamma']:.2f})."
        )

    with t2:
        method = st.radio(
            "Metoda obliczeń",
            options=["CMC", "SS"],
            format_func=lambda x: (
                "CMC — klasyczne Monte Carlo"
                if x == "CMC"
                else "Subset Simulation (SS) — rzadkie awarie"
            ),
            horizontal=True,
            key="method",
        )

        if method == "CMC":
            st.markdown(
                """
**CMC:** losuje N razy φ′, c′, γ → R_k → awaria gdy R_k &lt; V → β = −Φ⁻¹(p_f).
Proste i dokładne przy dużej liczbie awarii; przy β≳3,5 wymaga dużego N.
                """
            )
            n = st.select_slider(
                "Liczba realizacji N",
                options=[
                    2_000,
                    5_000,
                    10_000,
                    20_000,
                    50_000,
                    100_000,
                    130_000,
                    200_000,
                    500_000,
                ],
                value=100_000,
            )
            p0 = 0.1
        else:
            st.markdown(
                """
**Subset Simulation:** łańcuch poziomów warunkowych (Au & Beck) z MMA w przestrzeni
standardowej. Przy rzadkich awariach zwykle **znacznie mniej** wywołań nośności niż CMC.
                """
            )
            n = st.select_slider(
                "Liczba próbek na poziom N",
                options=[500, 1_000, 2_000, 3_000, 5_000, 10_000],
                value=2_000,
            )
            p0 = st.select_slider(
                "p0 — prawdopodobieństwo warunkowe na poziom",
                options=[0.05, 0.1, 0.2],
                value=0.1,
            )

        seed = st.number_input("Ziarno RNG", 0, 999_999, 42, 1)
        st.caption(tip35["note"] if method == "CMC" else "SS: typowo N=2000, p0=0,1.")

        btn = "Uruchom CMC → β" if method == "CMC" else "Uruchom Subset Simulation → β"
        if st.button(btn, type="primary"):
            inp = {
                "B": float(st.session_state["B"]),
                "L": float(st.session_state["L"]),
                "Df": float(st.session_state["Df"]),
                "V": float(st.session_state["V"]),
                "q_N": float(st.session_state["qN"]),
                "gamma_mu": float(st.session_state["g_mu"]),
                "gamma_sd": float(st.session_state["g_sd"]),
                "phi_mu": float(st.session_state["phi_mu"]),
                "phi_sd": float(st.session_state["phi_sd"]),
                "c_mu": float(st.session_state["c_mu"]),
                "c_sd": float(st.session_state["c_sd"]),
                "n": int(n),
                "seed": int(seed),
                "use_depth": bool(st.session_state["depth"]),
                "p0": float(p0),
                "method": method,
            }
            label = f"{n:,} realizacji" if method == "CMC" else f"SS N={n:,}/poziom, p0={p0}"
            with st.spinner(f"Liczenie ({label})…"):
                common = dict(
                    B=inp["B"],
                    L=inp["L"],
                    Df=inp["Df"],
                    V=inp["V"],
                    gamma_mu=inp["gamma_mu"],
                    gamma_sd=inp["gamma_sd"],
                    q_N=inp["q_N"],
                    phi_mu=inp["phi_mu"],
                    phi_sd=inp["phi_sd"],
                    c_mu=inp["c_mu"],
                    c_sd=inp["c_sd"],
                    n=inp["n"],
                    seed=inp["seed"],
                    use_depth=inp["use_depth"],
                )
                if method == "CMC":
                    st.session_state["mc"] = run_monte_carlo(**common)
                else:
                    st.session_state["mc"] = run_subset_simulation(
                        **common, p0=inp["p0"]
                    )
                st.session_state["mc_inp"] = inp

        res = st.session_state.get("mc")
        if not res:
            st.info("Ustaw dane w **1. Dane**, wybierz metodę, potem uruchom.")
        else:
            st.markdown(
                f"## 1) Główny wynik — β "
                f"({res.get('method_label', res.get('method', ''))})"
            )
            st.markdown(
                f"<p class='beta-big'>β = {res['beta']:.3f}</p>",
                unsafe_allow_html=True,
            )
            box = "box-warn" if res["beta_kind"] == "lower_bound" else "box-ok"
            bullets = "".join(f"<li>{x}</li>" for x in res["interpret"])
            st.markdown(
                f"<div class='{box}'><ul style='margin:0;padding-left:1.2rem'>{bullets}</ul>"
                f"<p style='margin:0.55rem 0 0 0;opacity:0.9'>{res['beta_note']}</p></div>",
                unsafe_allow_html=True,
            )

            if res.get("method") == "SS" and res.get("ss_levels"):
                st.markdown("### Poziomy Subset Simulation")
                rows = []
                for lv in res["ss_levels"]:
                    rows.append(
                        {
                            "poziom": lv["level"],
                            "b = g": f"{lv['b']:.3g}",
                            "p_cond": f"{lv['p_cond']:.4g}",
                            "finał": "tak" if lv.get("final") else "nie",
                        }
                    )
                st.dataframe(rows, hide_index=True, use_container_width=True)
                ar = res.get("ss_accept_rate", float("nan"))
                ar_txt = f"{ar:.3f}" if np.isfinite(ar) else "—"
                cov = res.get("cov_pf", float("nan"))
                cov_txt = f"{cov:.3f}" if np.isfinite(cov) else "—"
                st.caption(
                    f"Wywołań nośności ≈ **{res.get('n_eval', 0):,}** · "
                    f"akceptacja MMA ≈ {ar_txt} · CoV(p_f) ≈ {cov_txt}"
                )

            st.markdown("## 2) Prawdopodobieństwo awarii p_f")
            se_b = res["se_beta"]
            se_b_txt = f"{se_b:.3f}" if np.isfinite(se_b) else "—"
            st.markdown(
                f"""
| Wielkość | Wartość | Znaczenie |
|----------|---------|-----------|
| **p_f** | **{res['p_f']:.4e}** | P(R_k &lt; V) |
| awarie | **{res['n_fail']} / {res['n_ok']:,}** | CMC: cała próba; SS: ostatni poziom |
| SE(p_f) | {res['se_pf']:.2e} | niepewność p_f |
| SE(β) | {se_b_txt} | niepewność β |
| β (momenty G) | {res['beta_moments']:.3f} | μ_G/σ_G (orientacyjnie) |
                """
            )

            st.markdown("## 3) Nośność R_k")
            st.markdown(
                f"""
| | [kN] | Komentarz |
|--|------|-----------|
| **V** (stałe) | **{res['V']:.1f}** | obciążenie |
| R_k na średnich | **{res['R_det']:.1f}** | bez losowania |
| średnia R_k | **{res['R_mean']:.1f}** ± {res['R_std']:.1f} | z próby |
| 5. percentyl R_k | **{res['R_p5']:.1f}** | |
| mediana R_k | **{res['R_p50']:.1f}** | |
| 95. percentyl R_k | **{res['R_p95']:.1f}** | |
| FS = R(μ)/V | **{res['FS_det']:.2f}** | |
| FS₅ = R₅/V | **{res['FS_p5']:.2f}** | |
                """
            )

            fig, ax = plt.subplots(figsize=(7.4, 3.7))
            ax.hist(res["R"], bins=50, color="#5BA3D9", edgecolor="#1a2332", alpha=0.9)
            ax.axvline(res["V"], color="#e55", lw=2, label=f"V = {res['V']:.0f}")
            ax.axvline(
                res["R_det"],
                color="#3c9",
                lw=1.5,
                ls="--",
                label=f"R(μ) = {res['R_det']:.0f}",
            )
            ax.axvline(
                res["R_p5"],
                color="#e8a838",
                lw=1.2,
                ls=":",
                label=f"R₅ = {res['R_p5']:.0f}",
            )
            ax.set_xlabel("R_k [kN]")
            ax.set_ylabel("liczba realizacji")
            title = "Rozkład R_k — " + res.get("method_label", "")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="#aaa")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")
            ax.title.set_color("#e8eef5")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            inp = st.session_state.get("mc_inp")
            if inp:
                md = results_to_markdown(res, inp)
                tag = res.get("method", "MC")
                st.download_button(
                    label="Pobierz zestaw wyników (.md)",
                    data=md,
                    file_name=(
                        f"JEMEN_{tag}_N{inp['n']}_V{inp['V']:.0f}_"
                        f"SDphi{inp['phi_sd']:.2f}_SDg{inp['gamma_sd']:.2f}.md"
                    ),
                    mime="text/markdown",
                    type="secondary",
                )

    with t3:
        st.markdown(
            r"""
### Metody
- **CMC** — klasyczne Monte Carlo: p_f ≈ n_awarii / N, β = −Φ⁻¹(p_f).
- **Subset Simulation** — rzadkie zdarzenia przez poziomy warunkowe
  \(p_f = \prod p_i\) (Au & Beck), MCMC: Modified Metropolis w przestrzeni u.

### Dane
1. **B, L, D_f, V** oraz μ, SD dla **γ, φ′, c′**.
2. μ, SD → lognormal (ta sama średnia i wariancja).
3. Awaria: \(R_k < V\).

\[
\delta=\sigma/\mu,\quad
\sigma_{\ln}=\sqrt{\ln(1+\delta^2)},\quad
\mu_{\ln}=\ln\mu-\tfrac12\sigma_{\ln}^2
\]
            """
        )
        st.caption("JEMEN · jvk · MIT")


if __name__ == "__main__":
    main()
