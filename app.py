# JEMEN — stopa + Monte Carlo → indeks niezawodności β
# Author: jvk · MIT

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from bearing import resistance_Rk
from distributions import normal_moments_to_lognormal, recommend_n, run_monte_carlo

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
    """Zestaw wyników MC do pliku .md."""
    se_b = res["se_beta"]
    se_b_txt = f"{se_b:.4f}" if np.isfinite(se_b) else "—"
    interpret = "\n".join(f"- {x}" for x in res["interpret"])
    if res["c_ln"] is not None:
        c_row = _ln_md_row("c′", res["c_ln"])
    else:
        c_row = (
            f"| c′ | {inp['c_mu']:.6g} | {inp['c_sd']:.6g} | (stałe 0) | — | — |"
        )
    return f"""# JEMEN — zestaw wyników Monte Carlo

Wersja testowa — nie do dokumentacji projektowej.

## Dane wejściowe

| Wielkość | Wartość |
|----------|---------|
| B [m] | {inp['B']:.4g} |
| L [m] | {inp['L']:.4g} |
| D_f [m] | {inp['Df']:.4g} |
| V [kN] (stałe) | {inp['V']:.4g} |
| q_N [kPa] | {inp['q_N']:.4g} |
| współczynnik głębokości d | {"tak" if inp['use_depth'] else "nie"} |
| N (liczba realizacji) | {inp['n']:,} |
| ziarno RNG | {inp['seed']} |

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

## 2) Prawdopodobieństwo awarii

| Wielkość | Wartość |
|----------|---------|
| p_f = P(R_k < V) | {res['p_f']:.6e} |
| awarie | {res['n_fail']} / {res['n_ok']:,} |
| SE(p_f) | {res['se_pf']:.4e} |
| SE(β) | {se_b_txt} |
| β (momenty G = R−V) | {res['beta_moments']:.4f} |

## 3) Nośność R_k

| Wielkość | [kN] |
|----------|------|
| V | {res['V']:.2f} |
| R_k na średnich | {res['R_det']:.2f} |
| średnia R_k (MC) | {res['R_mean']:.2f} |
| odchylenie std R_k | {res['R_std']:.2f} |
| 5. percentyl R_k | {res['R_p5']:.2f} |
| mediana R_k | {res['R_p50']:.2f} |
| 95. percentyl R_k | {res['R_p95']:.2f} |
| FS = R(μ)/V | {res['FS_det']:.3f} |
| FS₅ = R₅/V | {res['FS_p5']:.3f} |

## Uwagi

- Awaria w modelu: **R_k < V**.
- β = −Φ⁻¹(p_f); przy braku awarii w próbie podawana jest tylko dolna granica.
- φ′, c′, γ losowane niezależnie (lognormalnie); V stałe.

---
JEMEN · jvk · MIT
"""


def main() -> None:
    st.set_page_config(page_title="JEMEN — stopa + β", page_icon="🎲", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title("JEMEN")
    st.caption(
        "Nośność stopy (SYNAJv2) + Monte Carlo · **φ′, c′, γ** lognormalne (z μ i SD) · "
        "wynik: **β** · **jvk** · MIT"
    )
    st.warning("Wersja testowa — nie do dokumentacji projektowej.")

    tip35 = recommend_n(3.5, 30)
    tip38 = recommend_n(3.8, 30)
    with st.expander("Ile obliczeń MC?", expanded=False):
        st.markdown(
            f"""
| Cel | Orientacyjne N |
|-----|----------------|
| Szybki rzut | **5 000 – 20 000** |
| β ≈ 3,5 i ~30 awarii | **≳ {tip35['n_suggested']:,}** |
| β ≈ 3,8 i ~30 awarii | **≳ {tip38['n_suggested']:,}** |

{tip35['note']}
            """
        )

    t1, t2, t3 = st.tabs(["1. Dane", "2. Monte Carlo → β", "3. Pomoc"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Stopa i obciążenie (stałe)")
            st.number_input("B [m]", 0.5, 20.0, 2.0, 0.1, key="B")
            st.number_input("L [m]", 0.5, 50.0, 3.0, 0.1, key="L")
            st.number_input("D_f [m]", 0.0, 10.0, 1.0, 0.1, key="Df")
            st.number_input(
                "V — obciążenie pionowe [kN] (stałe w MC)",
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
                    st.caption("c′ = 0 → w MC stałe zero.")
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
            f"**Krok kontrolny (bez MC):** na samych średnich "
            f"R_k = **{R0:.1f} kN**, FS = R_k/V = **{R0/V:.2f}**. "
            f"To jeszcze nie β — tylko punkt odniesienia "
            f"(N_c={f0['N_c']:.2f}, N_q={f0['N_q']:.2f}, N_γ={f0['N_gamma']:.2f})."
        )

    with t2:
        st.markdown(
            """
**Co robi ta zakładka (kolejność):**
1. Losuje N razy **φ′, c′, γ** (lognormalnie z Twoich μ, SD).
2. Za każdym razem liczy nośność **R_k**.
3. **Awaria** = R_k &lt; V (V stałe).
4. Z częstości awarii liczy **β** — to jest wynik, którego szukasz.
            """
        )
        n = st.select_slider(
            "Liczba realizacji N",
            options=[2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 130_000, 200_000, 500_000],
            value=500_000,
        )
        seed = st.number_input("Ziarno RNG", 0, 999_999, 42, 1)
        st.caption(tip35["note"])

        if st.button("Uruchom Monte Carlo → β", type="primary"):
            with st.spinner(f"Liczenie {n:,} realizacji…"):
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
                }
                st.session_state["mc"] = run_monte_carlo(
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
                st.session_state["mc_inp"] = inp

        res = st.session_state.get("mc")
        if not res:
            st.info("Ustaw dane w **1. Dane**, potem kliknij przycisk powyżej.")
        else:
            st.markdown("## 1) Główny wynik — indeks β")
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

            st.markdown("## 2) Prawdopodobieństwo awarii p_f")
            se_b = res["se_beta"]
            se_b_txt = f"{se_b:.3f}" if np.isfinite(se_b) else "— (za mało awarii)"
            st.markdown(
                f"""
| Wielkość | Wartość | Znaczenie |
|----------|---------|-----------|
| **p_f** | **{res['p_f']:.4e}** | P(R_k &lt; V) w modelu MC |
| awarie | **{res['n_fail']} / {res['n_ok']:,}** | ile razy nośność &lt; obciążenie |
| SE(p_f) | {res['se_pf']:.2e} | niepewność częstości |
| SE(β) | {se_b_txt} | niepewność β z próby |
| β (momenty G) | {res['beta_moments']:.3f} | przybliżenie μ_G/σ_G (nie zastępuje β z p_f) |
                """
            )

            st.markdown("## 3) Nośność R_k — co wyszło z MC")
            st.markdown(
                f"""
| | [kN] | Komentarz |
|--|------|-----------|
| **V** (stałe) | **{res['V']:.1f}** | obciążenie pionowe |
| R_k na średnich | **{res['R_det']:.1f}** | jedna liczba, bez losowania |
| średnia R_k z MC | **{res['R_mean']:.1f}** ± {res['R_std']:.1f} | rozrzut nośności |
| 5. percentyl R_k | **{res['R_p5']:.1f}** | „pesymistyczny” ogon |
| mediana R_k | **{res['R_p50']:.1f}** | |
| 95. percentyl R_k | **{res['R_p95']:.1f}** | |
| FS = R(μ)/V | **{res['FS_det']:.2f}** | klasyczny zapas na średnich |
| FS₅ = R₅/V | **{res['FS_p5']:.2f}** | zapas względem 5. percentyla |
                """
            )
            st.caption(
                "Histogram: czerwona linia = V (awaria na lewo od niej), "
                "zielona przerywana = R_k policzone na średnich."
            )

            fig, ax = plt.subplots(figsize=(7.4, 3.7))
            ax.hist(res["R"], bins=50, color="#5BA3D9", edgecolor="#1a2332", alpha=0.9)
            ax.axvline(res["V"], color="#e55", lw=2, label=f"V = {res['V']:.0f} (awaria: R<V)")
            ax.axvline(
                res["R_det"],
                color="#3c9",
                lw=1.5,
                ls="--",
                label=f"R(μ) = {res['R_det']:.0f}",
            )
            ax.axvline(res["R_p5"], color="#e8a838", lw=1.2, ls=":", label=f"R₅ = {res['R_p5']:.0f}")
            ax.set_xlabel("R_k [kN]")
            ax.set_ylabel("liczba realizacji")
            ax.set_title("Rozkład nośności z Monte Carlo")
            ax.legend(fontsize=8)
            ax.set_facecolor("#0e1117")
            fig.patch.set_facecolor("#0e1117")
            ax.tick_params(colors="#aaa")
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")
            ax.title.set_color("#e8eef5")
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

            with st.expander("Jak czytać β, p_f i FS obok siebie"):
                st.markdown(
                    """
- **FS na średnich** mówi tylko „ile razy średnia nośność jest większa od V” — **nie** uwzględnia rozrzutu.
- **p_f** mówi wprost: w jakim ułamku losowań grunt „nie uniósłby” obciążenia V.
- **β** to ta sama informacja w skali inżynierskiej: większe β = rzadsza awaria.
  Przykłady orientacyjne: β=0 → p_f≈50%; β=3 → p_f≈0,14%; β=3,8 → p_f≈7·10⁻⁵.
- Jeśli **brak awarii** przy małym N, β jest tylko **dolną granicą** — nie myl tego z dokładnym β.
                    """
                )

            inp = st.session_state.get("mc_inp")
            if inp:
                md = results_to_markdown(res, inp)
                st.download_button(
                    label="Pobierz zestaw wyników (.md)",
                    data=md,
                    file_name=(
                        f"JEMEN_MC_N{inp['n']}_V{inp['V']:.0f}_"
                        f"SDphi{inp['phi_sd']:.2f}_SDg{inp['gamma_sd']:.2f}.md"
                    ),
                    mime="text/markdown",
                    type="secondary",
                )

    with t3:
        st.markdown(
            r"""
### Po kolei
1. Wpisujesz **B, L, D_f, V** oraz dla gruntu **μ i SD** dla **γ, φ′, c′**.
2. Aplikacja przelicza μ, SD → parametry **lognormalne** (ta sama średnia i wariancja).
3. MC liczy rozkład **R_k**; awaria gdy \(R_k < V\).
4. Wynik docelowy: **β = −Φ⁻¹(p_f)**.

\[
\delta=\sigma/\mu,\quad
\sigma_{\ln}=\sqrt{\ln(1+\delta^2)},\quad
\mu_{\ln}=\ln\mu-\tfrac12\sigma_{\ln}^2
\]

**γ** też jest losowe (lognormalne), jeśli podasz SD &gt; 0.
            """
        )
        st.caption("JEMEN · jvk · MIT")


if __name__ == "__main__":
    main()
