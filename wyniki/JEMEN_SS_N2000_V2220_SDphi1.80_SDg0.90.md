# JEMEN — zestaw wyników

Wersja testowa — nie do dokumentacji projektowej.

**Metoda:** Subset Simulation

## Dane wejściowe

| Wielkość | Wartość |
|----------|---------|
| B [m] | 2 |
| L [m] | 3 |
| D_f [m] | 1 |
| V [kN] (stałe) | 2220 |
| q_N [kPa] | 0 |
| współczynnik głębokości d | tak |
| N (CMC) / N na poziom (SS) | 2,000 |
| ziarno RNG | 42 |
| p0 (SS) | 0.1 |

### Grunt — μ, SD → lognormal

| Zmienna | μ | SD | CoV | μ_ln | σ_ln |
|---------|---|----|-----|------|------|
| γ | 18 | 0.9 | 0.0500 | 2.889123 | 0.049969 |
| φ′ | 30 | 1.8 | 0.0600 | 3.399401 | 0.059946 |
| c′ | 0 | 0 | (stałe 0) | — | — |

## 1) Indeks niezawodności β

- **β = 3.8525** (from_pf)
- Subset Simulation: p_f = ∏ p_i = 5.8450e-05 (5 poziom(ów), p0=0.1, N/poziom=2,000). β = −Φ⁻¹(p_f).

### Interpretacja

- **Metoda: Subset Simulation** — 5 poziom(ów), łącznie ~10,000 wywołań nośności (p0=0.1).
- **β = 3.85** oznacza, że w standardowym modelu awarii prawdopodobieństwo przekroczenia nośności wynosi ok. **p_f = Φ(−β)** (im większe β, tym bezpieczniej).
- β ≈ 3,5…4,2 — typowy zakres docelowy wielu zastosowań (orientacyjnie rząd β≈3,8 przy niektórych założeniach CC2 / 50 lat).
- W tej próbie: **1169 awarii / 2,000** poprawnych realizacji.

## Subset Simulation — poziomy

| poziom | próg b (g=R−V) | p_cond | finał |
|--------|----------------|--------|-------|
| 0 | 1404 | 0.1 | nie |
| 1 | 755.1 | 0.1 | nie |
| 2 | 369.4 | 0.1 | nie |
| 3 | 58.67 | 0.1 | nie |
| 4 | 0 | 0.5845 | tak |

- p0 = 0.1
- akceptacja MMA ≈ 0.431
- wywołań nośności ≈ 10,000

## 2) Prawdopodobieństwo awarii

| Wielkość | Wartość |
|----------|---------|
| p_f = P(R_k < V) | 5.845000e-05 |
| awarie (ostatni poziom / CMC) | 1169 / 2,000 |
| SE(p_f) | 7.9189e-06 |
| SE(β) | 0.0332 |
| β (momenty G = R−V) | -0.4663 |

## 3) Nośność R_k

| Wielkość | [kN] |
|----------|------|
| V | 2220.00 |
| R_k na średnich | 4967.80 |
| średnia R_k | 2178.17 |
| odchylenie std R_k | 89.70 |
| 5. percentyl R_k | 2001.13 |
| mediana R_k | 2203.49 |
| 95. percentyl R_k | 2271.09 |
| FS = R(μ)/V | 2.238 |
| FS₅ = R₅/V | 0.901 |

## Uwagi

- Awaria w modelu: **R_k < V**.
- β = −Φ⁻¹(p_f).
- CMC: klasyczne MC. SS: Subset Simulation (Au & Beck) z MMA.

---
JEMEN · jvk · MIT
