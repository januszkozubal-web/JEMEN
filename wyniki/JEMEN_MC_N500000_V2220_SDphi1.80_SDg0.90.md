# JEMEN — zestaw wyników Monte Carlo

Wersja testowa — nie do dokumentacji projektowej.

## Dane wejściowe

| Wielkość | Wartość |
|----------|---------|
| B [m] | 2 |
| L [m] | 3 |
| D_f [m] | 1 |
| V [kN] (stałe) | 2220 |
| q_N [kPa] | 0 |
| współczynnik głębokości d | tak |
| N (liczba realizacji) | 500,000 |
| ziarno RNG | 42 |

### Grunt — μ, SD → lognormal

| Zmienna | μ | SD | CoV | μ_ln | σ_ln |
|---------|---|----|-----|------|------|
| γ | 18 | 0.9 | 0.0500 | 2.889123 | 0.049969 |
| φ′ | 30 | 1.8 | 0.0600 | 3.399401 | 0.059946 |
| c′ | 0 | 0 | (stałe 0) | — | — |

## 1) Indeks niezawodności β

- **β = 3.8461** (from_pf)
- β policzone z częstości awarii: β = −Φ⁻¹(p_f), p_f = 30/500000 = 6.0000e-05.

### Interpretacja

- **β = 3.85** oznacza, że w standardowym modelu awarii prawdopodobieństwo przekroczenia nośności wynosi ok. **p_f = Φ(−β)** (im większe β, tym bezpieczniej).
- β ≈ 3,5…4,2 — typowy zakres docelowy wielu zastosowań (orientacyjnie rząd β≈3,8 przy niektórych założeniach CC2 / 50 lat).
- W tej próbie: **30 awarii / 500,000** poprawnych realizacji.

## 2) Prawdopodobieństwo awarii

| Wielkość | Wartość |
|----------|---------|
| p_f = P(R_k < V) | 6.000000e-05 |
| awarie | 30 / 500,000 |
| SE(p_f) | 1.0954e-05 |
| SE(β) | 0.0448 |
| β (momenty G = R−V) | 2.2508 |

## 3) Nośność R_k

| Wielkość | [kN] |
|----------|------|
| V | 2220.00 |
| R_k na średnich | 4967.80 |
| średnia R_k (MC) | 5124.45 |
| odchylenie std R_k | 1290.41 |
| 5. percentyl R_k | 3408.13 |
| mediana R_k | 4927.13 |
| 95. percentyl R_k | 7507.74 |
| FS = R(μ)/V | 2.238 |
| FS₅ = R₅/V | 1.535 |

## Uwagi

- Awaria w modelu: **R_k < V**.
- β = −Φ⁻¹(p_f); przy braku awarii w próbie podawana jest tylko dolna granica.
- φ′, c′, γ losowane niezależnie (lognormalnie); V stałe.

---
JEMEN · jvk · MIT
