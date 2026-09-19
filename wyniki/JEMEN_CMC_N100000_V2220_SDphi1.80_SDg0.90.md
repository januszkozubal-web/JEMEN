# JEMEN — zestaw wyników

Wersja testowa — nie do dokumentacji projektowej.

**Metoda:** CMC (Crude Monte Carlo)

## Dane wejściowe

| Wielkość | Wartość |
|----------|---------|
| B [m] | 2 |
| L [m] | 3 |
| D_f [m] | 1 |
| V [kN] (stałe) | 2220 |
| q_N [kPa] | 0 |
| współczynnik głębokości d | tak |
| N (CMC) / N na poziom (SS) | 100,000 |
| ziarno RNG | 42 |
| p0 (SS) | — |

### Grunt — μ, SD → lognormal

| Zmienna | μ | SD | CoV | μ_ln | σ_ln |
|---------|---|----|-----|------|------|
| γ | 18 | 0.9 | 0.0500 | 2.889123 | 0.049969 |
| φ′ | 30 | 1.8 | 0.0600 | 3.399401 | 0.059946 |
| c′ | 0 | 0 | (stałe 0) | — | — |

## 1) Indeks niezawodności β

- **β = 3.8906** (from_pf)
- β policzone z częstości awarii: β = −Φ⁻¹(p_f), p_f = 5/100000 = 5.0000e-05.

### Interpretacja

- **Metoda: CMC** (klasyczne Monte Carlo) — 100,000 realizacji.
- **β = 3.89** oznacza, że w standardowym modelu awarii prawdopodobieństwo przekroczenia nośności wynosi ok. **p_f = Φ(−β)** (im większe β, tym bezpieczniej).
- β ≈ 3,5…4,2 — typowy zakres docelowy wielu zastosowań (orientacyjnie rząd β≈3,8 przy niektórych założeniach CC2 / 50 lat).
- W tej próbie: **5 awarii / 100,000** poprawnych realizacji.

## 2) Prawdopodobieństwo awarii

| Wielkość | Wartość |
|----------|---------|
| p_f = P(R_k < V) | 5.000000e-05 |
| awarie (ostatni poziom / CMC) | 5 / 100,000 |
| SE(p_f) | 2.2360e-05 |
| SE(β) | 0.1085 |
| β (momenty G = R−V) | 2.2425 |

## 3) Nośność R_k

| Wielkość | [kN] |
|----------|------|
| V | 2220.00 |
| R_k na średnich | 4967.80 |
| średnia R_k | 5121.56 |
| odchylenie std R_k | 1293.92 |
| 5. percentyl R_k | 3405.56 |
| mediana R_k | 4924.00 |
| 95. percentyl R_k | 7505.37 |
| FS = R(μ)/V | 2.238 |
| FS₅ = R₅/V | 1.534 |

## Uwagi

- Awaria w modelu: **R_k < V**.
- β = −Φ⁻¹(p_f).
- CMC: klasyczne MC. SS: Subset Simulation (Au & Beck) z MMA.

---
JEMEN · jvk · MIT
