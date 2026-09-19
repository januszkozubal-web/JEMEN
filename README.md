# JEMEN — stopa + Monte Carlo → β

Nośność jak SYNAJv2; **φ′, c′, γ** lognormalne z μ i SD; wynik: **indeks β** (oraz p_f, percentyle R_k). Eksport zestawu do `.md`.

## Uruchomienie lokalne

```bash
cd JEMEN
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8504
```

Potem: http://localhost:8504

## Streamlit Community Cloud

Po podłączeniu repozytorium w [share.streamlit.io](https://share.streamlit.io):
- Main file: `app.py`
- Python 3.11+

Autor: **jvk** · MIT · **wersja testowa** — nie do dokumentacji projektowej.
