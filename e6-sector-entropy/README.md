# e6 — Entropía permutacional por sector y por mercado (Sección 5.6)

¿Hay sectores o mercados consistentemente más o menos eficientes según la PE?
Dos sub-experimentos: los 11 ETFs sectoriales SPDR del S&P 500 (1998-2025) y
los índices nativos de nueve de las bolsas más grandes del mundo. Rankings
anuales con percentiles, bootstrap sobre años, reacción en cuatro episodios de
crisis pre-declarados y co-evolución de los cambios de PE.

## Datos

Cierres diarios vía yfinance (`fetch_data.py` descarga ETFs sectoriales, SPY y
los índices de mercados a `data/`). Nada manual.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py
python run_sectors.py        # sub-experimento 1: sectores
python run_markets.py        # sub-experimento 2: mercados
python gen_charts.py
python gen_charts_markets.py
```

## Configuración

PE normalizada (m=3, tau=1) en tres ventanas con roles distintos: W=252
(principal), W=504 (sensibilidad) y W=140 (crisis). Rankings anuales con
percentiles mid-rank y CI 95% por bootstrap sobre años; PE estática con m=6
sobre la serie completa de cada mercado como robustez. Episodios de crisis:
dot-com, 2008, COVID y bear de tasas 2022 (ventanas pico-valle). Semillas fijas.

## Outputs

`outputs/` con percentiles por config (CSV), resultados (JSON) y los charts de
la sección (rotación, crisis, co-evolución, choropleth de mercados).
