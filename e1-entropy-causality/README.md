# e1 — Causalidad entre entropía y dinámica del mercado (Sección 5.1)

¿La entropía permutacional es causa de los retornos futuros o solo una asociación?
Descubrimiento causal con DirectLiNGAM sobre paneles de factores diarios, con
moving-block bootstrap (respeta la autocorrelación), test de no-gausianidad de
residuos y análisis rolling multi-activo bajo el marco de la AMH.

## Nomenclatura de scripts

Los scripts `run_e1_*.py` a `run_e5_*.py` usan una numeración **interna de
sub-partes** de este experimento; no corresponde a las carpetas e1-e8 del repo.
El mapeo con la Sección 5.1 de la tesis es:

| Script | Parte de 5.1 | Qué hace |
|---|---|---|
| `run_e1_static.py` | Parte I | Grafo causal estático de SPY + rolling LiNGAM |
| `run_e2_predictive.py` | Parte I | ¿Algún factor causa el retorno de t+1 en SPY? |
| `run_e3_indices.py` | Parte I | Generalización a QQQ, IWM y DIA |
| `run_e4_subsets.py` | Parte I | Subsets alternativos de factores (3 tradiciones de asset pricing) |
| `run_e5_cross.py` | Parte I | Cross-product: 24 configuraciones (índices x subsets x PE/WPE) |
| `run_sensitivity.py` | Parte I | Sensibilidad a parámetros de PE y block size del bootstrap |
| `run_parte2_daily.py` | Parte II | Rolling multi-activo daily (20 activos, ventanas 400 y 200 días) |
| `run_parte2_hourly.py` | Parte II | Rolling intradía hourly (8 activos, últimos 2 años) |
| `generate_*_figures.py` | --- | Charts para el documento |
| `run_all.py` | --- | Orquesta la Parte I y escribe `outputs/results.json` |

## Datos

Cierres diarios y hourly descargados con yfinance (cache local en el módulo
`common`). No requiere descargas manuales.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all.py            # Parte I completa
python run_parte2_daily.py   # Parte II daily
python run_parte2_hourly.py  # Parte II hourly
python generate_parte1_figures.py
python generate_parte2_figures.py
```

## Configuración

Panel baseline de 7 factores (Returns, Volatility, Momentum, Reversal, Liquidity,
Volume, PE/WPE) con diferenciación fraccional (d=0,3) para estacionariedad.
Umbral de arista robusta: probabilidad bootstrap >= 0,5. Semillas fijas.

## Outputs

`outputs/results.json` (Parte I), `outputs/parte2_*.csv` (Parte II) y
`outputs/charts/` con las figuras de la sección.
