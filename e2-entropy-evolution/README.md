# e2 — Evolución temporal de la eficiencia (Sección 5.2)

¿La eficiencia del mercado accionario de EE.UU. creció con la modernización
tecnológica? PE rolling del S&P 500 en tres escalas: daily 1928-2025, y hourly
y 5 minutos sobre 2008-2021, con test de tendencia de Mann-Kendall y barrido
de sensibilidad de parámetros en cada escala.

## Datos

- **Daily**: S&P 500 desde 1928, descargado con yfinance (automático, cache local).
- **Intradía**: dataset público de barras de 1 minuto de SPY (2008-2021).
  Descargarlo de Kaggle: <https://www.kaggle.com/datasets/rockinbrock/spy-1-minute-data>
  y colocar el CSV como `datasets/spy_1min_2008_2021_cleaned.csv` **en la raíz
  del repo** (crear la carpeta `datasets/` si no existe).

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_daily.py    # sub-experimento (a): daily 1928-2025
python run_hourly.py   # sub-experimento (b1): hourly 2008-2021
python run_5min.py     # sub-experimento (b2): 5 minutos 2008-2021
```

## Configuración

PE normalizada sobre log-retornos. Configs principales: daily (m=3, W=252),
hourly (m=3, W=140), 5-min (m=3, W=390); cada sub-experimento corre además tres
configuraciones alternativas de sensibilidad (incluye m=4). Mann-Kendall sobre
las series promediadas anualmente.

## Outputs

`outputs/e2_*_results.json` (tendencias y p-values por config), series CSV y
`outputs/charts/` con las figuras de la sección. La serie densa de 5 minutos
(`e2_5min.csv`) se regenera al correr y no se versiona por tamaño.
