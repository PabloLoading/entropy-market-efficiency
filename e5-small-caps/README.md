# e5 — Entropía y capitalización bursátil (Sección 5.5)

¿Las small caps son menos eficientes que las large caps según la PE? Comparación
en dos niveles: los índices S&P 600 y S&P 500 (1994-2025, libre de survivorship
bias) y sus ~1.100 acciones constituyentes (2015-2025), con diferencias pareadas,
CIs por bootstrap y análisis por deciles.

## Datos

- **Nivel índices**: S&P 600 y S&P 500 diarios vía yfinance (`fetch_data.py`).
- **Nivel acciones**: constituyentes actuales del S&P 600 (snapshot incluido)
  vía `fetch_stocks.py`, más los precios del S&P 500 del experimento e4:
  **correr antes `e4-ml-crashes/fetch_data.py`** (e5 los reutiliza).

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py        # índices
python fetch_stocks.py      # acciones del S&P 600
python run_experiment.py    # nivel índices (m=3; `4` como argumento para m=4)
python run_multiasset.py    # nivel acciones (ídem)
python gen_charts.py
python gen_charts_multiasset.py
```

## Configuración

PE normalizada, m=3 y tau=1 (robustez de resolución con m=4), ventanas W=252
(anual) y W=504 (bienal). Inferencia sobre unidades independientes: ventanas
no superpuestas con bootstrap por bloques anuales (índices) y bootstrap
remuestreando acciones (nivel acciones). Filtros pre-declarados de historia
mínima y precio mediano >= 5 USD. Seed 42.

## Outputs

`outputs/results*.json`, `outputs/multiasset_results*.json`, brechas anuales en
CSV y `outputs/e5_*.png`.
