# e3 — Entropía como señal temprana de crashes (Sección 5.3)

¿La caída de PE durante un evento de mercado se relaciona con su profundidad (H1)?
¿La PE al inicio del evento aporta información sobre la severidad final (H2)?
Eventos de pullback y crash detectados algorítmicamente sobre el S&P 500
(1928-2025): caída >= 5% desde un máximo local rolling de 252 días.

## Datos

S&P 500 daily desde 1928, descargado con yfinance (automático, cache local).

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_h1_grid.py      # H1: grid de 30 configuraciones (W x corte de drawdown)
python run_h1.py           # H1: configuración principal
python run_h2_insample.py  # H2: OLS in-sample, 4 configuraciones
python run_h2_ml.py        # H2 extendido: Random Forest (apéndice)

python gen_h1_main_chart.py
python gen_h2_main_chart.py
python gen_h2_appendix_charts.py
```

## Configuración

PE con m=3; H1 barre W en {75, 100, 120, 140, 160, 180} y cortes de drawdown en
{25%, 30%, 35%, 40%, sin corte}; H2 usa W en {140, 180} con y sin filtro de
crashes extremos. Regresiones OLS con errores HC3. Filtro de eventos: duración
pico-fondo >= 15 días (H1: >= 60). Semillas fijas.

## Outputs

`outputs/` con los resultados del grid (CSV), las regresiones (JSON) y los
charts de la sección y sus apéndices.
