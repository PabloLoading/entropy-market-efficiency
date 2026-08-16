# e7 — Entropía intradiaria por franja horaria (Sección 5.7)

¿Existen franjas del día de trading con más estructura ordinal que otras?
PE por franja horaria del S&P 500 (SPY) sobre 14 años de barras de 1 minuto
(2008-2021): patrones ordinales acumulados por franja a lo largo de ~3.300 días,
con CIs por bootstrap de días y tres comparaciones pre-declaradas
(apertura, mediodía, cierre).

## Datos

Dataset público de barras de 1 minuto de SPY (2008-2021). Descargarlo de
Kaggle: <https://www.kaggle.com/datasets/rockinbrock/spy-1-minute-data> y
colocar el CSV como `datasets/spy_1min_2008_2021_cleaned.csv` **en la raíz del
repo** (crear la carpeta `datasets/` si no existe). `prepare_data.py` hace la
limpieza (duplicados, zona horaria, sesión regular) y genera los parquets
locales.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py   # limpieza del CSV crudo -> data/*.parquet
python run_intraday.py   # PE por franja, 4 configs, bootstrap e inferencia
python gen_charts.py
```

## Configuración

Dos particiones del día (franjas horarias con barras de 5 min; grupos de 5 min
con barras de 1 min) por dos órdenes de patrón (m=3 principal, m=4
sensibilidad). Inferencia por bootstrap remuestreando días (2.000 réplicas),
con robustez por bloques mensuales. Significancia formal solo en las tres
comparaciones pre-declaradas. Semillas fijas.

## Outputs

`outputs/` con los perfiles por franja (JSON/CSV) y los charts de la sección
(perfil horario y curva de 5 minutos).
