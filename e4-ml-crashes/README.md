# e4 — Entropía como feature en modelos predictivos de ML (Sección 5.4)

¿La PE aporta valor predictivo incremental como feature de un modelo de ML,
compitiendo contra variables clásicas de mercado? Regresión de la severidad
final de eventos de caída sobre una sección cruzada de ~500 acciones del
S&P 500 (2000-2025, ~20.000 eventos), con validación walk-forward de 8 folds,
sample weights por episodio sistémico y bootstrap por clusters para los CIs.

## Datos

Precios diarios de los constituyentes actuales del S&P 500 (snapshot de
Wikipedia incluido como CSV) descargados con yfinance. `fetch_data.py` genera
`data/prices/` (~500 parquets); todo re-descargable, nada versionado.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py                # descarga precios y snapshot del universo
python eda_coverage.py              # EDA: cobertura del panel
python eda_events.py                # EDA: dimensionamiento de eventos
python build_features.py            # dataset model-ready (1 fila por evento)
python run_models.py --no-wandb     # walk-forward RF/GB, baseline vs con PE
python run_inference.py             # CIs por bootstrap de clusters
python run_sp500_index.py --no-wandb  # sub-experimento: eventos del índice
python gen_charts.py
```

El flag `--no-wandb` corre todo sin tracking externo (vía recomendada). El uso
de Weights & Biases es opcional: requiere un `.env` local con la API key.

## Docker

```bash
docker build -f e4-ml-crashes/Dockerfile -t e4-ml .   # desde la raíz del repo
docker run -v ./e4-ml-crashes/data:/app/e4-ml-crashes/data \
           -v ./e4-ml-crashes/outputs:/app/e4-ml-crashes/outputs \
           e4-ml python run_models.py --no-wandb
```

## Configuración

6 features point-in-time (baseline de 5 + PE con W en {140, 180}, m=3);
Random Forest y Gradient Boosting con grilla pre-declarada y tuning en
validación; embargo de 1 año; extensión WPE sobre el mejor modelo. Seed 42.

## Outputs

`outputs/tabla_final.csv`, `outputs/inference.json`, `outputs/charts/`.
Los modelos entrenados (`outputs/models/`) y las predicciones por fold
(`outputs/preds/`) se regeneran al correr y no se versionan por tamaño.
