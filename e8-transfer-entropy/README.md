# e8 — Entropía de transferencia entre sectores (Sección 5.8)

¿Existe flujo rezagado de información entre los sectores del mercado
estadounidense, y es asimétrico? Effective transfer entropy (ETE) entre los
11 ETFs sectoriales SPDR con datos diarios 1998-2025: 110 pares dirigidos,
significancia por block bootstrap de la fuente y corrección por multiplicidad
(Benjamini-Hochberg), ranking de emisores y receptores netos.

## Datos

Reutiliza los parquets del experimento e6: **correr antes
`e6-sector-entropy/fetch_data.py`**. Este experimento no descarga nada propio.

## Pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_te.py       # ETE por par, bootstrap, FDR, crisis vs calma, serie anual
python gen_charts.py   # heatmap de la red, robustez de 9 sectores, crisis, anual
```

## Configuración

Config única, la estándar de la literatura financiera daily: 3 estados por
terciles de cada serie, un día de historia (k=l=1), TE en bits. Corrección de
sesgo effective (100 barajadas de la fuente), nula por block bootstrap (bloques
de 20 días, 1.000 réplicas), FDR al 5% sobre los 110 tests. Cada par se computa
sobre la historia común de ambas series. Seed 42.

## Outputs

`outputs/results.json` (pares, p-values, flujo neto, crisis, serie anual),
matrices CSV y los charts de la sección (incluye la robustez con los 9 sectores
de historia completa).
