# common/

Módulo compartido para los experimentos de la tesis. Centraliza el cálculo de
métricas de Teoría de la Información (PE/WPE), descubrimiento causal con
DirectLiNGAM con rigor temporal (block bootstrap, no-gausianidad, rolling), y
carga de datos con cache local.

## Estructura

| Archivo | Contenido |
|---|---|
| `entropy.py` | PE y WPE unificadas (`perm_entropy`, `rolling_perm_entropy`), diagnóstico `tie_rate`. |
| `causality.py` | DirectLiNGAM + utilidades de rigor: `fit_lingam`, `gaussianity_test`, `block_bootstrap_lingam`, `rolling_lingam`, `edge_stability`. |
| `data.py` | `load_prices` con cache parquet en `.cache/`, `build_factor_panel` (7 factores estándar), `difference_nonstationary` (ADF + diff). |
| `stats.py` | `holm_bonferroni` para corrección multi-test, `kruskal_blocks` para K-W con bloques no-superpuestos. |

## Instalación

```bash
cd other/experiments/common
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso desde otros experimentos

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    load_prices, build_factor_panel,
    fit_lingam, gaussianity_test,
    block_bootstrap_lingam, rolling_lingam,
)

prices = load_prices("SPY", "1993-01-29", "2025-01-01")
factors_pe = build_factor_panel(prices, weighted_pe=False)
factors_wpe = build_factor_panel(prices, weighted_pe=True)

# DirectLiNGAM con bootstrap respetuoso de autocorrelación
boot = block_bootstrap_lingam(factors_pe, block_size=30, n_boot=500)

# Rolling para test AMH
rolling = rolling_lingam(factors_pe, window=500, step=126)
```

## Justificación metodológica

- **Block bootstrap:** el bootstrap i.i.d. estándar (incluido el de la librería
  `lingam`) destruye la autocorrelación de las series financieras y subestima
  la incertidumbre de los edges causales. El moving-block bootstrap preserva
  estructura temporal dentro de cada bloque.
- **Test de no-gausianidad:** DirectLiNGAM (Shimizu et al. 2011) es identificable
  sólo si los residuos son no-gaussianos. El test (Anderson-Darling o
  Jarque-Bera) valida ese supuesto antes de interpretar el grafo.
- **Rolling LiNGAM:** la AMH (Lo 2004) predice que la estructura del mercado
  cambia con el tiempo. Si la dirección causal es estable a través de ventanas,
  el hallazgo es robusto; si cambia, es evidencia AMH directa.
