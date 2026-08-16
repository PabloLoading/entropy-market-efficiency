# Experimentos: Teoría de la Información y eficiencia del mercado accionario de EE.UU.

Código de los ocho experimentos computacionales de la tesis de grado en Ingeniería
en Sistemas (Universidad ORT Uruguay, 2026). Cada carpeta corresponde a una sección
del Capítulo 5 del documento, donde se desarrollan la pregunta, el método, los
hallazgos y las limitaciones de cada experimento.

| Carpeta | Sección de la tesis | Tema |
|---|---|---|
| `e1-entropy-causality` | 5.1 | Causalidad entre entropía y dinámica del mercado (DirectLiNGAM) |
| `e2-entropy-evolution` | 5.2 | Evolución temporal de la eficiencia (1928-2025) |
| `e3-entropy-crashes`   | 5.3 | Entropía como señal temprana de crashes |
| `e4-ml-crashes`        | 5.4 | Entropía como feature en modelos predictivos de ML |
| `e5-small-caps`        | 5.5 | Entropía y capitalización bursátil (S&P 600 vs S&P 500) |
| `e6-sector-entropy`    | 5.6 | Entropía permutacional por sector y por mercado |
| `e7-intraday-entropy`  | 5.7 | Entropía intradiaria por franja horaria |
| `e8-transfer-entropy`  | 5.8 | Entropía de transferencia entre sectores |
| `common`               | --- | Módulo compartido: PE/WPE, bootstrap, DirectLiNGAM, carga de datos |

## Requisitos y setup

Python 3.12 o superior. Cada experimento tiene su propio `requirements.txt` con
versiones fijadas; el patrón de uso es un entorno virtual por experimento:

```bash
cd <experimento>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datos

No se versiona ningún dato de mercado. Cada experimento incluye scripts que
descargan sus datos de fuentes públicas (yfinance) y los cachean en `data/`.
Dos experimentos (e2 y e7) usan además un dataset público de barras de 1 minuto
de SPY que debe descargarse a mano; las instrucciones están en sus README.

Dependencias entre experimentos (por reuso de datos): correr el fetch de
`e4-ml-crashes` antes de `e5-small-caps` (fase de acciones), y el fetch de
`e6-sector-entropy` antes de `e8-transfer-entropy`.

## Reproducibilidad

Todos los experimentos fijan semillas para los procedimientos estocásticos
(bootstrap, remuestreos, modelos) y declaran sus configuraciones en el código.
Los outputs livianos (charts, resultados en JSON y tablas) están versionados
para poder verificar los números de la tesis sin re-ejecutar; los artefactos
pesados (datos crudos, modelos entrenados) se regeneran con los pipelines.

## Licencia

MIT (ver `LICENSE`).
