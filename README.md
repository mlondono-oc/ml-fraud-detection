# Modelo para la detección de transacciones fraudulentas

## Estructura

```text
ml-fraud-detection/
├── config.json              # Parámetros de datos, entrenamiento, negocio e inferencia
├── data/
│   └── dataset.csv          # Dataset original de transacciones
├── models/                  # Salidas del entrenamiento (se crea al entrenar)
├── notebooks/
│   ├── 01_eda.ipynb         # Análisis exploratorio
│   └── 02_evaluacion.ipynb  # Evaluación del modelo en validación y prueba
├── src/ml_fraud_detection/
│   ├── datasets.py          # Carga y partición temporal (train / val / test)
│   ├── features.py          # Pipeline de preprocesamiento
│   ├── metrics.py           # Ganancia de negocio y métricas
│   ├── train.py             # Entrenamiento y búsqueda de hiperparámetros
│   └── predict.py           # Inferencia sobre datos nuevos
├── pyproject.toml
└── requirements.txt         # Export de dependencias para pip
```

## Requisitos

* Python `>= 3.12`
* [`uv`](https://docs.astral.sh/uv/) (recomendado) o `pip`

> macOS: LightGBM requiere OpenMP (`brew install libomp`)

## Instalación

### Opción A: `uv`

```bash
git clone <URL_DEL_REPOSITORIO>
cd ml-fraud-detection
uv sync
uv run python -m ipykernel install --user --name=ml-fraud-env --display-name="Python (Fraud Detection)"
```

### Opción B: `pip`

```bash
git clone <URL_DEL_REPOSITORIO>
cd ml-fraud-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-fraud-env --display-name="Python (Fraud Detection)"
```

Con `pip`, omite `uv run` en los comandos siguientes.

## Ejecución

Todos los comandos se ejecutan desde la raíz del repositorio.

```bash
# 1. Entrenar: búsqueda de hiperparámetros evaluada en validación
uv run python -m ml_fraud_detection.train

# 2. Evaluar: abrir notebooks/02_evaluacion.ipynb con el kernel "Python (Fraud Detection)"
uv run jupyter notebook

# 3. Predecir sobre un dataset nuevo (mismas columnas que data/dataset.csv)
uv run python -m ml_fraud_detection.predict <ruta_al_csv>
```

El entrenamiento guarda en `models/`:

| Archivo | Contenido |
| --- | --- |
| `modelo.pkl` | Preprocesador, modelo, umbral de decisión y parámetros |
| `tuning.csv` | Resultados de cada candidato de la búsqueda |
| `predicciones_val.csv` | Probabilidades del modelo elegido en validación |
| `metricas_val.json` | Métricas del modelo elegido en validación |

La inferencia escribe `models/predicciones.csv` con la probabilidad de fraude y la decisión (`rechazar = 1`).

## Actualizar dependencias

```bash
uv add <paquete>
uv export --format requirements-txt --no-hashes -o requirements.txt
```
