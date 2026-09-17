# Modelo para la detección de transacciones fraudulentas

## Configuración del Entorno de desarrollo

Este repositorio utiliza `pyproject.toml` y **`uv`** para la gestión de dependencias. Sin embargo, mantiene compatibilidad total con la herramienta convencional `pip`.

### Arquitectura y Estructura del Proyecto

```text
ml-fraud-detection/
├── data/
│   └── dataset.csv          # Dataset original de transacciones
├── docs/                    # Artefactos y modelos entrenados
├── models/                  # Informe final de la solución
├── notebooks/
│   └── 01_eda.ipynb         # Análisis Exploratorio de Datos (EDA)
├── src/
│   ├── features.py          # Preprocesamiento y Feature Engineering
│   ├── train.py             # Script de entrenamiento y evaluación
│   └── predict.py           # Script de inferencia para datos nuevos
├── pyproject.toml           # Definición de dependencias
├── requirements.txt         # Export de dependencias congeladas
└── README.md
```

### Requisitos Previos

* **Python:** `>= 3.10`
* **Git**

### Opción A: Usando `uv` (Recomendado)

1. **Clonar e instalar dependencias:**

```bash
git clone <URL_DEL_REPOSITORIO>
cd ml-fraud-detection

# Sincroniza el ambiente .venv e instala dependencias
uv sync

```

2. **Registrar Kernel para Jupyter:**
```bash
uv run python -m ipykernel install --user --name=ml-fraud-env --display-name="Python (Fraud Detection)"

```

3. **Comandos de ejecución con `uv`:**

```bash
# Abrir entorno de exploración
uv run jupyter notebook

# Reentrenar el modelo
uv run python src/train.py

# Ejecutar inferencia sobre un dataset nuevo
uv run python src/predict.py data/dataset.csv

```

### Opción B: Usando `pip` y `venv` tradicional

1. **Crear y activar el entorno virtual:**
```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

```

2. **Instalar dependencias e ipykernel:**
```bash
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-fraud-env --display-name="Python (Fraud Detection)"

```
---
