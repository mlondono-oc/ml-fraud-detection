"""
Inferencia del modelo de detección de transacciones fraudulentas.

Carga los artefactos generados por el entrenamiento y aplica sobre
transacciones nuevas el mismo preprocesamiento, el modelo y el umbral
de decisión.

Uso:
    uv run python -m ml_fraud_detection.predict data/nuevas.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_fraud_detection.datasets import DATE_COLUMN

CONFIG_PATH = "config.json"


def load_artifacts(path: str) -> dict:
    """Carga el preprocesador, el modelo, el umbral y los códigos 'otros'."""
    return joblib.load(path)


def predict_proba(df: pd.DataFrame, artefactos: dict) -> np.ndarray:
    """
    Probabilidad de fraude de cada transacción.

    Las categorías no vistas en entrenamiento se asignan al grupo 'otros'.
    """
    X = (
        artefactos["preprocessor"]
        .transform(df)
        .fillna(artefactos["codigos_otros"])
    )

    model = artefactos["model"]
    proba = model.predict_proba(X)[:, 1]

    return proba


def main(config_path: str = CONFIG_PATH) -> None:
    parser = argparse.ArgumentParser(
        description="Predice fraude sobre transacciones nuevas."
    )
    parser.add_argument(
        "csv", help="Ruta al CSV con las transacciones a evaluar"
    )
    args = parser.parse_args()

    config = json.loads(Path(config_path).read_text())
    df = pd.read_csv(args.csv, parse_dates=[DATE_COLUMN])
    artefactos = load_artifacts(config["inference"]["artifacts"])

    proba = predict_proba(df, artefactos)

    salida = df.assign(
        proba_fraude=proba,
        prediccion=(proba >= artefactos["threshold"]).astype(int),
    )
    salida.to_csv(config["inference"]["predictions"], index=False)
    print(
        f"{len(salida):,} transacciones | rechazadas: {salida['prediccion'].sum():,}"
    )
    print(f"Predicciones guardadas en {config['inference']['predictions']}")


if __name__ == "__main__":
    main()
