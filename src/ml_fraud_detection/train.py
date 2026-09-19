"""
Entrenamiento del modelo de detección de transacciones fraudulentas.

Flujo: carga y partición temporal, preprocesamiento ajustado solo con
entrenamiento, búsqueda de hiperparámetros evaluada en validación, elección
del umbral que maximiza la ganancia y guardado de los artefactos del mejor
modelo.

Uso:
    uv run python -m ml_fraud_detection.train
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from ml_fraud_detection.datasets import (
    TARGET_COLUMN,
    load_dataset,
    out_of_time_split,
    split_summary,
)
from ml_fraud_detection.features import (
    CATEGORICAL_COLUMNS,
    preprocessing_pipeline,
)
from ml_fraud_detection.metrics import best_threshold, evaluate, profit_std

CONFIG_PATH = "config.json"


def train_model(
    params: dict, X_train: pd.DataFrame, y_train: pd.Series
) -> LGBMClassifier:
    """Entrena un LightGBM con los hiperparámetros dados."""
    model = LGBMClassifier(**params, verbose=-1)
    model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLUMNS)
    return model


def score_on_val(
    model: LGBMClassifier, X_val: pd.DataFrame, val: pd.DataFrame, negocio: dict
) -> tuple[float, dict]:
    """Elige el umbral que maximiza la ganancia en validación y evalúa con él."""
    monto, y_true = val["monto"].values, val[TARGET_COLUMN].values
    proba = model.predict_proba(X_val)[:, 1]
    umbral = best_threshold(monto, y_true, proba, **negocio)
    y_pred = (proba >= umbral).astype(int)

    metricas = evaluate(monto, y_true, y_pred, **negocio)
    metricas["ap"] = average_precision_score(y_true, proba)
    metricas["ganancia_std"] = profit_std(monto, y_true, y_pred, **negocio)
    return umbral, metricas


def main(config_path: str = CONFIG_PATH) -> None:
    config = json.loads(Path(config_path).read_text())
    negocio = config["business-rules"]
    training = config["training"]

    # 1. Carga y partición temporal; test queda reservado
    df = load_dataset(config["data"]["path"])
    train, val, _ = out_of_time_split(
        df, config["data"]["val_weeks"], config["data"]["test_weeks"]
    )
    print(split_summary(train=train, val=val).round(2), "\n")

    # 2. Preprocesamiento, ajustado solo con entrenamiento
    threshold_otros = config["features"]["umbral_otros"]
    preprocessor = preprocessing_pipeline(threshold_otros).fit(train)
    X_train = preprocessor.transform(train)

    # el grupo 'otros' recibe el último código de cada columna de
    # alta cardinalidad
    codigos_otros = X_train[["g", "j"]].max().astype(int).to_dict()
    X_val = preprocessor.transform(val).fillna(codigos_otros)

    # 3. Búsqueda de hiperparámetros: cada candidato se entrena en train
    #    y se evalúa en validación con su propio umbral óptimo
    candidatos = ParameterSampler(
        training["search_space"],
        training["n_iter"],
        random_state=training["random_state"],
    )

    resultados, modelos, parametros = [], [], []
    for i, params in enumerate(candidatos, start=1):
        params = {**params, "random_state": training["random_state"]}
        model = train_model(params, X_train, train[TARGET_COLUMN])
        umbral, metricas = score_on_val(model, X_val, val, negocio)

        # Añadir AP de entrenamiento para evaluar sobreajuste, sin afectar la
        # elección del mejor candidato, que se hace solo con validación
        metricas["ap_train"] = average_precision_score(
            train[TARGET_COLUMN], model.predict_proba(X_train)[:, 1]
        )

        resultados.append({**params, "umbral": umbral, **metricas})
        modelos.append(model)
        parametros.append(params)
        print(
            f"[{i}/{training['n_iter']}] ganancia val = {metricas['ganancia']:,.0f}"
            f" | recall monto = {metricas['recall_monto']:.3f}"
        )

    # 4. Selección: el candidato con mayor ganancia en validación
    tabla = pd.DataFrame(resultados)
    elegido = tabla["ganancia"].idxmax()
    mejor = {
        "model": modelos[elegido],
        "params": parametros[elegido],
        "umbral": tabla.loc[elegido, "umbral"],
        "metricas": tabla.loc[elegido, list(metricas)].to_dict(),
    }

    print(
        f"\nMejor ganancia {mejor['metricas']['ganancia']:,.0f} "
        f"± {mejor['metricas']['ganancia_std']:,.0f}"
    )

    tabla = tabla.sort_values("ganancia", ascending=False)
    ruta_tuning = Path(config["inference"]["tuning_results"])
    ruta_tuning.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta_tuning, index=False)

    # 5. Resumen del modelo elegido en validación
    print("\nMejores candidatos:")
    print(tabla.head(5).round(4).to_string(index=False), "\n")
    print("Hiperparámetros elegidos:", mejor["params"])
    print(f"Umbral elegido: {mejor['umbral']:.2f}")
    print(pd.Series(mejor["metricas"]).round(4).to_string(), "\n")

    # 6. Sobreajuste: brecha entre entrenamiento y validación
    ap_train = average_precision_score(
        train[TARGET_COLUMN], mejor["model"].predict_proba(X_train)[:, 1]
    )
    print(
        f"AP train {ap_train:.4f} | AP val {mejor['metricas']['ap']:.4f}"
        f" | brecha {ap_train - mejor['metricas']['ap']:.4f}\n"
    )

    # 7. Predicciones y métricas de validación, para el notebook de evaluación
    pd.DataFrame(
        {
            "fecha": val["fecha"].values,
            "monto": val["monto"].values,
            "score": val["score"].values,
            TARGET_COLUMN: val[TARGET_COLUMN].values,
            "proba": mejor["model"].predict_proba(X_val)[:, 1],
        }
    ).to_csv(config["inference"]["val_predictions"], index=False)

    Path(config["inference"]["val_metrics"]).write_text(
        json.dumps(
            {
                **mejor["metricas"],
                "ap_train": ap_train,
                "umbral": mejor["umbral"],
                "params": mejor["params"],
            },
            indent=2,
            default=float,
        )
    )

    # 8. Artefactos para inferencia
    ruta = Path(config["inference"]["artifacts"])
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": mejor["model"],
            "threshold": mejor["umbral"],
            "codigos_otros": codigos_otros,
            "params": mejor["params"],
        },
        ruta,
    )
    print(f"Artefactos guardados en {ruta}")
    print(f"Resultados del tuning guardados en {ruta_tuning}")


if __name__ == "__main__":
    main()
