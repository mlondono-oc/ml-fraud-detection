"""
Métricas de negocio para evaluar el modelo de detección de transacciones
fraudulentas.

Regla del enunciado: cada transacción legítima aprobada deja una ganancia
del 25% de su monto, y cada fraude aprobado genera una pérdida del 100% de
su monto. Las transacciones rechazadas no generan ganancia ni pérdida.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def profit(
    amount: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margin: float = 0.25,
    fraud_loss: float = 1.0,
) -> float:
    """
    Calcula la ganancia de aprobar las transacciones indicadas.

    Args:
        amount (np.ndarray): Monto de cada transacción.
        y_true (np.ndarray): 1 si la transacción es fraude, 0 si es legítima.
        y_pred (np.ndarray): 1 si se predice fraude, 0 si se predice legítima.
        margin (float): Proporción del monto que gana una legítima aprobada.
        fraud_loss (float): Proporción del monto que pierde un fraude aprobado.

    Returns:
        float: Ganancia neta.
    """
    aprobada = y_pred == 0
    legitimas = aprobada & (y_true == 0)
    fraudes = aprobada & (y_true == 1)
    return margin * amount[legitimas].sum() - fraud_loss * amount[fraudes].sum()


def best_threshold(
    amount: np.ndarray,
    y_true: np.ndarray,
    proba: np.ndarray,
    margin: float = 0.25,
    fraud_loss: float = 1.0,
) -> float:
    """
    Busca el umbral de probabilidad que maximiza la ganancia.

    Se predice fraude cuando la probabilidad es mayor o igual al umbral.

    Returns:
        float: Umbral con mayor ganancia.
    """
    umbrales = np.arange(0.01, 1.0, 0.01)
    ganancias = [
        profit(amount, y_true, (proba >= u).astype(int), margin, fraud_loss)
        for u in umbrales
    ]
    return float(umbrales[np.argmax(ganancias)])


def evaluate(
    amount: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margin: float = 0.25,
    fraud_loss: float = 1.0,
) -> dict:
    """
    Resume el desempeño de las decisiones del modelo.

    La ganancia se compara contra la línea base de aprobar todas las
    transacciones.
    Las métricas de clasificación, derivadas de la matriz de confusión,
    se reportan.

    Args:
        y_pred (np.ndarray): 1 si se predice fraude (se rechaza),
        0 si se aprueba.

    Returns:
        dict: Ganancia, línea base, mejora, matriz de confusión y
        métricas de clasificación.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    ganancia = profit(amount, y_true, y_pred, margin, fraud_loss)
    base = profit(amount, y_true, np.zeros_like(y_pred), margin, fraud_loss)
    fraude_detectado = (y_true == 1) & (y_pred == 1)

    return {
        "ganancia": ganancia,
        "ganancia_aprobar_todo": base,
        "mejora": ganancia - base,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tasa_aprobacion": (tn + fn) / len(y_true),
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp > 0 else 0.0,
        "fpr": fp / (fp + tn),
        "recall_monto": amount[fraude_detectado].sum()
        / amount[y_true == 1].sum(),
    }
