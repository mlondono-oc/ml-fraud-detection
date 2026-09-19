"""
Carga y partición de dataset de transacciones para entrenamiento y evaluación
de modelos de detección de fraude.

Se aplica una partición temporal (out-of-time) y se hace por semanas completas
para simular un escenario de producción, donde los datos de entrenamiento son
anteriores a los datos de prueba.
"""

import pandas as pd

TARGET_COLUMN = "fraude"
DATE_COLUMN = "fecha"


def load_dataset(path: str) -> pd.DataFrame:
    """
    Carga el dataset de transacciones desde un archivo CSV ordenado por fecha.

    Args:
        path (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con los datos de transacciones.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(by=DATE_COLUMN).reset_index(drop=True)
    return df


def out_of_time_split(
    df: pd.DataFrame, val_weeks: int = 1, test_weeks: int = 2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Realiza una partición temporal (out-of-time) del dataset de transacciones
    en conjuntos de entrenamiento, validación y prueba.

    Args:
        df (pd.DataFrame): DataFrame con los datos de transacciones.
        val_weeks (int): Número de semanas para el conjunto de validación.
        test_weeks (int): Número de semanas para el conjunto de prueba.

    Returns:
        tuple: (train_df, val_df, test_df) DataFrames independientes.
    """
    semana = df[DATE_COLUMN].dt.to_period("W")
    semanas = semana.sort_values().unique()

    if val_weeks + test_weeks >= len(semanas):
        raise ValueError(
            "El número de semanas para validación y prueba excede el total de "
            "semanas disponibles."
        )

    corte_val = semanas[-(val_weeks + test_weeks)]
    corte_test = semanas[-test_weeks]

    train = df[semana < corte_val]
    val = df[(semana >= corte_val) & (semana < corte_test)]
    test = df[semana >= corte_test]

    return train, val, test


def split_summary(**dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Resume cada conjunto de la partición para verificarla y documentarla.

    Args:
        **dfs: DataFrames a resumir, nombrados
        (train=..., val=..., test=...).

    Returns:
        pd.DataFrame: Fechas, volumen y tasa de fraude de cada conjunto.
    """
    return pd.DataFrame(
        {
            nombre: {
                "desde": d[DATE_COLUMN].min().date(),
                "hasta": d[DATE_COLUMN].max().date(),
                "transacciones": len(d),
                "% del total": len(d) / sum(len(c) for c in dfs.values()) * 100,
                "tasa de fraude %": d[TARGET_COLUMN].mean() * 100,
            }
            for nombre, d in dfs.items()
        }
    ).T
