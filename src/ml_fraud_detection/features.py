"""
Preprocesamiento y features engineering para el dataset de transacciones.

Consolida las transformaciones definidas en el análisis exploratorio (notebooks/01_eda.ipynb)
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer

# Constantes para la preparación de columnas
CATEGORICAL_COLUMNS = ["a", "g", "j", "o", "p"]
NUMERICAL_COLUMNS = ["b", "c", "d", "e", "f", "h", "l", "m", "n", "monto", "score", "hora"]
UMBRAL_OTROS = 20

# Funciones auxiliares para la preparación de columnas
def _add_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega la columna de hora al dataset de transacciones.
    """
    return df.assign(hora=df['fecha'].dt.hour)


def preprocessing_pipeline(umbral_otros: int = UMBRAL_OTROS) -> Pipeline:
    """
    Crea un pipeline de preprocesamiento para transformar las 
    columnas del dataset de transacciones.
    
    Args:
        umbral_otros (int): Umbral para agrupar categorías raras.
        
    Returns:
        Pipeline: Pipeline de preprocesamiento.
    """

    # Definir transformaciones para columnas categóricas y numéricas
    categorical_transformer = Pipeline(steps=[
        ("sin_registro", SimpleImputer(strategy='constant', fill_value='sin_registro')),
        ("codigo", OrdinalEncoder(
            min_frequency=umbral_otros,
            handle_unknown='use_encoded_value',
            unknown_value=np.nan))
    ])

    # Crear pipeline de preprocesamiento
    preprocessor = Pipeline(steps=[
        ("add_hour", FunctionTransformer(_add_hour)),
        ("column_transformer", ColumnTransformer(
            [("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
             ("numerical", "passthrough", NUMERICAL_COLUMNS)
            ], verbose_feature_names_out=False
        ))
    ]).set_output(transform="pandas")

    return preprocessor
        