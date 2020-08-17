import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class DataFrameImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = X.copy()
        return data.fillna(pd.Series([data[c].value_counts().index[0] if data[c].dtype == np.dtype('O') else data[c].mean() for c in data], index=data.columns))

class DataFrameNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        """
        Normalizando los datos numéricos
        """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = X.copy()
        for column in data.columns:
            if not data[column].dtype == np.dtype('O'):
                training_mean = data[column].mean()
                training_std = data[column].std()
                data[column] = (data[column] - training_mean) / training_std  # normalizando (usando el promedio y la desviación estándar de los datos de entrenamiento)

        return data

class DataFrameImputerNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data = X.copy()
        data = data.fillna(pd.Series([data[c].value_counts().index[0] if data[c].dtype == np.dtype('O') else data[c].mean() for c in data], index=data.columns))

        for column in data.columns:
            if not data[column].dtype == np.dtype('O'):
                training_mean = data[column].mean()
                training_std = data[column].std()
                data[column] = (data[column] - training_mean) / training_std  # normalizando (usando el promedio y la desviación estándar de los datos de entrenamiento)

        return data
