# -*- coding: utf-8 -*-
"""
Regressao polinomial multipla para estimar chuva.
"""

import numpy as np
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from preprocessamento import (
    POLYNOMIAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    carregar_dataset_modelagem,
)


################## Preprocessamento ##################

base = carregar_dataset_modelagem()

cols_previsores = POLYNOMIAL_FEATURE_COLUMNS
col_objetivo = TARGET_COLUMN

base = base.dropna(subset=[col_objetivo])

previsores = base[cols_previsores]
objetivo = base[[col_objetivo]]

previsores_treinamento, previsores_teste, objetivo_treinamento, objetivo_teste = (
    train_test_split(
        previsores,
        objetivo,
        test_size=0.25,
        random_state=0,
    )
)

imputer = SimpleImputer(strategy="median")
previsores_treinamento = imputer.fit_transform(previsores_treinamento)
previsores_teste = imputer.transform(previsores_teste)

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)

################## Regressao Polinomial Multipla ##################

poly = PolynomialFeatures(degree=2, include_bias=False)
previsores_treinamento_poly = poly.fit_transform(previsores_treinamento)
previsores_teste_poly = poly.transform(previsores_teste)

regressor = Ridge(alpha=100.0)

# Treinamento
regressor.fit(previsores_treinamento_poly, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste_poly)

################## Avaliacao dos resultados ##################

score = regressor.score(previsores_teste_poly, objetivo_teste)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(mse)

print("Score:", score)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Parametros estimados para o modelo
coef_0 = regressor.intercept_
coeficientes = regressor.coef_
