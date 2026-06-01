# -*- coding: utf-8 -*-
"""
Regressao linear multipla para estimar chuva.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def carregar_base():
    data_path = (
        Path(__file__).resolve().parents[1]
        / "database"
        / "data"
        / "estacao-salinas-completo.csv"
    )
    base = pd.read_csv(data_path, decimal=",", na_values=["", " "])
    base["data"] = pd.to_datetime(base["data"], format="%d/%m/%Y", errors="coerce")
    base["hora"] = pd.to_numeric(base["hora"], errors="coerce")
    base["ano"] = base["data"].dt.year
    base["mes"] = base["data"].dt.month
    base["dia"] = base["data"].dt.day
    return base


################## Preprocessamento ##################

base = carregar_base()

cols_previsores = [
    "hora",
    "temperatura_inst",
    "temperatura_max",
    "temperatura_min",
    "umidade_inst",
    "umidade_max",
    "umidade_min",
    "ponto_orvalho_inst",
    "ponto_orvalho_max",
    "ponto_orvalho_min",
    "pressao_inst",
    "pressao_max",
    "pressao_min",
    "vento_velocidade",
    "vento_direcao",
    "vento_rajada",
    "radiacao",
    "ano",
    "mes",
    "dia",
]
col_objetivo = "chuva"

base = base.dropna(subset=[col_objetivo, "ano", "mes", "dia"])

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

################## Regressao Linear Multipla ##################

regressor = LinearRegression()

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento)

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliacao dos resultados ##################

score = regressor.score(previsores_teste, objetivo_teste)
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

print(f"\nIntercept: {coef_0}")
print("\nCoeficientes por variavel:")
for i, col in enumerate(cols_previsores):
    print(f"  {col}: {coeficientes[0][i]}")
