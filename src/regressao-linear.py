# -*- coding: utf-8 -*-
"""
Regressao linear (uma variavel) para estimar chuva.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
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
    return base


################## Preprocessamento ##################

base = carregar_base()

# Usando apenas temperatura como preditor (similar ao exemplo do plano de saude)
cols_previsores = ["temperatura_inst"]
col_objetivo = "chuva"

base = base[cols_previsores + [col_objetivo]].dropna()

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

# Visualizacao dos dados (treino)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(previsores_treinamento, objetivo_treinamento, alpha=0.5)
plt.title("Regressao Linear - Dados de Treinamento")
plt.xlabel("Temperatura (inst)")
plt.ylabel("Chuva")

# Visualizacao dos dados (teste)
plt.subplot(1, 2, 2)
plt.scatter(previsores_teste, objetivo_teste, alpha=0.5)
plt.title("Regressao Linear - Dados de Teste")
plt.xlabel("Temperatura (inst)")
plt.ylabel("Chuva")
plt.tight_layout()
plt.savefig("linear_regression_data.png", dpi=100, bbox_inches="tight")
plt.close()

################## Regressao Linear ##################

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
print(f"Coeficiente: {coeficientes[0]}")

# Visualizacao da reta de regressao
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(previsores, objetivo, alpha=0.5, label="Dados observados")
    plt.plot(
        previsores,
        regressor.predict(previsores),
        color="red",
        linewidth=2,
        label="Reta de regressao",
    )
    plt.title("Regressao Linear Simples")
    plt.xlabel("Temperatura (inst)")
    plt.ylabel("Chuva")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_line.png", dpi=100, bbox_inches="tight")
    print("Grafico de regressao linear salvo em: linear_regression_line.png")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar grafico: {e}")
