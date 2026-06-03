# -*- coding: utf-8 -*-
"""
Regressao linear (uma variavel) para estimar chuva.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from preprocessamento import (
    SIMPLE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    carregar_dataset_modelagem,
)


################## Preprocessamento ##################

base = carregar_dataset_modelagem()

cols_previsores = SIMPLE_FEATURE_COLUMNS
col_objetivo = TARGET_COLUMN

base = base[cols_previsores + [col_objetivo]].dropna(subset=[col_objetivo])

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

################## Visualizacao dos dados ##################

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(previsores_treinamento, objetivo_treinamento, alpha=0.5)
plt.title("Regressao Linear - Dados de Treinamento")
plt.xlabel("Chuva na hora anterior")
plt.ylabel("Chuva")

plt.subplot(1, 2, 2)
plt.scatter(previsores_teste, objetivo_teste, alpha=0.5)
plt.title("Regressao Linear - Dados de Teste")
plt.xlabel("Chuva na hora anterior")
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
    previsores_imputados = imputer.transform(previsores)
    x_plot = np.linspace(previsores_imputados.min(), previsores_imputados.max(), 200)
    x_plot = x_plot.reshape(-1, 1)

    plt.figure(figsize=(10, 6))
    plt.scatter(previsores_imputados, objetivo, alpha=0.5, label="Dados observados")
    plt.plot(
        x_plot,
        regressor.predict(x_plot),
        color="red",
        linewidth=2,
        label="Reta de regressao",
    )
    plt.title("Regressao Linear Simples")
    plt.xlabel("Chuva na hora anterior")
    plt.ylabel("Chuva")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_line.png", dpi=100, bbox_inches="tight")
    print("Grafico de regressao linear salvo em: linear_regression_line.png")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar grafico: {e}")
