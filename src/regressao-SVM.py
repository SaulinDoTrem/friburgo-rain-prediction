# -*- coding: utf-8 -*-
"""
Regressao com SVM para estimar chuva (usando uma variavel - temperatura).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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

# Padronizacao
scaler_x = StandardScaler()
previsores_treinamento = scaler_x.fit_transform(previsores_treinamento)
previsores_teste = scaler_x.transform(previsores_teste)

scaler_y = StandardScaler()
objetivo_treinamento = scaler_y.fit_transform(objetivo_treinamento)
objetivo_teste_scaled = scaler_y.transform(objetivo_teste)

################## Regressao com SVM ##################

regressor = SVR(kernel="rbf", C=1, gamma="scale", epsilon=0.1)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento.ravel())

# Teste (com despadronizacao)
previsoes = regressor.predict(previsores_teste)
previsoes_escala_original = scaler_y.inverse_transform(previsoes.reshape(-1, 1))

################## Avaliacao dos resultados ##################

score = metrics.r2_score(objetivo_teste_scaled, previsoes)

mae = metrics.mean_absolute_error(objetivo_teste, previsoes_escala_original)
mse = metrics.mean_squared_error(objetivo_teste, previsoes_escala_original)
rmse = np.sqrt(mse)

print("Score:", score)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

################## Visualizacao ##################

try:
    # Criando dados para plot
    X_plot = np.arange(previsores.min().values[0], previsores.max().values[0], 0.1)
    X_plot = X_plot.reshape(-1, 1)
    Y_plot = regressor.predict(scaler_x.transform(X_plot))
    Y_plot = scaler_y.inverse_transform(Y_plot.reshape(-1, 1))
    
    # Plot sem padronizacao
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(previsores_treinamento, objetivo_treinamento, alpha=0.5, label="Dados de treino (padronizados)")
    plt.plot(scaler_x.transform(X_plot), regressor.predict(scaler_x.transform(X_plot)), color="red", label="Previsoes")
    plt.title("Regressao com SVM (Dados Padronizados)")
    plt.xlabel("Temperatura (padronizada)")
    plt.ylabel("Chuva (padronizada)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(previsores.values, objetivo.values, alpha=0.5, label="Dados de treino")
    plt.plot(X_plot, Y_plot, color="red", linewidth=2, label="Previsoes")
    plt.title("Regressao com SVM (Escala Original)")
    plt.xlabel("Temperatura (inst)")
    plt.ylabel("Chuva")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("svm_regression.png", dpi=100, bbox_inches="tight")
    print("Grafico SVM salvo em: svm_regression.png")
    plt.close()
    
except Exception as e:
    print(f"Erro ao gerar grafico: {e}")
