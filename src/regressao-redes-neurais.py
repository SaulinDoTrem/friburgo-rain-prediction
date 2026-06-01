# -*- coding: utf-8 -*-
"""
Regressao com redes neurais para estimar chuva.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
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

scaler_x = StandardScaler()
previsores_treinamento = scaler_x.fit_transform(previsores_treinamento)
previsores_teste = scaler_x.transform(previsores_teste)

scaler_y = StandardScaler()
objetivo_treinamento = scaler_y.fit_transform(objetivo_treinamento)
objetivo_teste_scaled = scaler_y.transform(objetivo_teste)

################## Regressao com Redes Neurais ##################

regressor = MLPRegressor(
    activation="relu", max_iter=300, verbose=False, hidden_layer_sizes=(10, 9), random_state=0
)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento.ravel())

# Teste
previsoes = regressor.predict(previsores_teste)

# Despadronizacao
previsoes_escala_original = scaler_y.inverse_transform(previsoes.reshape(-1, 1))
objetivo_escala_original = scaler_y.inverse_transform(objetivo_teste_scaled.reshape(-1, 1))

################## Avaliacao dos resultados ##################

score = regressor.score(previsores_teste, objetivo_teste_scaled)

mae = metrics.mean_absolute_error(objetivo_escala_original, previsoes_escala_original)
mse = metrics.mean_squared_error(objetivo_escala_original, previsoes_escala_original)
rmse = np.sqrt(mse)

print("Score:", score)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Plotagem do "Mapa de calor" de uma rede neural
try:
    plt.figure(figsize=(12, 8))
    plt.imshow(regressor.coefs_[0], interpolation="none", cmap="viridis")
    plt.yticks(range(len(cols_previsores)), cols_previsores)
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    plt.colorbar()
    plt.title("Mapa de Calor - Primeira Camada da Rede Neural")
    plt.tight_layout()
    plt.savefig("neural_network_heatmap.png", dpi=100, bbox_inches="tight")
    print("Mapa de calor da rede neural salvo em: neural_network_heatmap.png")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar mapa de calor: {e}")
