# -*- coding: utf-8 -*-
"""
Regressao com redes neurais para estimar chuva.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from preprocessamento import FEATURE_COLUMNS, TARGET_COLUMN, carregar_dataset_modelagem


################## Preprocessamento ##################

base = carregar_dataset_modelagem()

cols_previsores = FEATURE_COLUMNS
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

scaler_x = StandardScaler()
previsores_treinamento = scaler_x.fit_transform(previsores_treinamento)
previsores_teste = scaler_x.transform(previsores_teste)

scaler_y = StandardScaler()
objetivo_treinamento = scaler_y.fit_transform(objetivo_treinamento)
objetivo_teste_scaled = scaler_y.transform(objetivo_teste)

################## Regressao com Redes Neurais ##################

regressor = MLPRegressor(
    activation="relu",
    max_iter=300,
    verbose=False,
    hidden_layer_sizes=(10, 9),
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=0,
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
    plt.figure(figsize=(12, 12))
    plt.imshow(regressor.coefs_[0], interpolation="none", cmap="viridis", aspect="auto")
    plt.yticks(range(len(cols_previsores)), cols_previsores, fontsize=7)
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
