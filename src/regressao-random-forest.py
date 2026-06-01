# -*- coding: utf-8 -*-
"""
Regressao com random forest para estimar chuva.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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

################## Regressao com Random Forest ##################

regressor = RandomForestRegressor(
    n_estimators=100,  # número de árvores
    max_features=10,  # qtd de características
    max_depth=15,
    random_state=0,
    n_jobs=-1,  # usar todos os cores disponíveis para acelerar
)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento.values.ravel())

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

# Visualizando a importancia das caracteristicas
try:
    n_features = previsores.shape[1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(n_features), regressor.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cols_previsores)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Importancia das Caracteristicas - Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance_random_forest.png", dpi=100, bbox_inches="tight")
    print("Grafico de importancia das features salvo em: feature_importance_random_forest.png")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar grafico: {e}")
