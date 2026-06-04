# -*- coding: utf-8 -*-
"""
Regressao com SVM linear escalavel para estimar chuva.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
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

################## Regressao com SVM ##################

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SGDRegressor(
    loss="squared_error",
    penalty="elasticnet",
    alpha=1e-5,
    l1_ratio=0.15,
    max_iter=10000,
    random_state=0,
    ),
)

regressor = TransformedTargetRegressor(
    regressor=pipeline,
    transformer=StandardScaler(),
)

# Treinamento
regressor.fit(previsores_treinamento, objetivo_treinamento.values.ravel())

# Teste
previsoes = regressor.predict(previsores_teste)

################## Avaliacao dos resultados ##################

score = metrics.r2_score(objetivo_teste, previsoes)
mae = metrics.mean_absolute_error(objetivo_teste, previsoes)
mse = metrics.mean_squared_error(objetivo_teste, previsoes)
rmse = np.sqrt(mse)

print("Score:", score)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

################## Visualizacao ##################

try:
    limite = max(float(objetivo_teste.max().iloc[0]), float(np.max(previsoes)))

    plt.figure(figsize=(8, 6))
    plt.scatter(objetivo_teste, previsoes, alpha=0.35)
    plt.plot([0, limite], [0, limite], color="red", linewidth=2, label="Previsao ideal")
    plt.title("SVM Linear - Observado x Previsto")
    plt.xlabel("Chuva observada")
    plt.ylabel("Chuva prevista")
    plt.legend()
    plt.tight_layout()
    plt.savefig("svm_regression.png", dpi=100, bbox_inches="tight")
    print("Grafico SVM salvo em: svm_regression.png")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar grafico: {e}")
