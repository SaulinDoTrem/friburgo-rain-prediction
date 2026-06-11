# %%

# -*- coding: utf-8 -*-
"""
Regressao com random forest para estimar chuva.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)

################## Regressao com Random Forest ##################

regressor = RandomForestRegressor(
    n_estimators=100,
    max_features=15,
    max_depth=15,
    min_samples_leaf=5,
    random_state=0,
    n_jobs=-1,
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
    plt.figure(figsize=(10, 12))
    plt.barh(range(n_features), regressor.feature_importances_, align="center")
    plt.yticks(np.arange(n_features), cols_previsores, fontsize=7)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Importancia das Caracteristicas - Random Forest")
    plt.tight_layout()
    plt.savefig("feature_importance_random_forest.png", dpi=100, bbox_inches="tight")
    print("Grafico de importancia das features salvo em: feature_importance_random_forest.png")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar grafico: {e}")

# %%

################ VALIDACAO CRUZADA ################

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(((y_true - y_pred) / y_true)) * 100)

tscv = TimeSeriesSplit(n_splits=5)

previsores = base[cols_previsores]
objetivo = base[[col_objetivo]]

scores = []
maes = []
mses = []
rmses = []
mapes = []

for indice_treinamento, indice_teste in tscv.split(previsores):
    X_treino = previsores.iloc[indice_treinamento]
    X_teste = previsores.iloc[indice_teste]
    
    y_treino = objetivo.iloc[indice_treinamento]
    y_teste = objetivo.iloc[indice_teste]
    
    # 2. Padronização DENTRO do loop (evita Data Leakage)
    scaler_x = StandardScaler()
    X_treino_scaled = scaler_x.fit_transform(X_treino)
    X_teste_scaled = scaler_x.transform(X_teste)
    
    scaler_y = StandardScaler()
    y_treino_scaled = scaler_y.fit_transform(y_treino)
    
    # 3. Construção e Treinamento do Modelo
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_features=15,
        max_depth=15,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1,
    )
    
    # Treinamento
    regressor.fit(X_treino_scaled, y_treino_scaled.ravel())
    
    # 4. Previsões (feitas com os dados de teste padronizados)
    previsoes_scaled = regressor.predict(X_teste_scaled)
    
    # 5. Voltar as previsões para a escala original para calcular os erros reais
    previsoes = scaler_y.inverse_transform(previsoes_scaled.reshape(-1, 1))
    
    # 6. Avaliação
    
    score = metrics.r2_score(y_teste, previsoes)
    mae = metrics.mean_absolute_error(y_teste, previsoes)
    mse = metrics.mean_squared_error(y_teste, previsoes)
    rmse = np.sqrt(mse)
    # Remover linhas onde o valor real é zero para calcular o MAPE

    y_real_mape = y_teste.iloc[:, 0]
    y_pred_mape = previsoes.flatten()

    mask = y_real_mape != 0

    mape = metrics.mean_absolute_percentage_error(
        y_real_mape[mask],
        y_pred_mape[mask]
    )

    scores.append(score)
    maes.append(mae)
    mses.append(mse)
    rmses.append(rmse)
    mapes.append(mape)


# Métricas médias
scores = np.asarray(scores)
score_final_medio = scores.mean()
score_final_desvio_padrao = scores.std()

maes = np.asarray(maes)
mae_final_medio = maes.mean()
mae_final_desvio_padrao = maes.std()

mses = np.asarray(mses)
mse_final_medio = mses.mean()
mse_final_desvio_padrao = mses.std()

rmses = np.asarray(rmses)
rmse_final_medio = rmses.mean()
rmse_final_desvio_padrao = rmses.std()

mapes = np.asarray(mapes)
mape_final_medio = mapes.mean()
mape_final_desvio_padrao = mapes.std()

print("\n--- Resultados Finais ---")
print(f"R² Médio: {score_final_medio}")
print(f"Desvio Padrão R²: {score_final_desvio_padrao}")
print(f"MAPE Médio: {mape_final_medio}%")
print(f"Desvio Padrão MAPE: {mape_final_desvio_padrao}")
print(f"MAE Médio: {mae_final_medio}")
print(f"Desvio Padrão MAE: {mae_final_desvio_padrao}")
print(f"RMSE Médio: {rmse_final_medio}")
print(f"Desvio Padrão RMSE: {rmse_final_desvio_padrao}")
print(f"MSE Médio: {mse_final_medio}")
print(f"Desvio Padrão MSE: {mse_final_desvio_padrao}")

################## Gráficos de Avaliação #######################################
import seaborn as sns


sns.set_style("whitegrid")
sns.despine(top=True, right=False, left=False, bottom=False, offset=None, trim=False)

# Usando o modelo da última iteração (último fold) para plotar os gráficos
previsoes_treinamento_scaled = regressor.predict(X_treino_scaled)
previsoes_treinamento = scaler_y.inverse_transform(previsoes_treinamento_scaled.reshape(-1, 1))

# Cálculo dos erros (desvio relativo)
erros_treinamento = (y_treino - previsoes_treinamento) / y_treino
erros_teste = (y_teste - previsoes) / y_teste

# 1. Gráfico de Resíduos (Residplot)
plt.figure(figsize=(8, 5))
# .ravel() é usado para transformar a matriz 2D em 1D e evitar warnings do seaborn
# ax1 = sns.residplot(x=y_treino.ravel(), y=previsoes_treinamento.ravel(), lowess=False, color="blue", label='Treinamento')
# ax1 = sns.residplot(x=y_teste.ravel(), y=previsoes.ravel(), lowess=False, color="orange", label='Teste')
ax1 = sns.residplot(
    x=y_treino.values.ravel(),
    y=previsoes_treinamento.ravel(),
    lowess=False,
    color="blue",
    label='Treinamento'
)

ax1 = sns.residplot(
    x=y_teste.values.ravel(),
    y=previsoes.ravel(),
    lowess=False,
    color="orange",
    label='Teste'
)
ax1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
ax1.set_xlabel("Valor Real", fontsize=12)
ax1.set_ylabel("Resíduos", fontsize=12)
ax1.set_title("Gráfico de Resíduos")
plt.show()
# 2. Gráfico de Previsão vs Real
plt.figure(figsize=(8, 5))
plt.scatter(x=y_treino, y=previsoes_treinamento, alpha=0.5, label='Treinamento', color="blue")
plt.scatter(x=y_teste, y=previsoes, alpha=0.5, label='Teste', color="orange")
# Adicionando uma reta de referência (onde Previsão = Valor Real)
min_val = min(y_treino.min(), y_teste.min())
max_val = max(y_treino.max(), y_teste.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Previsão Perfeita')
plt.xlabel("Valor Real")
plt.ylabel("Previsão")
plt.title("Previsões vs Valores Reais")
plt.legend()
plt.show()
# 3. Histograma dos resíduos (Desvio Relativo)
# plt.figure(figsize=(8, 5))
# # Atualizado de sns.distplot para sns.histplot 
# ax2 = sns.histplot(erros_treinamento.ravel(), kde=True, stat="density", color="blue", label="Treinamento", alpha=0.4)
# ax2 = sns.histplot(erros_teste.ravel(), kde=True, stat="density", color="orange", label="Teste", alpha=0.4)
# ax2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=1, shadow=True, borderpad=1)
# ax2.set_xlabel("Desvio Relativo", fontsize=12)
# ax2.set_ylabel("Densidade", fontsize=12)
# ax2.set_title("Distribuição do Desvio Relativo")