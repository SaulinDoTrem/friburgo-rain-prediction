# -*- coding: utf-8 -*-
"""
Gera graficos para a secao de resultados.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from experimentos_regressao import criar_pipeline_base
from preprocessamento import FEATURE_COLUMNS, TARGET_COLUMN, carregar_dataset_modelagem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

RESULTADOS_MODELOS = [
    ("Redes Neurais", "hidden_layer_sizes=(64, 32)", 0.3796936, 1.028910),
    ("Redes Neurais", "hidden_layer_sizes=(10, 9)", 0.362342, 1.043202),
    ("Random Forest", "n_estimators=100, max_depth=15", 0.3405416, 1.060884),
    ("Random Forest", 'n_estimators=200, max_features="sqrt"', 0.336288, 1.064300),
    ("Random Forest", "n_estimators=200, max_features=0.5", 0.335344, 1.065057),
    ("Regressão Polinomial", "degree=2, alpha=10.0", 0.279403, 1.108972),
    ("Regressão Polinomial", "degree=2, alpha=100.0", 0.279041, 1.109250),
    ("Árvore de Decisão", "max_depth=18", 0.241634, 1.137663),
    ("Regressão Linear Múltipla", "LinearRegression", 0.2400749, 1.138832),
    ("Árvore de Decisão", "max_depth=15", 0.223642, 1.151080),
    ("Árvore de Decisão", "max_depth=12", 0.223637, 1.151083),
    ("Redes Neurais", "hidden_layer_sizes=(128, 64, 32)", 0.222934, 1.151604),
    ("SVM Linear", "SGD squared_error", 0.192792, 1.173727),
    ("SVM Linear", "SGD epsilon_insensitive", 0.158181, 1.198626),
    ("SVM Linear", "SGD huber", 0.154323, 1.201370),
    ("Regressão Polinomial", "degree=3", 0.132486, 1.216782),
]


def modelo_rede_neural():
    return TransformedTargetRegressor(
        regressor=criar_pipeline_base(
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                batch_size=512,
                learning_rate="adaptive",
                learning_rate_init=0.001,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=0,
            )
        ),
        transformer=StandardScaler(),
    )


def modelo_random_forest():
    return criar_pipeline_base(
        RandomForestRegressor(
            n_estimators=100,
            max_features=15,
            max_depth=15,
            min_samples_leaf=5,
            random_state=0,
            n_jobs=-1,
        )
    )


def carregar_resultados_experimentos():
    return pd.DataFrame(
        RESULTADOS_MODELOS,
        columns=["Método", "Configuração", "Score (R²)", "RMSE"],
    )


def rotulo_configuracao(configuracao):
    texto = str(configuracao)
    primeira_linha = texto.splitlines()[0] if texto and texto != "nan" else ""
    return primeira_linha[:55]


def preparar_rotulos_modelos(df):
    df = df.copy()
    df["rotulo"] = df.apply(
        lambda row: f"{row['Método']} - {rotulo_configuracao(row['Configuração'])}"
        if rotulo_configuracao(row["Configuração"]) not in {"", "-"}
        else str(row["Método"]),
        axis=1,
    )
    return df


def salvar_comparacao_r2(df):
    dados = preparar_rotulos_modelos(df).sort_values("Score (R²)", ascending=True)
    altura = max(7, 0.42 * len(dados))

    fig, ax = plt.subplots(figsize=(12, altura))
    bars = ax.barh(dados["rotulo"], dados["Score (R²)"], color="#2f6f9f")
    ax.set_title("Comparação do Score (R²) dos modelos")
    ax.set_xlabel("Score (R²)")
    ax.set_ylabel("Modelo")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    path = FIGURES_DIR / "comparacao_r2_modelos.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def salvar_comparacao_rmse(df):
    dados = preparar_rotulos_modelos(df).sort_values("RMSE", ascending=False)
    altura = max(7, 0.42 * len(dados))

    fig, ax = plt.subplots(figsize=(12, altura))
    bars = ax.barh(dados["rotulo"], dados["RMSE"], color="#9f4d2f")
    ax.set_title("Comparação do RMSE dos modelos")
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Modelo")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    path = FIGURES_DIR / "comparacao_rmse_modelos.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def treinar_rede_neural(x_treino, y_treino):
    modelo = modelo_rede_neural()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(x_treino, y_treino)
    return modelo


def salvar_observado_vs_previsto(y_teste, previsoes):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_teste, previsoes, alpha=0.25, s=10, color="#2f6f9f")
    limite_min = min(float(np.min(y_teste)), float(np.min(previsoes)))
    limite_max = max(float(np.max(y_teste)), float(np.max(previsoes)))
    ax.plot([limite_min, limite_max], [limite_min, limite_max], color="#c43d3d", lw=2)
    ax.set_title("Rede Neural: valor observado vs. valor previsto")
    ax.set_xlabel("Valor observado")
    ax.set_ylabel("Valor previsto")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = FIGURES_DIR / "rede_neural_observado_vs_previsto.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def salvar_residuos(y_teste, previsoes):
    residuos = y_teste - previsoes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(previsoes, residuos, alpha=0.25, s=10, color="#2f6f9f")
    ax.axhline(0, color="#c43d3d", lw=2)
    ax.set_title("Rede Neural: resíduos vs. valores previstos")
    ax.set_xlabel("Valor previsto")
    ax.set_ylabel("Resíduo")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    path = FIGURES_DIR / "rede_neural_residuos.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def salvar_importancias_random_forest(x_treino, y_treino):
    modelo = modelo_random_forest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(x_treino, y_treino)

    rf = modelo.named_steps["randomforestregressor"]
    importancias = pd.DataFrame(
        {
            "Variável": FEATURE_COLUMNS,
            "Importância": rf.feature_importances_,
        }
    ).sort_values("Importância", ascending=False).head(15)
    importancias = importancias.sort_values("Importância", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(importancias["Variável"], importancias["Importância"], color="#4f7f3f")
    ax.set_title("Random Forest: 15 variáveis mais importantes")
    ax.set_xlabel("Importância")
    ax.set_ylabel("Variável")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    path = FIGURES_DIR / "random_forest_top15_importancias.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    base = carregar_dataset_modelagem().dropna(subset=[TARGET_COLUMN])
    x = base[FEATURE_COLUMNS]
    y = base[TARGET_COLUMN]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
    )

    arquivos = []
    df_resultados = carregar_resultados_experimentos()
    arquivos.append(salvar_comparacao_r2(df_resultados))
    arquivos.append(salvar_comparacao_rmse(df_resultados))

    rede = treinar_rede_neural(x_treino, y_treino)
    previsoes = rede.predict(x_teste)
    arquivos.append(salvar_observado_vs_previsto(y_teste.to_numpy(), previsoes))
    arquivos.append(salvar_residuos(y_teste.to_numpy(), previsoes))

    arquivos.append(salvar_importancias_random_forest(x_treino, y_treino))

    melhor_rede = df_resultados[
        (df_resultados["Método"] == "Redes Neurais")
        & (df_resultados["Configuração"] == "hidden_layer_sizes=(64, 32)")
    ].iloc[0]

    print("Artefatos gerados:")
    for arquivo in arquivos:
        print(f"- {arquivo.relative_to(PROJECT_ROOT)}")
    print()
    print("Rede Neural (64, 32) usada nos graficos comparativos:")
    print(f"- Score (R²): {melhor_rede['Score (R²)']:.6f}")
    print(f"- RMSE: {melhor_rede['RMSE']:.6f}")


if __name__ == "__main__":
    main()
