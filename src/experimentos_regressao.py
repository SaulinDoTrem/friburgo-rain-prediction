# -*- coding: utf-8 -*-
"""
Executa varias configuracoes de modelos de regressao para estimar chuva.

O objetivo e comparar os metodos com o mesmo split, mesmo tratamento de dados
e as mesmas metricas usadas nos scripts individuais.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from preprocessamento import (
    FEATURE_COLUMNS,
    POLYNOMIAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    carregar_dataset_modelagem,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "resultados"


def criar_pipeline_base(estimator):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        estimator,
    )


def criar_pipeline_polinomial(degree, estimator):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        estimator,
    )


def formatar_parametros(estimator):
    parametros = estimator.get_params(deep=False)
    parametros_formatados = [
        f"{nome}={repr(valor)}" for nome, valor in parametros.items()
    ]
    return f"{estimator.__class__.__name__}({', '.join(parametros_formatados)})"


def separar_pipeline(modelo):
    tratamento_alvo = "-"
    modelo_base = modelo

    if isinstance(modelo, TransformedTargetRegressor):
        modelo_base = modelo.regressor
        tratamento_alvo = formatar_parametros(modelo.transformer)

    if isinstance(modelo_base, Pipeline):
        etapas_tratamento = [
            formatar_parametros(etapa)
            for _, etapa in modelo_base.steps[:-1]
        ]
        estimador = modelo_base.steps[-1][1]
    else:
        etapas_tratamento = []
        estimador = modelo_base

    tratamento_dados = " + ".join(etapas_tratamento) if etapas_tratamento else "-"
    configuracao_completa = formatar_parametros(estimador)

    return tratamento_dados, tratamento_alvo, configuracao_completa


def estimador_final(modelo):
    modelo_base = modelo.regressor if isinstance(modelo, TransformedTargetRegressor) else modelo
    if isinstance(modelo_base, Pipeline):
        return modelo_base.steps[-1][1], modelo_base.steps
    return modelo_base, []


def parametros_artigo(estimator, nomes):
    parametros = estimator.get_params(deep=False)
    linhas = []
    for nome in nomes:
        valor = parametros[nome]
        if isinstance(valor, str):
            valor = f'"{valor}"'
        linhas.append(f"{nome}={valor}")
    return linhas


def configuracao_artigo(modelo):
    estimator, steps = estimador_final(modelo)

    if isinstance(estimator, LinearRegression):
        return "-"

    if isinstance(estimator, Ridge):
        linhas = []
        tem_polinomial = False
        for _, step in steps:
            if isinstance(step, PolynomialFeatures):
                tem_polinomial = True
                linhas.extend(parametros_artigo(step, ["degree", "include_bias"]))
        if tem_polinomial:
            linhas.append(f"Ridge(alpha={estimator.alpha})")
        else:
            linhas.extend(["Ridge", f"alpha={estimator.alpha}"])
        return linhas

    if isinstance(estimator, ElasticNet):
        return ["ElasticNet"] + parametros_artigo(
            estimator, ["alpha", "l1_ratio", "max_iter", "random_state"]
        )

    if isinstance(estimator, DecisionTreeRegressor):
        return parametros_artigo(
            estimator,
            ["max_depth", "min_samples_leaf", "random_state"],
        )

    if isinstance(estimator, RandomForestRegressor):
        return parametros_artigo(
            estimator,
            [
                "n_estimators",
                "max_features",
                "max_depth",
                "min_samples_leaf",
                "random_state",
                "n_jobs",
            ],
        )

    if isinstance(estimator, SGDRegressor):
        nomes = [
            "loss",
            "penalty",
            "alpha",
            "l1_ratio",
            "epsilon",
            "learning_rate",
            "eta0",
            "max_iter",
            "tol",
            "random_state",
        ]
        parametros = estimator.get_params(deep=False)
        return [linha for linha in parametros_artigo(estimator, nomes) if not linha.endswith("=None")]

    if isinstance(estimator, MLPRegressor):
        return parametros_artigo(
            estimator,
            [
                "hidden_layer_sizes",
                "activation",
                "solver",
                "alpha",
                "batch_size",
                "learning_rate",
                "learning_rate_init",
                "max_iter",
                "early_stopping",
                "validation_fraction",
                "n_iter_no_change",
                "random_state",
                "verbose",
            ],
        )

    return formatar_parametros(estimator)


def configuracoes_experimentos(quick=False):
    experimentos = [
        {
            "metodo": "Regressao Linear Multipla",
            "configuracao": "LinearRegression",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(LinearRegression()),
        },
        {
            "metodo": "Regressao Linear Multipla",
            "configuracao": "Ridge(alpha=1.0)",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(Ridge(alpha=1.0)),
        },
        {
            "metodo": "Regressao Linear Multipla",
            "configuracao": "Ridge(alpha=10.0)",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(Ridge(alpha=10.0)),
        },
        {
            "metodo": "Regressao Linear Multipla",
            "configuracao": "ElasticNet(alpha=0.001,l1_ratio=0.15)",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=5000, random_state=0)
            ),
        },
        {
            "metodo": "Regressao Polinomial",
            "configuracao": "degree=2,Ridge(alpha=10.0)",
            "features": POLYNOMIAL_FEATURE_COLUMNS,
            "modelo": criar_pipeline_polinomial(2, Ridge(alpha=10.0)),
        },
        {
            "metodo": "Regressao Polinomial",
            "configuracao": "degree=2,Ridge(alpha=100.0)",
            "features": POLYNOMIAL_FEATURE_COLUMNS,
            "modelo": criar_pipeline_polinomial(2, Ridge(alpha=100.0)),
        },
        {
            "metodo": "Regressao Polinomial",
            "configuracao": "degree=3,Ridge(alpha=100.0)",
            "features": POLYNOMIAL_FEATURE_COLUMNS,
            "modelo": criar_pipeline_polinomial(3, Ridge(alpha=100.0)),
        },
        {
            "metodo": "Arvore de Decisao",
            "configuracao": "max_depth=12,min_samples_leaf=50",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                DecisionTreeRegressor(
                    max_depth=12, min_samples_leaf=50, random_state=0
                )
            ),
        },
        {
            "metodo": "Arvore de Decisao",
            "configuracao": "max_depth=15,min_samples_leaf=50",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                DecisionTreeRegressor(
                    max_depth=15, min_samples_leaf=50, random_state=0
                )
            ),
        },
        {
            "metodo": "Arvore de Decisao",
            "configuracao": "max_depth=18,min_samples_leaf=100",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                DecisionTreeRegressor(
                    max_depth=18, min_samples_leaf=100, random_state=0
                )
            ),
        },
        {
            "metodo": "Random Forest",
            "configuracao": "n_estimators=100,max_features=15,max_depth=15,min_samples_leaf=5",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                RandomForestRegressor(
                    n_estimators=100,
                    max_features=15,
                    max_depth=15,
                    min_samples_leaf=5,
                    random_state=0,
                    n_jobs=-1,
                )
            ),
        },
        {
            "metodo": "Random Forest",
            "configuracao": "n_estimators=200,max_features=sqrt,max_depth=18,min_samples_leaf=5",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                RandomForestRegressor(
                    n_estimators=200,
                    max_features="sqrt",
                    max_depth=18,
                    min_samples_leaf=5,
                    random_state=0,
                    n_jobs=-1,
                )
            ),
        },
        {
            "metodo": "Random Forest",
            "configuracao": "n_estimators=200,max_features=0.5,max_depth=20,min_samples_leaf=10",
            "features": FEATURE_COLUMNS,
            "modelo": criar_pipeline_base(
                RandomForestRegressor(
                    n_estimators=200,
                    max_features=0.5,
                    max_depth=20,
                    min_samples_leaf=10,
                    random_state=0,
                    n_jobs=-1,
                )
            ),
        },
        {
            "metodo": "SVM Linear",
            "configuracao": "SGD squared_error,elasticnet,alpha=1e-5,l1_ratio=0.15",
            "features": FEATURE_COLUMNS,
            "modelo": TransformedTargetRegressor(
                regressor=criar_pipeline_base(
                    SGDRegressor(
                        loss="squared_error",
                        penalty="elasticnet",
                        alpha=1e-5,
                        l1_ratio=0.15,
                        max_iter=10000,
                        random_state=0,
                    )
                ),
                transformer=StandardScaler(),
            ),
        },
        {
            "metodo": "SVM Linear",
            "configuracao": "SGD epsilon_insensitive,epsilon=0.05,alpha=1e-5",
            "features": FEATURE_COLUMNS,
            "modelo": TransformedTargetRegressor(
                regressor=criar_pipeline_base(
                    SGDRegressor(
                        loss="epsilon_insensitive",
                        epsilon=0.05,
                        alpha=1e-5,
                        max_iter=10000,
                        tol=1e-5,
                        random_state=0,
                    )
                ),
                transformer=StandardScaler(),
            ),
        },
        {
            "metodo": "SVM Linear",
            "configuracao": "SGD huber,alpha=1e-5,adaptive",
            "features": FEATURE_COLUMNS,
            "modelo": TransformedTargetRegressor(
                regressor=criar_pipeline_base(
                    SGDRegressor(
                        loss="huber",
                        alpha=1e-5,
                        learning_rate="adaptive",
                        eta0=0.01,
                        max_iter=10000,
                        tol=1e-5,
                        random_state=0,
                    )
                ),
                transformer=StandardScaler(),
            ),
        },
        {
            "metodo": "Redes Neurais",
            "configuracao": "MLP(10,9),relu,early_stopping",
            "features": FEATURE_COLUMNS,
            "modelo": TransformedTargetRegressor(
                regressor=criar_pipeline_base(
                    MLPRegressor(
                        hidden_layer_sizes=(10, 9),
                        activation="relu",
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=10,
                        random_state=0,
                    )
                ),
                transformer=StandardScaler(),
            ),
        },
        {
            "metodo": "Redes Neurais",
            "configuracao": "MLP(64,32),relu,alpha=0.001",
            "features": FEATURE_COLUMNS,
            "modelo": TransformedTargetRegressor(
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
            ),
        },
        {
            "metodo": "Redes Neurais",
            "configuracao": "MLP(128,64,32),relu,alpha=0.001",
            "features": FEATURE_COLUMNS,
            "modelo": TransformedTargetRegressor(
                regressor=criar_pipeline_base(
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.001,
                        batch_size=512,
                        learning_rate="adaptive",
                        learning_rate_init=0.001,
                        max_iter=1200,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=25,
                        random_state=0,
                    )
                ),
                transformer=StandardScaler(),
            ),
        },
    ]

    if not quick:
        return experimentos

    return [
        exp
        for exp in experimentos
        if exp["configuracao"]
        in {
            "Ridge(alpha=10.0)",
            "degree=2,Ridge(alpha=100.0)",
            "max_depth=18,min_samples_leaf=100",
            "n_estimators=100,max_features=15,max_depth=15,min_samples_leaf=5",
            "SGD squared_error,elasticnet,alpha=1e-5,l1_ratio=0.15",
            "MLP(10,9),relu,early_stopping",
        }
    ]


def avaliar_modelo(experimento, x_treino, x_teste, y_treino, y_teste):
    inicio = time.perf_counter()
    modelo = experimento["modelo"]
    tratamento_dados, tratamento_alvo, configuracao_completa = separar_pipeline(modelo)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(x_treino[experimento["features"]], y_treino)

    previsoes = modelo.predict(x_teste[experimento["features"]])
    duracao = time.perf_counter() - inicio

    mse = metrics.mean_squared_error(y_teste, previsoes)
    return {
        "metodo": experimento["metodo"],
        "configuracao_resumida": experimento["configuracao"],
        "tratamento_dados": tratamento_dados,
        "tratamento_alvo": tratamento_alvo,
        "configuracao_completa": configuracao_completa,
        "configuracao_artigo": configuracao_artigo(modelo),
        "n_features": len(experimento["features"]),
        "score_r2": metrics.r2_score(y_teste, previsoes),
        "mae": metrics.mean_absolute_error(y_teste, previsoes),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "tempo_s": duracao,
    }


def salvar_resultados(resultados):
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(resultados).sort_values(
        ["rmse", "score_r2"], ascending=[True, False]
    )

    csv_path = RESULTS_DIR / "resultados_experimentos_regressao.csv"
    md_path = RESULTS_DIR / "resultados_experimentos_regressao.md"

    df_artigo = formatar_dataframe_artigo(df)
    df_csv = df_artigo.copy()
    df_csv["Configuração"] = df_csv["Configuração"].apply(
        lambda valor: "\n".join(valor) if isinstance(valor, list) else valor
    )

    df_csv.to_csv(csv_path, index=False)
    md_path.write_text(formatar_markdown_artigo(df_artigo), encoding="utf-8")

    return df, csv_path, md_path


def formatar_dataframe_artigo(df):
    nomes_metodos = {
        "Regressao Linear Multipla": "Regressão Linear Múltipla",
        "Regressao Polinomial": "Regressão Polinomial",
        "Arvore de Decisao": "Árvore de Decisão",
    }

    return pd.DataFrame(
        {
            "Método": df["metodo"].replace(nomes_metodos),
            "Configuração": df["configuracao_artigo"],
            "Tratamento de dados": "-",
            "Score (R²)": df["score_r2"],
            "RMSE": df["rmse"],
        }
    )


def formatar_valor_markdown(valor, coluna):
    if isinstance(valor, list):
        return "<br>".join(str(item) for item in valor)
    if isinstance(valor, float):
        if coluna == "RMSE":
            return f"{valor:.4f}"
        return f"{valor:.6f}"
    return str(valor)


def formatar_markdown_artigo(df):
    colunas = list(df.columns)
    linhas = [
        "| " + " | ".join(colunas) + " |",
        "| " + " | ".join(["---"] * len(colunas)) + " |",
    ]

    for _, row in df.iterrows():
        valores = []
        for coluna in colunas:
            valores.append(formatar_valor_markdown(row[coluna], coluna))
        linhas.append("| " + " | ".join(valores) + " |")

    return "\n".join(linhas) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Executa uma configuracao representativa por metodo.",
    )
    args = parser.parse_args()

    base = carregar_dataset_modelagem().dropna(subset=[TARGET_COLUMN])
    x = base[FEATURE_COLUMNS]
    y = base[TARGET_COLUMN]

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=0,
    )

    print(f"Total de registros: {len(base)}")
    print(f"Treino: {len(x_treino)} | Teste: {len(x_teste)}")
    print(f"Total de caracteristicas: {len(FEATURE_COLUMNS)}")
    print()

    resultados = []
    for indice, experimento in enumerate(configuracoes_experimentos(args.quick), 1):
        print(
            f"[{indice:02d}] {experimento['metodo']} - "
            f"{experimento['configuracao']}"
        )
        resultado = avaliar_modelo(experimento, x_treino, x_teste, y_treino, y_teste)
        resultados.append(resultado)
        print(
            "     "
            f"R2={resultado['score_r2']:.6f} | "
            f"RMSE={resultado['rmse']:.6f} | "
            f"MAE={resultado['mae']:.6f} | "
            f"tempo={resultado['tempo_s']:.1f}s"
        )

    df, csv_path, md_path = salvar_resultados(resultados)

    print()
    print("Melhores resultados:")
    print(df.head(10).to_string(index=False))
    print()
    print(f"CSV salvo em: {csv_path}")
    print(f"Markdown salvo em: {md_path}")


if __name__ == "__main__":
    main()
