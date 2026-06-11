# -*- coding: utf-8 -*-
"""
Preparo comum do dataset para os modelos de regressao.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "database"
    / "data"
    / "estacao-salinas-completo.csv"
)

TARGET_COLUMN = "chuva"

BASE_COLUMNS = [
    "data",
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
    TARGET_COLUMN,
]

MEASUREMENT_COLUMNS = BASE_COLUMNS[2:]

METEO_FEATURE_COLUMNS = [
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
    "vento_rajada",
    "radiacao",
]

ENGINEERED_FEATURE_COLUMNS = [
    "hora_do_dia",
    "dia_semana",
    "dia_do_ano",
    "fim_de_semana",
    "estacao_chuvosa",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "dia_ano_sin",
    "dia_ano_cos",
    "vento_sin",
    "vento_cos",
    "temperatura_amplitude",
    "umidade_amplitude",
    "ponto_orvalho_amplitude",
    "pressao_amplitude",
    "temperatura_media",
    "umidade_media",
    "ponto_orvalho_media",
    "pressao_media",
    "vento_rajada_ratio",
    "chuva_lag_1h",
    "chuva_lag_3h",
    "chuva_lag_6h",
    "chuva_acum_3h",
    "chuva_acum_6h",
    "chuva_acum_12h",
    "chuva_acum_24h",
    "chuva_media_6h",
    "chuva_media_24h",
    "choveu_1h",
    "choveu_6h",
]

FEATURE_COLUMNS = METEO_FEATURE_COLUMNS + [
    "ano",
    "mes",
    "dia",
] + ENGINEERED_FEATURE_COLUMNS

SIMPLE_FEATURE_COLUMNS = ["chuva_lag_1h"]

POLYNOMIAL_FEATURE_COLUMNS = [
    "temperatura_inst",
    "umidade_inst",
    "pressao_inst",
    "vento_velocidade",
    "vento_rajada",
    "radiacao",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "dia_ano_sin",
    "dia_ano_cos",
    "vento_sin",
    "vento_cos",
    "temperatura_amplitude",
    "umidade_amplitude",
    "pressao_amplitude",
    "chuva_lag_1h",
    "chuva_lag_3h",
    "chuva_acum_6h",
    "chuva_acum_24h",
    "choveu_6h",
]


def _to_numeric(series):
    if series.dtype == "object":
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
        series = series.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(series, errors="coerce")


def preparar_dataset_modelagem(base):
    base = base.copy()

    if list(base.columns[: len(BASE_COLUMNS)]) != BASE_COLUMNS:
        base = base.iloc[:, : len(BASE_COLUMNS)]
        base.columns = BASE_COLUMNS

    data_parseada = pd.to_datetime(
        base["data"], format="%d/%m/%Y", errors="coerce"
    )
    datas_invalidas = data_parseada.isna()
    if datas_invalidas.any():
        data_parseada.loc[datas_invalidas] = pd.to_datetime(
            base.loc[datas_invalidas, "data"], format="%Y-%m-%d", errors="coerce"
        )
    datas_invalidas = data_parseada.isna()
    if datas_invalidas.any():
        data_parseada.loc[datas_invalidas] = pd.to_datetime(
            base.loc[datas_invalidas, "data"], errors="coerce", dayfirst=True
        )
    base["data"] = data_parseada

    numeric_columns = [col for col in BASE_COLUMNS if col != "data"]
    for col in numeric_columns:
        base[col] = _to_numeric(base[col])

    base = base.dropna(subset=["data", "hora"])
    base = base.dropna(subset=MEASUREMENT_COLUMNS, how="all")

    base["hora_do_dia"] = (base["hora"] // 100).clip(0, 23)
    base["data_hora"] = base["data"] + pd.to_timedelta(base["hora_do_dia"], unit="h")
    base = (
        base.dropna(subset=["data_hora"])
        .sort_values("data_hora")
        .drop_duplicates("data_hora", keep="last")
        .reset_index(drop=True)
    )

    base["ano"] = base["data_hora"].dt.year
    base["mes"] = base["data_hora"].dt.month
    base["dia"] = base["data_hora"].dt.day
    base["dia_semana"] = base["data_hora"].dt.dayofweek
    base["dia_do_ano"] = base["data_hora"].dt.dayofyear
    base["fim_de_semana"] = (base["dia_semana"] >= 5).astype(int)
    base["estacao_chuvosa"] = base["mes"].isin([10, 11, 12, 1, 2, 3]).astype(int)

    base["hora_sin"] = np.sin(2 * np.pi * base["hora_do_dia"] / 24)
    base["hora_cos"] = np.cos(2 * np.pi * base["hora_do_dia"] / 24)
    base["mes_sin"] = np.sin(2 * np.pi * base["mes"] / 12)
    base["mes_cos"] = np.cos(2 * np.pi * base["mes"] / 12)
    base["dia_ano_sin"] = np.sin(2 * np.pi * base["dia_do_ano"] / 366)
    base["dia_ano_cos"] = np.cos(2 * np.pi * base["dia_do_ano"] / 366)

    vento_rad = np.deg2rad(base["vento_direcao"])
    base["vento_sin"] = np.sin(vento_rad)
    base["vento_cos"] = np.cos(vento_rad)

    base["temperatura_amplitude"] = base["temperatura_max"] - base["temperatura_min"]
    base["umidade_amplitude"] = base["umidade_max"] - base["umidade_min"]
    base["ponto_orvalho_amplitude"] = (
        base["ponto_orvalho_max"] - base["ponto_orvalho_min"]
    )
    base["pressao_amplitude"] = base["pressao_max"] - base["pressao_min"]

    base["temperatura_media"] = base[
        ["temperatura_inst", "temperatura_max", "temperatura_min"]
    ].mean(axis=1)
    base["umidade_media"] = base[["umidade_inst", "umidade_max", "umidade_min"]].mean(
        axis=1
    )
    base["ponto_orvalho_media"] = base[
        ["ponto_orvalho_inst", "ponto_orvalho_max", "ponto_orvalho_min"]
    ].mean(axis=1)
    base["pressao_media"] = base[["pressao_inst", "pressao_max", "pressao_min"]].mean(
        axis=1
    )
    base["vento_rajada_ratio"] = base["vento_rajada"] / base[
        "vento_velocidade"
    ].replace(0, np.nan)

    chuva_anterior = base[TARGET_COLUMN].shift(1)
    for lag in [1, 3, 6]:
        base[f"chuva_lag_{lag}h"] = base[TARGET_COLUMN].shift(lag)

    for window in [3, 6, 12, 24]:
        base[f"chuva_acum_{window}h"] = chuva_anterior.rolling(
            window, min_periods=1
        ).sum()

    base["chuva_media_6h"] = chuva_anterior.rolling(6, min_periods=1).mean()
    base["chuva_media_24h"] = chuva_anterior.rolling(24, min_periods=1).mean()
    base["choveu_1h"] = np.where(
        base["chuva_lag_1h"].isna(), np.nan, (base["chuva_lag_1h"] > 0).astype(float)
    )
    base["choveu_6h"] = np.where(
        base["chuva_acum_6h"].isna(),
        np.nan,
        (base["chuva_acum_6h"] > 0).astype(float),
    )

    return base.replace([np.inf, -np.inf], np.nan)


def carregar_dataset_modelagem(data_path=DATA_PATH):
    base = pd.read_csv(data_path, decimal=",", na_values=["", " "])
    return preparar_dataset_modelagem(base)
