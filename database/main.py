import glob
import os
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "src"))

from preprocessamento import BASE_COLUMNS, preparar_dataset_modelagem


csv_dir = os.path.join(os.path.dirname(__file__), "data", "mes-a-mes")
csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="utf-8", sep=";", engine="python")
        df = df.iloc[:, : len(BASE_COLUMNS)]
        df.columns = BASE_COLUMNS
        dfs.append(df)
    except Exception as e:
        print(f"Erro ao ler {file}: {e}")

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = preparar_dataset_modelagem(df_all)
    out_path = os.path.join(
        os.path.dirname(__file__), "data", "estacao-salinas-completo.csv"
    )
    df_all.to_csv(out_path, index=False, decimal=",")
    print(f"Arquivo salvo: {out_path}")
    print(f"Linhas: {len(df_all)} | Colunas: {len(df_all.columns)}")
else:
    print("Nenhum arquivo CSV encontrado na pasta data/mes-a-mes.")
