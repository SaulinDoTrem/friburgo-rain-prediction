
import os
import glob
import pandas as pd

csv_dir = os.path.join(os.path.dirname(__file__), 'data', 'mes-a-mes')
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
linhas_sem_dado = 18
colunas_csv = [
	'data', # DD/MM/YYYY
	'hora', # UTC
	'temperatura_inst', # °C
	'temperatura_max', # °C
	'temperatura_min', # °C
	'umidade_inst', # %
	'umidade_max', # %
	'umidade_min', # %
	'ponto_orvalho_inst', # °C
	'ponto_orvalho_max', # °C
	'ponto_orvalho_min', # °C
	'pressao_inst', # hPa
	'pressao_max', # hPa
	'pressao_min', # hPa
	'vento_velocidade', # m/s
	'vento_direcao', # graus º
	'vento_rajada', # m/s
	'radiacao', # kj/m²
	'chuva' # mm
]

dfs = []
for file in csv_files:
	try:
		df = pd.read_csv(file, encoding='utf-8', sep=None, engine='python')
		dfs.append(df)
	except Exception as e:
		print(f'Erro ao ler {file}: {e}')

if dfs:
	df_all = pd.concat(dfs, ignore_index=True)
	df_all = df_all.iloc[linhas_sem_dado:, :]
	df_all.columns = colunas_csv
	out_path = os.path.join(os.path.dirname(__file__), 'data', 'estacao-salinas-completo.csv')
	df_all.to_csv(out_path, index=False)
	print(f'Arquivo salvo: {out_path}')
else:
	print('Nenhum arquivo CSV encontrado na pasta data/mes-a-mes.')
