
import os
import glob
import pandas as pd

csv_dir = os.path.join(os.path.dirname(__file__), 'files')
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

dfs = []
for file in csv_files:
	try:
		df = pd.read_csv(file, encoding='utf-8', sep=None, engine='python')
		dfs.append(df)
	except Exception as e:
		print(f'Erro ao ler {file}: {e}')

if dfs:
	df_all = pd.concat(dfs, ignore_index=True)
	out_path = os.path.join(os.path.dirname(__file__), 'data', 'estacao-salinas.csv')
	df_all.to_csv(out_path, index=False)
	print(f'Arquivo salvo: {out_path}')
else:
	print('Nenhum arquivo CSV encontrado na pasta files.')
