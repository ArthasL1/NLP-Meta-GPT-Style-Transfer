import pandas as pd
import os

data_folder = 'path/to/data_folder'  # 数据文件夹的路径
data_types = ['train', 'val', 'test']

dfs = {data_type: pd.DataFrame() for data_type in data_types}

for subdir, dirs, files in os.walk(data_folder):
    for file in files:
        for data_type in data_types:

            if file.startswith(data_type) and file.endswith('.tsv'):
                file_path = os.path.join(subdir, file)

                df = pd.read_csv(file_path, sep='\t')
                dfs[data_type] = pd.concat([dfs[data_type], df], ignore_index=True)

for data_type, df in dfs.items():
    output_path = os.path.join(data_folder, f'{data_type}_combined.csv')
    df.to_csv(output_path, index=False)
    print(f'{data_type}_combined.csv saved to {output_path}')
