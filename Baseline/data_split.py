import pandas as pd
import os

data_folder = 'path/to/data_folder'
data_types = ['train', 'val', 'test']

dfs = {data_type: pd.DataFrame() for data_type in data_types}

for subdir, dirs, files in os.walk(data_folder):
    for file in files:
        for data_type in data_types:
            if file.startswith(data_type) and file.endswith('.tsv'):
                file_path = os.path.join(subdir, file)


                df = pd.read_csv(file_path, sep='\t', header=None)
                dfs[data_type] = pd.concat([dfs[data_type], df], ignore_index=True)


for data_type, df in dfs.items():
    output_path = os.path.join(data_folder, f'{data_type}_combined.tsv')
    df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f'{data_type}_combined.tsv saved to {output_path}')
