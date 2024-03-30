import pandas as pd
import sys
import os

k_shot = int(sys.argv[1])
n_samples = int(sys.argv[2])
task_path = sys.argv[3]

folder_path = os.path.dirname(task_path)

total_samples = k_shot + n_samples

df = pd.read_csv(task_path, sep='\t')

assert len(df) >= total_samples, "samples not enough"

samples = df.sample(n=total_samples, random_state=42)

support = samples[:k_shot]
query = samples[k_shot:]

support_path = os.path.join(folder_path, 'support.tsv')
query_path = os.path.join(folder_path, 'query.tsv')

support.to_csv(support_path, sep='\t', index=False)
query.to_csv(query_path, sep='\t', index=False)
