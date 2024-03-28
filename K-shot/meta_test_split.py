import pandas as pd
import sys
import os

k_shot = int(sys.argv[1])
n_samples = int(sys.argv[2])
task_path = sys.argv[3]

# 获取task_path所在的文件夹路径
folder_path = os.path.dirname(task_path)

# 确保总样本数满足要求
total_samples = k_shot + n_samples

df = pd.read_csv(task_path, sep='\t')

assert len(df) >= total_samples, "数据集中的样本不足以进行分组"

samples = df.sample(n=total_samples, random_state=42)

# 分成support和query组
support = samples[:k_shot]
query = samples[k_shot:]

# 构建新文件的完整路径
support_path = os.path.join(folder_path, 'support.tsv')
query_path = os.path.join(folder_path, 'query.tsv')

# 保存到新的TSV文件
support.to_csv(support_path, sep='\t', index=False)
query.to_csv(query_path, sep='\t', index=False)
