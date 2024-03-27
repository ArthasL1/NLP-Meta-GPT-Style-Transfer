import transformers
import torch
from torch.utils.data import DataLoader
from utils import *
import os
import multiprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.00002,weight_decay=0.015)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
num_epoch=50
early_stop_counter = 0
patience = 100
model_dir = 'baseline_best_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

train_path=''
val_path=''
trainset=read_tsv_to_list(train_path)
valset=read_tsv_to_list(val_path)
train_set=PairDataset(trainset,gpt_tokenizer)
valid_set=PairDataset(valset,gpt_tokenizer)
train_loader=DataLoader(train_set, batch_size=40, shuffle=True, num_workers=2)
val_loader=DataLoader(valid_set, batch_size=10, shuffle=True, num_workers=2)

if __name__ == '__main__':
    # 创建进程池或者进程，并开始执行你的函数
    multiprocessing.freeze_support()  # 在 Windows 上可能需要这行
    train_n_val(train_loader,val_loader,optimizer,model,num_epoch,device,patience,model_dir)