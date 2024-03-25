import transformers
import torch
from torch.utils.data import DataLoader
from utils import *
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2').to(device)
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.00002,weight_decay=0.015)
num_epoch=50
best_train_loss = np.inf
best_val_loss = np.inf
train_losses = []
val_losses = []
early_stop_counter = 0
patience = 5

trainset=[]
valset=[]

train_set=PairDataset(trainset,gpt_tokenizer)
valid_set=PairDataset(valset,gpt_tokenizer)
train_loader=DataLoader(train_set, batch_size=20, shuffle=True, num_workers=2)
val_loader=DataLoader(valid_set, batch_size=2, shuffle=True, num_workers=2)

use_cuda = torch.cuda.is_available()
print(use_cuda)

for epoch in range(0, num_epoch):
    print('\nEpoch: %d/%d' % (epoch, num_epoch))
    model.train()
    train_total = 0
    train_loss = 0

    for batch_idx, (inputs, label, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        ret = model.forward(inputs, attention_mask=masks, labels=label)
        loss = ret[0]
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_total += inputs.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        torch.save(model.state_dict(), 'best_train_model.pt')  # 保存最佳训练模型
        print(f"New best train loss: {best_train_loss}. Model saved at epoch {epoch}.")
    else:
        early_stop_counter += 1

    model.eval()
    val_loss = 0
    val_total = 0
    with torch.no_grad():
        for batch_idx, (val_inputs, val_label, val_masks) in enumerate(val_loader):
            if use_cuda:
                val_inputs, val_label, val_masks = val_inputs.to(device), val_label.to(device), val_masks.to(device)
            ret = model.forward(val_inputs, attention_mask=val_masks, labels=val_label)
            loss = ret[0]
            val_loss += loss.item()
            val_total += val_inputs.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_val_model.pt')  # 保存最佳验证模型
        print(f"New best val loss: {best_val_loss}. Model saved at epoch {epoch}.")

    if early_stop_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()




