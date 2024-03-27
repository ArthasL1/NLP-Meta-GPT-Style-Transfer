from torch.utils.data import Dataset
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
class PairDataset(Dataset):
    def __init__(self, data, tokenizer,totalpad=80):
        """
        Args:
            data (list of tuples): A list of data items, where each item is a tuple containing two parts in the form of (X, y).
            device (string): Specifies the device on which the data should be processed (e.g., "cuda" or "cpu").
            totalpad (int): The total desired length for the output list.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.totalpad = totalpad

    # Converts all text in the given pairs and tests to lowercase.
    def lowering(self, text):
        return text.lower()

    # Processes the pairs list, replacing any word starting with a digit with "NUM" in each pair of texts.
    def num_convert(self, text):
        digit_chars = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
        modified_text = ' '.join(["NUM" if word[0] in digit_chars else word for word in text.split()])
        return modified_text

    # Pad the input sequences to ensure that their lengths are consistent with the defined length
    def padinput(self, inputlist):
        pads = [0] * (self.totalpad - len(inputlist))
        input_padded = inputlist + pads
        mask = [1] * len(inputlist) + pads
        return input_padded, mask

    # Create label for training, with padding values set to -100 for ignoring in loss calculations.
    def labels(self, inlen, outputlist):
        pads1 = [-100] * inlen
        pads2 = [-100] * (self.totalpad - inlen - len(outputlist))
        return pads1 + outputlist + pads2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        X_processed = self.lowering(X)
        X_processed=self.num_convert(X_processed)
        y_processed = self.lowering(y)
        y_processed=self.num_convert(y_processed)

        X_encoded = self.tokenizer.encode(X_processed + "<|endoftext|>")
        y_encoded = self.tokenizer.encode(y_processed + "<|endoftext|>")

        X_padin = self.padinput(X_encoded)
        inputs, mask = X_padin[0], X_padin[1]
        label = self.labels(len(X_encoded), y_encoded)

        inputs = torch.tensor(inputs, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return inputs, label, mask

def read_tsv_to_list(file_path):
    """
    Reads a TSV file and converts its contents into a list.

    Parameters:
    - file_path: The path to the TSV file.

    Returns:
    - data: A list containing each row from the file, where each row is itself a list.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')  # Use tab as the delimiter
        for row in reader:
            data.append(row)
    return data


def train_n_val(train_loader,val_loader,optimizer,model,num_epoch,device,patience,model_dir):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    train_losses = []
    val_losses = []
    best_train_loss = np.inf
    best_val_loss = np.inf
    early_stop_counter = 0

    for epoch in range(0, num_epoch):
        print('\nEpoch: %d/%d' % (epoch, num_epoch))
        model.train()
        train_total = 0
        train_loss = 0

        for batch_idx, (inputs, label, masks) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_cuda:
                inputs, label, masks = inputs.to(device), label.to(device), masks.to(device)
            ret = model.forward(inputs, attention_mask=masks, labels=label)
            loss = ret[0]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_total += inputs.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        if avg_train_loss < best_train_loss:
            early_stop_counter = 0
            best_train_loss = avg_train_loss

            train_dir = os.path.join(model_dir, "train")
            os.makedirs(train_dir, exist_ok=True)

            train_path = os.path.join(train_dir, f'best_train_model_at_{epoch}_epoch_with{best_train_loss}.pt')
            torch.save(model, train_path)
            print(f"New best train loss: {best_train_loss}. Model saved at epoch {epoch} in {train_path}.")

            all_files = os.listdir(train_dir)
            model_files = [file for file in all_files if file.endswith('.pt')]


            if len(model_files) > 5:

                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(train_dir, x)))

                for file_to_delete in model_files[:-5]:
                    os.remove(os.path.join(train_dir, file_to_delete))

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

            val_dir = os.path.join(model_dir, "val")
            os.makedirs(val_dir, exist_ok=True)

            val_path = os.path.join(val_dir, f'best_val_model_at_{epoch}_epoch_with{best_val_loss}.pt')

            torch.save(model, val_path)
            print(f"New best val loss: {best_val_loss}. Model saved at epoch {epoch} in {val_path}.")

            all_files = os.listdir(val_dir)
            model_files = [file for file in all_files if file.endswith('.pt')]


            if len(model_files) > 5:
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(val_dir, x)))

                for file_to_delete in model_files[:-5]:
                    os.remove(os.path.join(val_dir, file_to_delete))

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()