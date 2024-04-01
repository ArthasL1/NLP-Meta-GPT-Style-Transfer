from torch.utils.data import Dataset
import transformers
import torch.optim as optim
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu

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

def calculate_bleu_score(reference, candidate):
    """
    Calculate the BLEU score between a candidate and a reference.

    Args:
        reference: a list of strings
        candidate: a list of strings

    Returns:
        The BLEU score
    """
    bleu_scores = []
    reference = [reference.split()]
    candidate = candidate.split()

    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    bleu_scores.append(bleu1)
    bleu_scores.append(bleu2)
    bleu_scores.append(bleu3)
    bleu_scores.append(bleu4)

    return bleu_scores

def y_pred_text_n_bleu_score(ret, inputs, labels, gpt_tokenizer):
    logits = ret.logits
    pred_ids = torch.argmax(logits, dim=-1)
    scores = []

    for batch_index in range(pred_ids.size(0)):
        pred_id = pred_ids[batch_index]
        last_element = pred_id[-1:]
        rest_of_elements = pred_id[:-1]
        adjusted_pred_id = torch.cat((last_element, rest_of_elements), dim=0)

        label_id = labels[batch_index]

        pred_text = gpt_tokenizer.decode(adjusted_pred_id[label_id != -100], skip_special_tokens=True)
        actual_text = gpt_tokenizer.decode(label_id[label_id != -100], skip_special_tokens=True)

        scores.append(calculate_bleu_score(actual_text, pred_text))

    # calculate the average bleu score
    avg_bleu_scores = [0, 0, 0, 0]
    for score in scores:
        for i in range(4):
            avg_bleu_scores[i] += score[i]
    for i in range(4):
        avg_bleu_scores[i] /= len(scores)

    return avg_bleu_scores

def train_n_val(train_path, val_path, optimizer_key, model_key, tokenizer_key, batch_size, num_epoch, patience, model_dir,):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    bleu_scores=[]
    best_train_loss = np.inf
    best_val_loss = np.inf
    best_val_bleu4=np.inf
    early_stop_counter = 0

    #drop out
    config = transformers.GPT2Config.from_pretrained('gpt2')
    config.attention_probs_dropout_prob = 0.1
    config.hidden_dropout_prob = 0.1

    models = {
        "GPT-2": transformers.GPT2LMHeadModel.from_pretrained('gpt2', config=config).to(device),
    }
    model = models[model_key]

    optimizers = {
        "SGD": optim.SGD(model.parameters(), lr=2e-4),
        "Adam": optim.Adam(model.parameters(), lr=2e-4),
        "AdamW": optim.AdamW(model.parameters(), lr=2e-4),
        "RMSprop": optim.RMSprop(model.parameters(), lr=2e-4),
        "Adagrad": optim.Adagrad(model.parameters(), lr=2e-4)
    }
    optimizer = optimizers[optimizer_key]

    tokenizers={
        "GPT":transformers.GPT2Tokenizer.from_pretrained('gpt2')
    }
    tokenizer=tokenizers[tokenizer_key]

    trainset = read_tsv_to_list(train_path)
    valset = read_tsv_to_list(val_path)
    train_set = PairDataset(trainset, tokenizer)
    valid_set = PairDataset(valset, tokenizer)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(valid_set, batch_size=10, shuffle=True, num_workers=16)


    for epoch in range(0, num_epoch):
        print('\nEpoch: %d/%d' % (epoch, num_epoch))
        model.train()
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

        avg_train_loss = train_loss / len(train_loader)
        print('Epoch: %d| Training loss: %.3f' % (
            epoch, avg_train_loss
        ))
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
        total_bleu_score=[0,0,0,0]
        bleu_score4=0
        with torch.no_grad():
            for batch_idx, (val_inputs, val_label, val_masks) in enumerate(val_loader):
                if use_cuda:
                    val_inputs, val_label, val_masks = val_inputs.to(device), val_label.to(device), val_masks.to(device)
                ret = model.forward(val_inputs, attention_mask=val_masks, labels=val_label)
                loss = ret[0]
                val_loss += loss.item()
                bleu_score = y_pred_text_n_bleu_score(ret, val_inputs, val_label, tokenizer)
                total_bleu_score = [total_bleu_score[i] + bleu_score[i] for i in range(len(total_bleu_score))]
                bleu_score4+=bleu_score[3]

                del loss
                del ret

        avg_val_loss = val_loss / len(val_loader)
        avg_val_bleu4=bleu_score4/len(val_loader)
        val_losses.append(avg_val_loss)
        average_bleu_scores = [score / len(val_loader) for score in total_bleu_score]
        bleu_scores.append(average_bleu_scores)

        print('Epoch: %d| Val loss: %.3f| BlEU1: %.3f| BlEU2: %.3f| BlEU3: %.3f| BlEU4: %.3f' % (
            epoch, avg_val_loss, bleu_score[0], bleu_score[1], bleu_score[2], bleu_score[3]
        ))

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

        if avg_val_bleu4 < best_val_bleu4:
            best_val_bleu4 = avg_val_bleu4

            bleu_dir = os.path.join(model_dir, "bleu")
            os.makedirs(bleu_dir, exist_ok=True)

            bleu_path = os.path.join(bleu_dir, f'best_bleu4_model_at_{epoch}_epoch_with{best_val_bleu4}.pt')

            torch.save(model, bleu_path)
            print(f"New best val bleu4: {best_val_bleu4}. Model saved at epoch {epoch} in {bleu_path}.")

            all_files = os.listdir(bleu_dir)
            model_files = [file for file in all_files if file.endswith('.pt')]


            if len(model_files) > 5:
                model_files.sort(key=lambda x: os.path.getmtime(os.path.join(bleu_dir, x)))

                for file_to_delete in model_files[:-5]:
                    os.remove(os.path.join(bleu_dir, file_to_delete))

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    file_path = os.path.join(train_dir, "train_losses.txt")
    with open(file_path, "w") as file:
        for loss in train_losses:
            file.write(f"{loss}\n")

    file_path = os.path.join(val_dir, "val_losses.txt")
    with open(file_path, "w") as file:
        for loss in val_losses:
            file.write(f"{loss}\n")

    file_path = os.path.join(bleu_dir, "bleu_scores.txt")
    with open(file_path, "w") as file:
        for bleu_score in bleu_scores:
            file.write(f"{bleu_score}\n")


    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.show()
    plt.figure(figsize=(20, 10))

    # subplot1：train_losses and val_losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.grid(True)

    # subplot2：BLEU_score
    plt.subplot(1, 2, 2)
    for i in range(4):
        plt.plot(range(1, bleu_scores.shape[0] + 1), bleu_scores[:, i], label=f'BLEU-{i + 1}', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.title('BLEU Scores')
    plt.grid(True)

    plt.tight_layout()
    plt.show()