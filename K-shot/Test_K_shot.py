import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from Baseline import utils
from nltk.translate.bleu_score import sentence_bleu
import transformers


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
    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    bleu_scores.append(bleu1)
    bleu_scores.append(bleu2)
    bleu_scores.append(bleu3)
    bleu_scores.append(bleu4)

    return bleu_scores

def y_pred_text(ret, input, label, gpt_tokenizer):
    logits = ret.logits
    pred_ids = torch.argmax(logits, dim=-1)
    # add 50256 to the pred_ids first index
    rest_of_elements = pred_ids[:, :-1]
    last_element = pred_ids[:, -1:]
    # 将最后一个元素拼接到剩余元素的前面
    shifted_pred_ids = torch.cat((last_element, rest_of_elements), dim=1)
    pred_ids = shifted_pred_ids

    for input_id in input:
        # 将Tensor转换为列表
        input_id_list = input_id.tolist()
        # 移除填充token ID
        filtered_input_id_list = [tok_id for tok_id in input_id_list if tok_id != gpt_tokenizer.pad_token_id]
        filtered_input_id_list = [tok_id for tok_id in filtered_input_id_list if tok_id != -100]
        # 使用decode方法
        input_text = gpt_tokenizer.decode(filtered_input_id_list, skip_special_tokens=True)
        print("Input text:", input_text)
    # 过滤掉-100之后进行解码

    filtered_pred_ids = pred_ids[label != -100]
    print("filtered_pred_ids:", filtered_pred_ids)
    filtered_label_ids = label[label != -100]
    print("filtered_label_ids:", filtered_label_ids)

    pred_texts = gpt_tokenizer.decode(filtered_pred_ids, skip_special_tokens=True)
    actual_texts = gpt_tokenizer.decode(filtered_label_ids, skip_special_tokens=True)

    return actual_texts, pred_texts

def k_shot_evaluation(model, k_shot, n_samples, optim, num_steps=100):
    """
    Evaluate a model using k-shot learning.

    Args:
        model: a model that implements the k_shot_learning method
        k_shot:  examples to use for training
        n_samples:  examples to use for testing
        num_episodes: the number of episodes to run
    """

    gpt_model = torch.load(model)
    gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    '''Load data'''

    # load k-shot data
    # k_shot = read_data(k_shot)
    # k_input =...
    # k_mask =...
    # k_label =...
    # K = len(k_input)

    # load n-samples data
    # n_samples = read_data(n_samples)
    # n_input =...
    # n_mask =...
    # n_label =...
    # N = len(n_input)

    ''' K-shot learning, Train model'''

    # test losses

    # for each step
    # test (1st test is the zero-shot learning)
    # test bleu score
    # test loss

    # train
    # train loss
    # optimize

    # plot losses

    trainset = read_tsv_to_list(k_shot)
    testset = read_tsv_to_list(n_samples)
    train_set = PairDataset(trainset, tokenizer)
    test_set = PairDataset(testset, tokenizer)
    train_loader = DataLoader(train_set, batch_size=len(trainset), shuffle=True, num_workers=16)
    test_loader = DataLoader(valid_set, batch_size=len(testset), shuffle=True, num_workers=16)

    test_losses = []
    bleu_scores = []
    for i in range(num_steps):
        test_loss = 0
        # test, first test is the zero-shot learning
        with torch.no_grad():
            for N, (n_inputs, n_label, n_masks) in enumerate(test_loader):
                ret = model.forward(n_input, attention_mask=n_mask, labels=n_label)
                test_loss = ret[0]
                test_loss += test_loss.item()

                # get actual text and predicted text
                y_text, pred_text = y_pred_text(ret, n_inputs, n_label, gpt_tokenizer)

                # test bleu score
                bleu_score = sentence_bleu(y_text, pred_text)
                bleu_scores.append(bleu_score)

            avg_test_loss = test_loss / len(n_samples)
            test_losses.append(avg_test_loss)

        # train, train K examples
        for K, (k_inputs, k_label, k_masks) in enumerate(train_loader):
            optimizer.zero_grad()
            ret = model.forward(k_input, attention_mask=k_mask, labels=k_label)
            train_loss = ret[0]
            train_loss.backward()
            optimizer.step()

    # plot losses
    plt.plot(test_losses)
    plt.show()

    # test bleu score

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

if __name__ == '__main__':
    k_shot_evaluation(model, k_shot, n_samples, optim, num_steps=10)


