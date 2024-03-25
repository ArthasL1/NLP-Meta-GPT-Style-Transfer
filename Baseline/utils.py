from torch.utils.data import Dataset
import torch

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



