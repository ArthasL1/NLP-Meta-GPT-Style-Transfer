import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from Baseline import utils
from torch.utils.data import DataLoader
import torch.optim as optim
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
    rest_of_elements = pred_ids[:, :-1]
    last_element = pred_ids[:, -1:]
    shifted_pred_ids = torch.cat((last_element, rest_of_elements), dim=1)
    pred_ids = shifted_pred_ids

    for input_id in input:
        input_id_list = input_id.tolist()
        filtered_input_id_list = [tok_id for tok_id in input_id_list if tok_id != gpt_tokenizer.pad_token_id]
        filtered_input_id_list = [tok_id for tok_id in filtered_input_id_list if tok_id != -100]
        input_text = gpt_tokenizer.decode(filtered_input_id_list, skip_special_tokens=True)
        print("Input text:", input_text)

    filtered_pred_ids = pred_ids[label != -100]
    print("filtered_pred_ids:", filtered_pred_ids)
    filtered_label_ids = label[label != -100]
    print("filtered_label_ids:", filtered_label_ids)

    pred_texts = gpt_tokenizer.decode(filtered_pred_ids, skip_special_tokens=True)
    actual_texts = gpt_tokenizer.decode(filtered_label_ids, skip_special_tokens=True)

    print(actual_texts,pred_texts)
    return actual_texts, pred_texts

def k_shot_evaluation(model, k_shot, n_samples, optim, num_steps=10):
    """
    Evaluate a model using k-shot learning.

    Args:
        model: a model that implements the k_shot_learning method
        k_shot:  examples to use for training
        n_samples:  examples to use for testing
        num_episodes: the number of episodes to run
    """

    model = torch.load(model)
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

    trainset = utils.read_tsv_to_list(k_shot)
    testset = utils.read_tsv_to_list(n_samples)
    train_set = utils.PairDataset(trainset, gpt_tokenizer)
    test_set = utils.PairDataset(testset, gpt_tokenizer)
    train_loader = DataLoader(train_set, batch_size=len(trainset), shuffle=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=len(testset), shuffle=True, num_workers=16)

    test_losses = []
    bleu_scores = []
    for i in range(num_steps):
        test_loss = 0
        # test, first test is the zero-shot learning
        with torch.no_grad():
            for N, (n_inputs, n_label, n_masks) in enumerate(test_loader):
                ret = model.forward(n_inputs, attention_mask=n_masks, labels=n_label)
                test_loss = ret[0]
                test_loss += test_loss.item()

                # get actual text and predicted text
                y_text, pred_text =y_pred_text(ret, n_inputs, n_label, gpt_tokenizer)

                # test bleu score
                bleu_score = sentence_bleu(y_text, pred_text)
                bleu_scores.append(bleu_score)

            avg_test_loss = test_loss / len(n_samples)
            test_losses.append(avg_test_loss)

        # train, train K examples
        for K, (k_inputs, k_label, k_masks) in enumerate(train_loader):
            optimizer.zero_grad()
            ret = model.forward(k_inputs, attention_mask=k_masks, labels=k_label)
            train_loss = ret[0]
            train_loss.backward()
            optimizer.step()

    # plot losses
    plt.plot(test_losses)
    plt.show()

    # test bleu score

optimizer=optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if __name__ == '__main__':
    k_shot_evaluation('model',
                      'k_shot',
                      'n_samples',
                      optimizer,
                      10)