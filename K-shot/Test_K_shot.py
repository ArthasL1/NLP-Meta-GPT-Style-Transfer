import matplotlib.pyplot as plt
import torch
from Baseline import utils
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
import transformers
import torch.optim as optim
import os

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

def y_pred_text1(ret, input, label, gpt_tokenizer):
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
    filtered_label_ids = label[label != -100]


    pred_texts = gpt_tokenizer.decode(filtered_pred_ids, skip_special_tokens=True)
    actual_texts = gpt_tokenizer.decode(filtered_label_ids, skip_special_tokens=True)

    print("Y label: ",actual_texts)
    print("Predict: ", pred_texts)
    return actual_texts, pred_texts

def y_pred_text(ret, inputs, labels, gpt_tokenizer):
    logits = ret.logits
    pred_ids = torch.argmax(logits, dim=-1)
    scores = []

    for batch_index in range(pred_ids.size(0)):
        pred_id = pred_ids[batch_index]
        last_element = pred_id[-1:]
        rest_of_elements = pred_id[:-1]
        adjusted_pred_id = torch.cat((last_element, rest_of_elements), dim=0)

        input_id = inputs[batch_index]
        label_id = labels[batch_index]

        filtered_input_id = [tok_id for tok_id in input_id.tolist() if
                             tok_id != gpt_tokenizer.pad_token_id and tok_id != -100]
        input_text = gpt_tokenizer.decode(filtered_input_id, skip_special_tokens=True)
        print("Input text:", input_text)

        pred_text = gpt_tokenizer.decode(adjusted_pred_id[label_id != -100], skip_special_tokens=True)
        actual_text = gpt_tokenizer.decode(label_id[label_id != -100], skip_special_tokens=True)

        scores.append(calculate_bleu_score(actual_text, pred_text))

        print("Actual text:", actual_text)
        print("Predicted text:", pred_text)
        print("---------------")

    # calculate the average bleu score
    avg_bleu_scores = [0, 0, 0, 0]
    for score in scores:
        for i in range(4):
            avg_bleu_scores[i] += score[i]
    for i in range(4):
        avg_bleu_scores[i] /= len(scores)
    
    return avg_bleu_scores



def k_shot_evaluation(model, k_suppot_path, n_query_path,loss_dir,support_batch=None,query_batch=None,num_steps=10):
    """
    Evaluate a model using k-shot learning.

    Args:
        model: a model that implements the k_shot_learning method
        k_shot_path:  path of support set
        n_samples_path:  path of query set
        loss_dir:path of folder for storing query losses
        support_batch: batch size of support
        query_batch:batch size of query
        num_steps: the number of steps to run
    """
    global avg_support_loss
    model = torch.load(model)
    gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')

    support_set = utils.read_tsv_to_list(k_suppot_path)
    query_set = utils.read_tsv_to_list(n_query_path)
    support_set = utils.PairDataset(support_set, gpt_tokenizer)
    query_set = utils.PairDataset(query_set, gpt_tokenizer)

    if support_batch==None:
        support_batchsize=len(support_set)
    else:
        support_batchsize=support_batch
    if query_batch==None:
        query_batchsize=len(query_set)
    else:
        query_batchsize=query_batch

    support_loader = DataLoader(support_set, batch_size=support_batchsize, shuffle=True, num_workers=16)
    query_loader = DataLoader(query_set, batch_size=query_batchsize, shuffle=True, num_workers=16)
    optimizer = optim.SGD(model.parameters(), lr=2e-4)
    support_losses=[]
    query_losses = []
    bleu_scores = []
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_steps):
        query_loss = 0
        # test, first test is the zero-shot learning
        with torch.no_grad():
            for N, (n_inputs, n_label, n_masks) in enumerate(query_loader):
                if use_cuda:
                    n_inputs, n_label, n_masks = n_inputs.to(device), n_label.to(device), n_masks.to(device)
                ret = model.forward(n_inputs, attention_mask=n_masks, labels=n_label)
                loss = ret[0]
                query_loss += loss.item()
                del loss

                # get actual text and predicted text

                y_text, pred_text =y_pred_text(ret,n_inputs,n_label,gpt_tokenizer)

                # test bleu score
                bleu_score = sentence_bleu(y_text, pred_text)

                bleu_score =y_pred_text(ret, n_inputs, n_label, gpt_tokenizer)
                del ret

                bleu_scores.append(bleu_score)

            avg_query_loss = query_loss / len(support_loader)
            query_losses.append(avg_query_loss)
            print('Step: %d| Query loss: %.3f| BlEU1: %.3f| BlEU2: %.3f| BlEU3: %.3f| BlEU4: %.3f'%(
                i,avg_query_loss,bleu_score[0],bleu_score[1],bleu_score[2],bleu_score[3]
            ))

        # support, train K examples
        support_loss = 0
        for K, (k_inputs, k_label, k_masks) in enumerate(support_loader):
            if use_cuda:
                k_inputs, k_label, k_masks = k_inputs.to(device), k_label.to(device), k_masks.to(device)
            optimizer.zero_grad()
            ret = model.forward(k_inputs, attention_mask=k_masks, labels=k_label)
            loss = ret[0]
            support_loss += loss.item()
            

            avg_support_loss = support_loss / len(support_loader)
            loss.backward()
            optimizer.step()
            
            del ret
            del loss
        print('Step: %d| Support loss: %.3f' % (
            i+1, avg_support_loss
        ))
        support_losses.append(avg_support_loss)

        if i==num_steps-1:
            query_loss = 0 
            with torch.no_grad():
                for N, (n_inputs, n_label, n_masks) in enumerate(query_loader):
                    if use_cuda:
                        n_inputs, n_label, n_masks = n_inputs.to(device), n_label.to(device), n_masks.to(device)
                    ret = model.forward(n_inputs, attention_mask=n_masks, labels=n_label)
                    query_loss += ret[0].item()

                    # get actual text and predicted text
                    y_text, pred_text = y_pred_text(ret, n_inputs, n_label, gpt_tokenizer)
                    del ret
                    # test bleu score
                    bleu_score = calculate_bleu_score(y_text, pred_text)
                    bleu_scores.append(bleu_score)

                avg_query_loss = query_loss / len(query_loader)
                query_losses.append(avg_query_loss)
                print('Step: %d| Test loss: %.3f| BlEU1: %.3f| BlEU2: %.3f| BlEU3: %.3f| BlEU4: %.3f' % (
                    i+1, avg_query_loss, bleu_score[0], bleu_score[1], bleu_score[2], bleu_score[3]
                ))


    # plot losses
    step_support=range(1, len(support_losses) + 1)
    step_query=range(len(query_losses))

    plt.plot(step_support,support_losses, label='Support Loss')
    plt.plot(step_query,query_losses, label='Query Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Support and Query Losses')
    plt.show()


    file_path = os.path.join(loss_dir, "query_losses.txt")
    with open(file_path, "w") as file:
        for loss in query_losses:
            file.write(f"{loss}\n")


if __name__ == '__main__':

    loss_dir = 'query_losses'
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    k_shot_evaluation('test_model.pt',
                      './splitedtestset1/part1.tsv',
                      './splitedtestset1/part2.tsv',
                      loss_dir,
                      None,
                      None,
                      10
                      )
