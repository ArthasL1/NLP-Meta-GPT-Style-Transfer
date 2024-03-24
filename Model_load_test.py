import transformers
import torch
import csv

def lowering(pairs, tests):
    for pair in pairs:
        for i in range(0, 2):
            pair[i] = pair[i].lower()
    for pair in tests:
        for i in range(0, 2):
            pair[i] = pair[i].lower()

def numpreprocess(pairs, tests):
    for pair in pairs + tests:
        for i in range(0, 2):
            rep = []
            for word in pair[i].split(' '):
                if len(word) > 0 and word[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    rep.append("NUM")
                else:
                    rep.append(word)
            pair[i] = ' '.join(rep)
                 

def valid(src, trg):
    
    padin = [padinput(l) for l in src] 
    print('padin:', padin)
    padedin = torch.LongTensor([padin[i][0] for i in range(0,len(trg))]).to(device)
    masks = torch.LongTensor([padin[i][1] for i in range(0,len(trg))]).to(device)
    label = torch.LongTensor([labels(len(src[i]),trg[i]) for i in range(0,len(trg))]).to(device)
    print('label:', label)
    print('label length:', len(label[0]))
    
    
    with torch.no_grad():
        ret = gpt_model.forward(padedin, attention_mask=masks, labels=label)
        loss = ret[0]
        
        logits = ret.logits
        pred_ids = torch.argmax(logits, dim=-1)
        # add 50256 to the pred_ids first index
        rest_of_elements = pred_ids[:, :-1]
        last_element = pred_ids[:, -1:]
        # 将最后一个元素拼接到剩余元素的前面
        shifted_pred_ids = torch.cat((last_element, rest_of_elements), dim=1)
        pred_ids = shifted_pred_ids
        print("pred_ids length:", len(pred_ids[0]))
        print("pred_ids:", pred_ids)


        for input_id in padedin:
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

        print("Predicted texts:", pred_texts)
        print("Actual texts:", actual_texts)
        print()


    return loss

def padinput(inputlist, totalpad=80):
    pads = [0] * (totalpad - len(inputlist))
    input = inputlist + pads
    mask = [1] * len(inputlist) + pads
    return input, mask


# create label for training
def labels(inlen, outputlist, totalpad=80):
    pads1 = [-100] * inlen
    pads2 = [-100] * (totalpad - inlen - len(outputlist))
    # print(outputlist)
    return pads1 + outputlist + pads2

def batchvalid(src, trg):
    validloss = 0.0
    for i in range(0, len(src) // batchsize):
        asrc = []
        atrg = []
        for pair in src[i * batchsize:(i + 1) * batchsize]:
            asrc.append(pair)
        for pair in trg[i * batchsize:(i + 1) * batchsize]:
            atrg.append(pair)
        validloss += valid(asrc, atrg)
    return validloss / (len(src) // batchsize)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


batchsize = 1
gpt_model = torch.load('modelIter8.pt')
gpt_tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
pairs = []
tests = []

f = open('./TFU/train.tsv', 'r')
ff = csv.reader(f, delimiter='\t')
limit = 10
cur = 0
for row in ff:
    pairs.append(row)

f = open('./TFU/valid.tsv', 'r')
ff = csv.reader(f, delimiter='\t')
for row in ff:
    tests.append(row)

lowering(pairs, tests)
# use this for single style transfer
numpreprocess(pairs, tests)

pairsEncode = []
testsEncode = []
for i in pairs:
    pairsEncode.append((gpt_tokenizer.encode(i[0] + " <|endoftext|>"), gpt_tokenizer.encode(i[1] + " <|endoftext|>")))
for i in tests:
    testsEncode.append((gpt_tokenizer.encode(i[0] + " <|endoftext|>"), gpt_tokenizer.encode(i[1] + " <|endoftext|>")))
tsrc = []
ttrg = []
for pair in testsEncode[:20]:
    tsrc.append(pair[0])
    ttrg.append(pair[1])

validloss = batchvalid(tsrc, ttrg)
print(" Valid loss: " + str(validloss))


