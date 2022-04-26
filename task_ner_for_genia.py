from transformers import BertConfig, BertTokenizer, BertModel
from torch.utils.data import IterableDataset, DataLoader
import torch
import numpy as np
from torch import nn
from torchcrf import CRF
import config
from torch.optim import Adam
from tqdm import tqdm
import conlleval

bert_model_path = r'bert_base_uncased'

bert_model = BertModel.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

max_vocab_size = 1000
batch_size = 16
maxlen = 64
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_file = 'i2b2/train.txt'
test_file = 'i2b2/test.txt'

words = {}
labels = {}
all_ori_tokens = []
ori_tokens = []
ori_labels = []
with open(test_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line == ' \n':
            if all([x == 'O' for x in ori_labels]):
                ori_labels = []
                ori_tokens = []
                continue
            all_ori_tokens.append(['[CLS]']+ori_tokens+['[SEP]'])
            ori_tokens = []
            ori_labels = []
            continue
        line = line.strip().split()
        word = line[0].lower()
        label = line[1]
        ori_tokens.append(word)
        ori_labels.append(label)
        words[word] = words.get(word, 0) + 1
        labels[label] = labels.get(label, 0) + 1

words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
char2id = {j:i+1 for i,(j,_) in enumerate(words)}
id2char = {i+1:j for i,(j,_) in enumerate(words)}
labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
label2id = {j:i for i,(j,_) in enumerate(labels)}
id2label = {i:j for i,(j,_) in enumerate(labels)}
num_labels = len(label2id)

def padding(x):
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]


class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        X, Y, x, y = [], [], [], []
        masks = []
        with open(self.file_path, 'r', encoding='utf-8') as file_obj:
            for line in file_obj:
                if line == ' \n':
                    if len(x) > maxlen-2:
                        x = x[:maxlen-2]
                        y = [0] + y[:maxlen-2] + [0]
                    else:
                        y = [0] + y + [0]
                    if not any(y):
                        x, y = [], []
                        continue
                    x = bert_tokenizer.encode_plus(x)
                    input_ids = x['input_ids']
                    attention_mask = x['attention_mask']
                    X.append(input_ids)
                    Y.append(y)
                    masks.append(attention_mask)
                    x, y = [], []
                    continue
                line_data = line.strip().split()
                x.append(line_data[0].lower())
                y.append(label2id.get(line_data[1], 0))
                if len(X) == batch_size:
                    yield np.array(padding(X)), np.array(padding(Y)), np.array(padding(masks))
                    X, Y, masks = [], [], []



dataset = MyIterableDataset(train_file)
train_dataloader = DataLoader(dataset)
dataset = MyIterableDataset(test_file)
test_dataloader = DataLoader(dataset)


class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self):
        super(BERT_BiLSTM_CRF, self).__init__()

        self.num_tags = num_labels
        self.bert = bert_model
        self.lstm = nn.LSTM(config.hidden_size, config.lstm_dim, num_layers=1, bidirectional=True, batch_first=True)

        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden2tag = nn.Linear(config.lstm_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())

        return loss

    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):

        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        sequence_output, (_, _) = self.lstm(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())


def evaluate(dataloader, model, id2label, all_ori_tokens):
    model.eval()

    print("***** Running eval *****")
    pred_labels = []
    ori_labels = []

    for b_i, (input_ids, label_ids, input_mask) in enumerate(tqdm(dataloader, desc="Evaluating")):

        input_ids = input_ids[0].to(device).long()
        input_mask = input_mask[0].to(device).long()
        label_ids = label_ids[0].to(device).long()

        with torch.no_grad():
            logits = model.predict(input_ids, input_mask=input_mask)

        for l in logits:
            pred_labels.append([id2label[idx] for idx in l])

        for l in label_ids:
            ori_labels.append([id2label[idx.item()] for idx in l])

    eval_list = []
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")

    # eval the model
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)

    overall, by_type = conlleval.metrics(counts)

    return overall, by_type


model = BERT_BiLSTM_CRF().to(device)
optimizer = Adam(model.parameters(), lr=0.00003)

best_f1 = 0
for epoch in range(config.epochs):
    model.train()
    total_loss = []
    pbar = tqdm(train_dataloader, desc='train')
    for x, y, masks in pbar:
        x = x.long().to(device)
        y = y.long().to(device)
        masks = masks.long().to(device)
        loss = model(x[0], y[0], input_mask=masks[0])
        loss.backward()
        optimizer.step()
        model.zero_grad()
        total_loss.append(loss.item())
        pbar.set_description(f'total loss {np.mean(total_loss)}')
    overall, by_type = evaluate(test_dataloader, model, id2label, all_ori_tokens)
    if overall.fscore > best_f1:
        best_f1 = overall.fscore
        torch.save(model.state_dict(), 'model.pt')

        