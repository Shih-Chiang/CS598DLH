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

import gensim.downloader

max_vocab_size = 1000
batch_size = 16
maxlen = 64
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_file = 'i2b2/train-try.txt'
test_file = 'i2b2/test-try.txt'

embedding_name = 'word2vec-google-news-300'
embedding_unknow_token = 0
# embedding_name = 'glove-wiki-gigaword-200'
# embedding_unknow_token = 399998
embedding_vector = gensim.downloader.load(embedding_name)
embedding_vector_len = len(embedding_vector.key_to_index)

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
            all_ori_tokens.append([ori_tokens])
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
print(num_labels)

def padding(x):
    ml = max([len(i) for i in x])
    return [i + [0] * (ml-len(i)) for i in x]

def padding2(x):
    ml = max([len(i) for i in x])
    return [i + [[0.0] * 25] * (ml-len(i)) for i in x]

class MyIterableDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        word_array_set, token_array_set, label_array_set, mask_array_set = [], [], [], []
        word_array, token_array, label_array, mask_array = [], [], [], []
        with open(self.file_path, 'r', encoding='utf-8') as file_obj:
            for line in file_obj:
                if line == ' \n':
                    if not any(label_array):
                        continue
                    if len(word_array) > maxlen:
                        word_array = word_array[:maxlen]
                        token_array = token_array[:maxlen]
                        label_array = label_array[:maxlen]
                        mask_array = mask_array[:maxlen]
                    word_array_set.append(word_array)
                    token_array_set.append(token_array)
                    label_array_set.append(label_array)
                    mask_array_set.append(mask_array)
                    word_array, token_array, label_array, mask_array = [], [], [], []
                    continue
                line_data = line.strip().split()
                w = line_data[0].lower()
                l = label2id.get(line_data[1], 0)
                char_array = [0.0] * 25
                for i, ch in enumerate(w):
                    if i < 25:
                        char_array[i] = float(ord(ch))
                word_array.append(char_array)
                token = embedding_vector.key_to_index.get(w, embedding_unknow_token)
                token_array.append(token)
                label_array.append(l)
                mask_array.append(1)

                if len(word_array_set) == batch_size:
                    yield np.array(padding2(word_array_set)), np.array(padding(token_array_set)), np.array(padding(label_array_set)), np.array(padding(mask_array_set))
                    word_array_set, token_array_set, label_array_set, mask_array_set = [], [], [], []


dataset = MyIterableDataset(train_file)
train_dataloader = DataLoader(dataset)
dataset = MyIterableDataset(test_file)
test_dataloader = DataLoader(dataset)

class BiLSTM_CRF(nn.Module):

    def __init__(self):
        super(BiLSTM_CRF, self).__init__()

        self.num_tags = num_labels
        self.char_lstm = nn.LSTM(25, config.char_lstm_hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_vector.vectors))
        self.lstm = nn.LSTM(320, config.lstm_dim, num_layers=1, bidirectional=True, batch_first=True)

        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden2tag = nn.Linear(config.lstm_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, char_tensor, token_tensor, tags, mask):
        emissions = self.tag_outputs(char_tensor, token_tensor, mask)
        loss = -1 * self.crf(emissions, tags, mask=mask.byte())

        return loss

    def tag_outputs(self, char_tensor, token_tensor, mask):

        char_embedding, (_, _) = self.char_lstm(char_tensor)
        word_embedding = self.embedding(token_tensor)

        combined_input = torch.cat((char_embedding, word_embedding), dim=2)
        combined_input = self.dropout(combined_input)
        sequence_output, (_, _) = self.lstm(combined_input)
        emissions = self.hidden2tag(sequence_output)

        return emissions

    def predict(self, char_tensor, token_tensor, mask):
        emissions = self.tag_outputs(char_tensor, token_tensor, mask)
        return self.crf.decode(emissions, mask.byte())


def evaluate(dataloader, model, id2label, all_ori_tokens):
    model.eval()

    print("***** Running eval *****")
    eval_list = []

    for b_i, (word, token, label, mask) in enumerate(tqdm(dataloader, desc="Evaluating")):

        word = word[0].float().to(device)
        token = token[0].long().to(device)
        mask = mask[0].long().to(device)
        label = label[0].tolist()
        with torch.no_grad():
            logits = model.predict(word, token, mask)

        for idx, l in enumerate(logits):
            for jdx, pred_label in enumerate(l):
                eval_list.append(f"xxx {id2label[label[idx][jdx]]} {id2label[pred_label]}\n")
        eval_list.append("\n")
    print(len(eval_list))
    # eval the model
    counts = conlleval.evaluate(eval_list)
    conlleval.report(counts)

    overall, by_type = conlleval.metrics(counts)

    return overall, by_type

model = BiLSTM_CRF().to(device)
optimizer = Adam(model.parameters(), lr=0.003)

best_f1 = 0
for epoch in range(config.epochs):
    model.train()
    total_loss = []
    pbar = tqdm(train_dataloader, desc='train')
    for word, token, label, mask in pbar:
        word = word.float().to(device)
        token = token.long().to(device)
        label = label.long().to(device)
        mask = mask.long().to(device)
        loss = model(word[0], token[0], label[0], mask[0])
        loss.backward()
        optimizer.step()
        model.zero_grad()
        total_loss.append(loss.item())
        pbar.set_description(f'total loss {np.mean(total_loss)}')
    overall, by_type = evaluate(test_dataloader, model, id2label, all_ori_tokens)
    if overall.fscore > best_f1:
        best_f1 = overall.fscore
        torch.save(model.state_dict(), 'lstm-model.pt')

for word, token, label, mask in pbar:
    model.train()
    