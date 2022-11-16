#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import random
import math
import spacy
from spacy.tokens import Doc
from spacy import displacy
from scipy.sparse import coo_matrix
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
import os
os.environ['PWD'] = '/home/root-demo1/code/kaggle_nlp_demo/code'

df_train = pd.read_csv("./input/feedback-prize-english-language-learning/train.csv", nrows=1000)
df_test = pd.read_csv("./input/feedback-prize-english-language-learning/test.csv", nrows=1000)

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx  # '<pad>': 0
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx  # '<unk>': 1
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower().replace("\n", "").strip()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower().replace("\n", "").strip()
        words = text.split()
        sequence = []
        for w in words:
            word_id = self.word2idx.get(w, -1) # get id from word dic
            if word_id == -1:
                word_id = self.word2idx['<unk>'] # unknownidx
            sequence.append(word_id)
        # unknownidx = 1
        # sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return np.array([sequence], dtype=np.int32)


tokenizer = Tokenizer()
for text in df_train.full_text.to_list():
    tokenizer.fit_on_text(text)
for text in df_test.full_text.to_list():
    tokenizer.fit_on_text(text)


tokenizer.text_to_sequence("<pad> I love df <pad>")


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.array(vec, dtype=np.float32)
    return word_vec


def build_embedding_matrix(word2idx, embed_dim=300):
    embedding_matrix = np.zeros((len(word2idx), embed_dim))
    embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))

    fname = './input/glove840b300dtxt/glove.840B.300d.txt'
    word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)

    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            embedding_matrix[i] = vec

    return embedding_matrix




embedding_matrix = build_embedding_matrix(tokenizer.word2idx, 300)


# ## dependency graph

class WhitespaceTokenizer(object):  # 自定义分词器为空格分词器
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()  # 空格符号分割，最简单的一个分词器
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm') # 需要额外下载tar gz
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)




def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        for child in token.children:  # 无向对称图
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix


idx2graph = {}
for i in tqdm(range(df_train.shape[0])):
    text = df_train.full_text[i].lower().replace("\n", "").strip()
    adj_matrix = dependency_adj_matrix(text)
    coo = coo_matrix(adj_matrix)
    idx2graph[i] = np.array([coo.row, coo.col], dtype=np.int32)


test_idx2graph = {}
for i in tqdm(range(df_test.shape[0])):
    text = df_test.full_text[i].lower().replace("\n", "").strip()
    adj_matrix = dependency_adj_matrix(text)
    coo = coo_matrix(adj_matrix)
    test_idx2graph[i] = np.array([coo.row, coo.col], dtype=np.int32)




idx2graph[0]


idx2graph[0].shape


text = df_train.full_text[0].lower().replace("\n", "").strip()
doc = nlp(text)
print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Head', 'Children'))
print ("-" * 70)
for token in doc:
  print ("{:<15} | {:<8} | {:<15} | {:<20}"
         .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))


displacy.render(doc, style='dep', jupyter=True)



class MyTrainDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return "/kaggle/input/feedback-prize-english-language-learning"

    @property
    def processed_dir(self):
        return os.path.join(self.root, "train_processed")

    @property
    def raw_file_names(self):
        return ['train.csv']

    @property
    def processed_file_names(self):
        return ['train-data.pt']

    def download(self):
        pass

    def process(self):

        data_list = self.read_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def read_data(self):
        df_train = pd.read_csv(self.raw_paths[0])
        all_data = []
        for i in range(df_train.shape[0]):
            text = df_train.full_text[i].lower().replace("\n", "").strip()
            input_ids = tokenizer.text_to_sequence(text)
            label = df_train.loc[i, ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].to_list()

            x = torch.tensor(input_ids.reshape(-1, 1), dtype=torch.int32)
            edge_index  = torch.tensor(idx2graph[i], dtype=torch.long)
            y = torch.tensor(np.array(label), dtype=torch.float32).reshape(-1, 6)
            data = Data(x=x, edge_index=edge_index, y=y)

            all_data.append(data)
        return all_data



dataset = MyTrainDataset(root='/kaggle/working/')



print(dataset.raw_paths, dataset.processed_paths)


len(dataset)


train_loader = DataLoader(dataset, batch_size=128, shuffle=True)



class FeedbackModel(nn.Module):
    def __init__(self, embedding_matrix):
        
        super(FeedbackModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.gc1 = GCNConv(300, 128)
        self.gc2 = GCNConv(128, 64)
        self.out = nn.Linear(64, 6)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.squeeze(1)
        x = self.embed(x)
        x = F.relu(self.gc1(x, edge_index))
        x = F.relu(self.gc2(x, edge_index))
        x = pyg_nn.global_mean_pool(x, batch)
        output = F.relu(self.out(x))
        return output


# In[32]:


model = FeedbackModel(embedding_matrix)
model


# ## Train

# In[33]:


epochs = 60

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion.to(device)


# In[34]:


model.train()
for epoch_num in range(epochs):
    total_loss_train = 0
    for sample_batched in train_loader:
        sample_batched = sample_batched.to(device)
        optimizer.zero_grad()
        outputs = model(sample_batched)
        label = sample_batched.y.to(device)
        loss = criterion(outputs, label)
        loss.backward()
        total_loss_train += loss.item()
        optimizer.step()
    print(f'Epoch: %02.0f | Train Loss: {total_loss_train / len(dataset): .3f}' % (epoch_num + 1))


# ## TestDataset Class

# In[35]:


class MyTestDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return "/kaggle/input/feedback-prize-english-language-learning"

    @property
    def processed_dir(self):
        return os.path.join(self.root, "test_processed")
    
    @property
    def raw_file_names(self):
        return ['test.csv']

    @property
    def processed_file_names(self):
        return ['test-data.pt']

    def download(self):
        pass

    def process(self):

        data_list = self.read_data()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def read_data(self):
        df_test = pd.read_csv(self.raw_paths[0])
        all_data = []
        for i in range(df_test.shape[0]):
            text = df_test.full_text[i].lower().replace("\n", "").strip()
            input_ids = tokenizer.text_to_sequence(text)

            x = torch.tensor(input_ids.reshape(-1, 1), dtype=torch.int32)
            edge_index  = torch.tensor(test_idx2graph[i], dtype=torch.long)
            data = Data(x=x, edge_index=edge_index)

            all_data.append(data)
        return all_data


# In[36]:


test_dataset = MyTestDataset(root='/kaggle/working/')


# In[37]:


print(test_dataset.raw_paths, test_dataset.processed_paths)


# In[38]:


test_dataset[0].edge_index


# In[39]:


len(test_dataset)


# In[40]:


test_loader = DataLoader(test_dataset, batch_size=128)


# ## Prediction

# In[41]:


print("make prediction!!!")


# In[42]:


prediction = []
model.eval()
with torch.no_grad():
    for sample_batched in test_loader:
        sample_batched = sample_batched.to(device)
        outputs = model(sample_batched).detach().cpu().numpy()
        prediction.append(outputs)
prediction = np.concatenate(prediction)


# ## Submission

# In[43]:


print("make submission!!!")


# In[44]:


submission = pd.DataFrame(
    data=prediction,
    index=df_test.text_id,
    columns=["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"])
submission.head()


# In[45]:


submission.to_csv("submission.csv", index = True)

