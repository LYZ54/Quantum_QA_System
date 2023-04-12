import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from QLSTM import QLSTM
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

# QLSTM的参数
vocab_dim = 3  # 输入的词向量的维度
embedding_dim = 3  # 处理的词向量的维度,可随意赋值
hidden_dim = 5  # h_t 的维度，可随意赋值
label_size = 2  # label 的种类
dev_type = 'default.qubit'  # device
n_qubits = 0  # 量子数

seq_len = 10  # 一句话中最多有seq_len个单词
n_epoch = 40  # 迭代次数
batch_size = 40  # 每次送入网络中的句子
f = open('./dataset/vocab_dim3.pkl', 'rb')  # 训练好的词典
index_dic = pickle.load(f)
word_vectors = pickle.load(f)

n_symbols = len(index_dic) + 1  # 总共的词的数量
embedding_weights = np.zeros((n_symbols, vocab_dim))  # 创建一个 n_symbols * 3 的零矩阵

for word, index in index_dic.items():
    embedding_weights[index, :] = word_vectors[word]  # 填充embedding_weights 每一行是一个词向量


# 将文本数据映射为一个编号矩阵（每个编号对应一个词向量）
def sentence2index_array(index_dic, sents):
    if type(sents) is list:
        new_sentences = []
        for sent in sents:
            new_sen = []
            words = sent.split(' ')
            for word in words:
                try:
                    new_sen.append(index_dic[word])
                except:
                    new_sen.append(0)
            new_sentences.append(new_sen)
        return np.array(new_sentences)
    else:
        new_sentence = []
        new_sen = []
        words = sents.split(' ')
        for word in words:
            try:
                new_sen.append(index_dic[word])
            except:
                new_sen.append(0)
        new_sentence.append(new_sen)
        return new_sentence


# 将数据切割成一样的指定长度
def text_cut_to_same_long(sents):
    data_num = len(sents)
    new_sents = np.zeros((data_num, seq_len))  # 构建一个矩阵来装修剪好的数据
    se = []
    for i in range(len(sents)):
        new_sents[i, :] = sents[i, :seq_len]
    new_sents = np.array(new_sents)
    return new_sents


# 将每个句子的序号矩阵替换成词向量矩阵

def create_wordvec_tensor(embedding_weights, X_T):
    X_tt = np.zeros((len(X_T), seq_len, vocab_dim))
    num1 = 0
    num2 = 0
    for sent in X_T:
        for word in sent:
            X_tt[num1, num2, :] = embedding_weights[int(word), :]
            num2 = num2 + 1
        num1 = num1 = 1
        num2 = 0
    return X_tt


# 由句子所组成的列表转换成（len(sents), seq_len, vec_dim) 的三维数组
def sentence2vector(sentences):
    sentences = sentence2index_array(index_dic, sentences)
    sentences = pad_sequence([torch.from_numpy(np.array(x)) for x in sentences], batch_first=True).float()
    sentences = text_cut_to_same_long(sentences)
    sentences = create_wordvec_tensor(embedding_weights, sentences)
    return sentences


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, labelset_size, n_qubits=0, dev_type='default.qubit'):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            print(f"Tagger will use QLSTM running on backend {dev_type}")
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, dev_type=dev_type)
        else:
            print("Tagger will use Classical LSTM")
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2label = nn.Linear(hidden_dim, labelset_size)

    def forward(self, sentence):
        print(sentence.shape)
        embeds = self.word_embeddings(sentence)
        print(embeds.shape)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        label_logit = self.hidden2label(lstm_out.view(len(sentence, -1)))
        label_scores = F.log_softmax(label_logit, dim=1)
        return label_scores


path1 = "./dataset/Text_Emotion.csv"
path2 = "./dataset/test.csv"
data = pd.read_csv(path1, encoding='utf-8')
sentences_list = data['text']
sentences_list = sentences_list.to_list()
labels = data['emotion']
label_list = []
for label in labels:
    if label == "☹️":
        label_list.append(0)
    else:
        label_list.append(1)

# 划分训练集和测试集， 得到 训练集文本， 测试集文本， 训练标签， 测试标签
X_train, X_test, y_train, y_test = train_test_split(sentences_list, label_list, test_size=0.2)
# 将句子转换为索引值
X_train = sentence2index_array(index_dic, X_train)
X_test = sentence2index_array(index_dic, X_test)

# 将数据补长为最大长度
X_train = pad_sequence([torch.from_numpy(np.array(x)) for x in X_train], batch_first=True).float()
X_test = pad_sequence([torch.from_numpy(np.array(x)) for x in X_test], batch_first=True).float()

# 切割数据长度到 seq_len
X_train = text_cut_to_same_long(X_train)
X_test = text_cut_to_same_long(X_test)

# 将索引转换为词向量
X_train = create_wordvec_tensor(embedding_weights, X_train)
X_test = create_wordvec_tensor(embedding_weights, X_test)

print("训练集shape： ", X_train.shape)
print("测试集shape： ", X_test.shape)

# 创建Tensor Dataset
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(np.array(y_train)))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(np.array(y_test)))

# 创建DataLoader 和 batch
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
# 定义模型（输入参数）
model = LSTMTagger(embedding_dim, hidden_dim, vocab_size=vocab_dim, labelset_size=label_size, n_qubits=n_qubits,
                   dev_type=dev_type)

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

####训练####
print('----------进行训练集训练-----------------')
for epoch in range(n_epoch):
    correct = 0
    total = 0
    epoch_loss = 0
    batch_num = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_num = batch_idx + 1
        data = torch.as_tensor(data, dtype=torch.long)
        target = target.long()
        optimizer.zero_grad()
        output, (h_t, c_t) = model(data)

        correct += int(torch.sum(torch.argmax(output, dim=1) == target))
        total += len(target)

        # 梯度清零， 反向传播
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    loss = epoch_loss / batch_num
    print(f'epoch:{epoch}, accuracy:{correct / total}, loss:{loss}')

####测试####
print('----------进行测试集验证-----------------')
correct = 0
total = 0
epoch_loss = 0
batch_num = 0
model.eval()
for batch_idx, (data, target) in enumerate(train_loader):
    batch_num = batch_idx + 1
    data = torch.as_tensor(data, dtype=torch.float32)
    target = target.long()
    optimizer.zero_grad()
    output, (h_t, c_t) = model(data)

    correct += int(torch.sum(torch.argmax(output, dim=1) == target))
    total += len(target)

    # 梯度清零， 反向传播
    optimizer.zero_grad()
    loss = F.cross_entropy(output, target)
    epoch_loss += loss.item()
    loss.backward()
    optimizer.step()

loss = epoch_loss / batch_num
print(f'accuracy:{correct / total}, loss:{loss}')