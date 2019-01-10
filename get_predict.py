import torch
import torch.utils.data
import torch.nn as nn
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import time, datetime
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from gensim.models.word2vec import Word2Vec
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
#from lstm import SentimentLstm
import csv
import os
import pickle
import json
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys
import importlib
import numpy
from numpy import linalg
importlib.reload(sys)
current_path = os.path.dirname(__file__)

class SelfAttention(nn.Module):
	def __init__(self, hidden_dim):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.projection = nn.Sequential(
			nn.Linear(hidden_dim, 64),
			nn.ReLU(True),
			nn.Linear(64, 1)
		)

	def forward(self, encoder_outputs):
		# batch_size = encoder_outputs.size(0)
		# (B, L, H) -> (B , L, 1)
		energy = self.projection(encoder_outputs)
		weights = nn.functional.softmax(energy.squeeze(-1), dim=1)
		# (B, L, H) * (B, L, 1) -> (B, H)
		outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
		return outputs, weights


class SentimentLstm(torch.nn.Module):
	def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
				 bidirectional, weight, labels):
		super(SentimentLstm, self).__init__()
		self.num_hiddens = num_hiddens
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.embedding = torch.nn.Embedding(vocab_size, embed_size)
		self.embedding.weight = torch.nn.Parameter(torch.FloatTensor(weight))
		self.embedding.weight.requires_grad = False

		self.lstm = torch.nn.LSTM(input_size=embed_size,
								  hidden_size=self.num_hiddens,
								  num_layers=self.num_layers,
								  bidirectional=self.bidirectional,
								  dropout=0.0)
		self.attention = SelfAttention(self.num_hiddens)
		self.fc = nn.Linear(self.num_hiddens, labels)

	def forward(self, inputs):
		batch_size = inputs.size(0)
		lengths = inputs.size(1)
		embeddings = self.embedding(inputs).permute([1, 0, 2])
		# embeddings = length*batch_size*300
		out, hidden = self.lstm(embeddings)
		out = out[:, :, :self.num_hiddens] + out[:, :, self.num_hiddens:]
		# print('out',out.shape)
		# (L, B, H)
		embedding, attn_weights = self.attention(out.transpose(0, 1))
		# (B, HOP, H)
		outputs = self.fc(embedding.view(batch_size, -1))
		outputs = torch.nn.functional.softmax(outputs)
		# (B, 1)

		return outputs




# review 是一条评论
# wv就是model.wv, model = pickle.load(open('.pickle', 'rb'))
# vocab_size 词嵌入矩阵词的个数
# max_len RNN的长度
def review2index(review, wv, vocab_size, max_len):
	index = []
	for sentence in review:
		for word in sentence:
			if word in wv.index2word:  # wv.index2word是一个词列表，顺序是按照词嵌入矩阵的顺序来的
				index.append(wv.index2word.index(word))
			else:  # for <unk>
				index.append(vocab_size + 1)

	if len(index) > max_len:
		index = index[:max_len]
	else:
		while (len(index) < max_len):
			index.append(vocab_size + 1)  # 补<unk>

	index_ = np.array(index, dtype=np.int32)
	#print(index_)
	return index_

def predict(vector):
	#m = SentimentLstm()
	m = torch.load(current_path + '\\data\\model.pth')
	vector = vector.reshape((1,len(vector)))
	vector = torch.tensor(vector)
	test_feature = vector.cuda().long()
	print(m(test_feature))
	score = int(torch.max(m(test_feature), 1)[1])
	print('检测到分数为：',score+1)
	return score+1

def pretreat(x):
	words = jieba.lcut(x)
	#print(words)
	f = open(current_path + '\\data\\embedding_matrix3.pickle', 'rb')
	model = pickle.load(f)
	vocab_size = len(model.wv.vocab.keys())
	#print(vocab_size)
	vector = review2index(words, model.wv, vocab_size, 50)
	return predict(vector)


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def get_sentence_vec(text, size, model):
	vec = np.zeros(size).reshape((1, size))
	count = 0.
	for word in text:
		# print(word)
		try:
			vec += model.wv[word].reshape((1, size))
			count += 1.
		except KeyError or TypeError:
			continue

	if count != 0:
		vec /= count
	# print(vec)
	return vec

def find_friend(x):
	print(x)
	size=300
	words = jieba.lcut(x)
	f = open(current_path + '\\data\\embedding_matrix3.pickle', 'rb')
	model = pickle.load(f)
	vec1 = get_sentence_vec(words, size, model)
	reviews = np.load(current_path + "\\data\\task_reviews.npy")
	friends_list = []
	for review in reviews:
		vec2 = get_sentence_vec(review, size, model)
		#print(vec2.T)
		num = float(vec1.dot(vec2.T))  # 若为行向量则 A * B.T
		denom = np.linalg.norm(vec1) * linalg.norm(vec2)
		cos = num / denom  # 余弦值
		dist = 0.5 + 0.5 * cos  # 归一化
		#print(dist)
		if dist > 0.85:
			print(dist)
			print(review)
			friends_list.append(review)
		#print(len(friends_list))
		if len(friends_list) == 17:
			break
	return friends_list


if __name__=="__main__":


	x = "可能我太老了，已经完全失去了孩子的纯真，我惶恐地发现我已经无法欣赏这样的电影。" \
		"因为我要看的是一群假装自己是孩子的大人过家家（里面没有一个真正的大人），这让我觉得有些脸红。而" \
		"像《变形金刚》这样的系列电影，最终也成功地将电影变成了一种比肥皂剧还无聊的东西。电影并不是不可以幻想，" \
		"不是不可以肤浅，但是在幻想和肤浅的前提下，为什么还无法呈现一些独属于电影这种形式才能收容的超越想象的东西？" \
		"这实在是太让人难以接受了。既然都有“高等外星机械生命”“外星球大战”这样听上去就很炫酷的设定和背景了，为什么还要" \
		"让我们看一个小女孩与她的外星宠物的故事并且假装自己乐在其中？片中小女孩查莉和宠物大黄蜂都在拼命对抗全世界，" \
		"但我感受不到他们的“挣扎”，一切都是儿戏，所以我也不会在他们打败坏人后拍手叫好，我宁愿看他们俩谈恋爱"
	pretreat(x)
	#find_friend(x)
