# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
import datetime
import tqdm
import fire
import argparse
from sklearn import metrics
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

import model as Model


class Predict(object):
	def __init__(self, config):
		self.model_type = config.model_type
		self.model_load_path = config.model_load_path
		self.dataset = config.dataset
		self.data_path = config.data_path
		self.batch_size = config.batch_size
		self.is_cuda = torch.cuda.is_available()
		self.build_model()
		self.get_dataset()

	def get_model(self):
		if self.model_type == 'fcn':
			self.input_length = 29 * 16000
			return Model.FCN()
		elif self.model_type == 'musicnn':
			self.input_length = 3 * 16000
			return Model.Musicnn(dataset=self.dataset)
		elif self.model_type == 'crnn':
			self.input_length = 29 * 16000
			return Model.CRNN()
		elif self.model_type == 'sample':
			self.input_length = 59049
			return Model.SampleCNN()
		elif self.model_type == 'se':
			self.input_length = 59049
			return Model.SampleCNNSE()
		elif self.model_type == 'vgg':
			self.input_length = 59049
			return Model.VggishCNN()
		elif self.model_type == 'attention':
			self.input_length = 15 * 16000
			return Model.CNNSA()
		else:
			print('model_type has to be one of [fcn, musicnn, crnn, sample, se, vgg, attention]')

	def build_model(self):
		self.model = self.get_model()

		# cuda
		if self.is_cuda:
			self.model.cuda()

		# load model
		self.load(self.model_load_path)

	def get_dataset(self):
		if self.dataset == 'mtat':
			self.test_list = np.load('./../split/mtat/test.npy')
			self.binary = np.load('./../split/mtat/binary.npy')

	def load(self, filename):
		S = torch.load(filename)
		if 'spec.mel_scale.fb' in S.keys():
			S['spec.mel_scale.fb'] = torch.tensor([])
		self.model.load_state_dict(S)

	def to_var(self, x):
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x)

	def get_tensor(self, fn):
		# load audio
		if self.dataset == 'mtat':
			npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
		raw = np.load(npy_path, mmap_mode='r')

		# split chunk
		length = len(raw)
		hop = (length - self.input_length) // self.batch_size
		x = torch.zeros(self.batch_size, self.input_length)
		for i in range(self.batch_size):
			x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
		return x

	def get_auc(self, est_array, gt_array):
		roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
		pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
		return roc_aucs, pr_aucs

	def test(self):
		roc_auc, pr_auc, loss = self.get_test_score()
		print('loss: %.4f' % loss)
		print('roc_auc: %.4f' % roc_aucs)
		print('pr_auc: %.4f' % pr_aucs)

	def get_test_score(self):
		self.model.eval()
		est_array = []
		gt_array = []
		losses = []
		reconst_loss = nn.BCELoss()
		for line in tqdm.tqdm(self.test_list):
			if self.dataset == 'mtat':
				ix, fn = line.split('\t')

			# load and split
			x = self.get_tensor(fn)

			# ground truth
			if self.dataset == 'mtat':
				ground_truth = self.binary[int(ix)]

			# forward
			x = self.to_var(x)
			y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
			out = self.model(x)
			loss = reconst_loss(out, y)
			losses.append(float(loss.data))
			out = out.detach().cpu()

			# estimate
			estimated = np.array(out).mean(axis=0)
			est_array.append(estimated)
			gt_array.append(ground_truth)

		est_array, gt_array = np.array(est_array), np.array(gt_array)
		loss = np.mean(losses)

		roc_auc, pr_auc = self.get_auc(est_array, gt_array)
		return roc_auc, pr_auc, loss


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'mtg-jamendo'])
	parser.add_argument('--model_type', type=str, default='fcn',
						choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'vgg', 'attention'])
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--model_load_path', type=str, default='.')
	parser.add_argument('--data_path', type=str, default='./data')

	config = parser.parse_args()

	p = Predict(config)
	p.test()






