# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import datetime
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import model as Model


class Solver(object):
	def __init__(self, data_loader, config):
		# data loader
		self.data_loader = data_loader
		self.dataset = config.dataset
		self.data_path = config.data_path
		self.input_length = config.input_length

		# training settings
		self.n_epochs = config.n_epochs
		self.lr = config.lr
		self.use_tensorboard = config.use_tensorboard

		# model path and step size
		self.model_save_path = config.model_save_path
		self.model_load_path = config.model_load_path
		self.log_step = config.log_step
		self.batch_size = config.batch_size
		self.model_type = config.model_type

		# cuda
		self.is_cuda = torch.cuda.is_available()

		# Build model
		self.get_dataset()
		self.build_model()

		# Tensorboard
		self.writer = SummaryWriter()

	def get_dataset(self):
		if self.dataset == 'mtat':
			self.valid_list = np.load('./../split/mtat/valid.npy')
			self.binary = np.load('./../split/mtat/binary.npy')

	def get_model(self):
		if self.model_type == 'fcn':
			return Model.FCN()
		elif self.model_type == 'musicnn':
			return Model.Musicnn(dataset=self.dataset)
		elif self.model_type == 'crnn':
			return Model.CRNN()
		elif self.model_type == 'sample':
			return Model.SampleCNN()
		elif self.model_type == 'se':
			return Model.SampleCNNSE()
		elif self.model_type == 'vgg':
			return Model.VggishCNN()
		elif self.model_type == 'attention':
			return Model.CNNSA()

	def build_model(self):
		# model
		self.model = self.get_model()

		# cuda
		if self.is_cuda:
			self.model.cuda()

		# load pretrained model
		if len(self.model_load_path) > 1:
			self.load(self.model_load_path)

		# optimizers
		self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

	def load(self, filename):
		S = torch.load(filename)
		self.model.load_state_dict(S)

	def to_var(self, x):
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x)

	def get_loss_function(self):
		return nn.BCELoss()

	def train(self):
		# Start training
		start_t = time.time()
		current_optimizer = 'adam'
		reconst_loss = self.get_loss_function()
		best_metric = 0
		drop_counter = 0

		# Iterate
		for epoch in range(self.n_epochs):
			ctr = 0
			drop_counter += 1
			self.model.train()
			for x, y in self.data_loader:
				ctr += 1
				# Forward
				x = self.to_var(x)
				y = self.to_var(y)
				out = self.model(x)

				# Backward
				loss = reconst_loss(out, y)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# Log
				self.print_log(epoch, ctr, loss, start_t)
			self.writer.add_scalar('Loss/train', loss.item(), epoch)

			# validation
			best_metric = self.validation(best_metric, epoch)

			# schedule optimizer
			current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

		print("[%s] Train finished. Elapsed: %s"
				% (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
					datetime.timedelta(seconds=time.time() - start_t)))

	def opt_schedule(self, current_optimizer, drop_counter):
		# adam to sgd
		if current_optimizer == 'adam' and drop_counter == 80:
			self.load(os.path.join(self.model_save_path, 'best_model.pth'))
			self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001, 
											momentum=0.9, weight_decay=0.0001, 
											nesterov=True)
			current_optimizer = 'sgd_1'
			drop_counter = 0
			print('sgd 1e-3')
		# first drop
		if current_optimizer == 'sgd_1' and drop_counter == 20:
			self.load(os.path.join(self.model_save_path, 'best_model.pth'))
			for pg in self.optimizer.param_groups:
				pg['lr'] = 0.0001
			current_optimizer = 'sgd_2'
			drop_counter = 0
			print('sgd 1e-4')
		# second drop
		if current_optimizer == 'sgd_2' and drop_counter == 20:
			self.load(os.path.join(self.model_save_path, 'best_model.pth'))
			for pg in self.optimizer.param_groups:
				pg['lr'] = 0.00001
			current_optimizer = 'sgd_3'
			print('sgd 1e-5')
		return current_optimizer, drop_counter

	def save(self, filename):
		model = self.model.state_dict()
		torch.save({'model': model}, filename)

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
		print('roc_auc: %.4f' % roc_aucs)
		print('pr_auc: %.4f' % pr_aucs)
		return roc_aucs, pr_aucs

	def print_log(self, epoch, ctr, loss, start_t):
		if (ctr) % self.log_step == 0:
			print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
					(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
						epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
						datetime.timedelta(seconds=time.time()-start_t)))

	def validation(self, best_metric, epoch):
		roc_auc, pr_auc, loss = self.get_validation_score(epoch)
		score = 1 - loss
		if score > best_metric:
			print('best model!')
			best_metric = score
			torch.save(self.model.state_dict(), 
					   os.path.join(self.model_save_path, 'best_model.pth'))
		return best_metric


	def get_validation_score(self, epoch):
		self.model.eval()
		est_array = []
		gt_array = []
		losses = []
		reconst_loss = self.get_loss_function()
		index = 0
		for line in tqdm.tqdm(self.valid_list):
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
			index += 1

		est_array, gt_array = np.array(est_array), np.array(gt_array)
		loss = np.mean(losses)
		print('loss: %.4f' % loss)

		roc_auc, pr_auc = self.get_auc(est_array, gt_array)
		self.writer.add_scalar('Loss/valid', loss, epoch)
		self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
		self.writer.add_scalar('AUC/PR', pr_auc, epoch)
		return roc_auc, pr_auc, loss

