import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from utils import *
import numpy as np
import sys

def print_lass_loss(opt):
	print 'loss: ',opt.losses[-1]

def print_last_validation_result(opt):
	print'validation loss: ',opt.validation_result[-1]

class ModelTrainer:
	def __init__(self,
				model,
				dataset,
				validset,
				optimizer=optim.Adam,
				lr=0.001,
				weight_decay=0,
				gradient_clipping=None,
				logger=Logger(),
				snapshot_path=None,
				snapshot_name = 'snapshot',
				snapshot_interval=1000,
				snapshot_thresh = 10000,
				dtype=torch.FloatTensor,
				ltype=torch.LongTensor):
		self.model = model
		self.dataset = dataset
		self.validset = validset        
		self.dataloader = None
		self.lr = lr
		self.weight_decay = weight_decay
		self.clip = gradient_clipping
		self.optimizer_type = optimizer
		self.optimizer = self.optimizer_type(params=self.model.parameters(), 
											 lr=self.lr, 
		 									 weight_decay=self.weight_decay)
		self.logger = logger
		self.logger.trainer = self
		self.snapshot_path = snapshot_path
		self.snapshot_name = snapshot_name
		self.snapshot_interval = snapshot_interval
		self.snapshot_thresh = snapshot_thresh
		self.dtype = dtype
		self.ltype = ltype
        	self.validloader = torch.utils.data.DataLoader(self.validset,
        											 batch_size = 16,
        											 shuffle = False,
        											 num_workers = 8,
        											 pin_memory=False)
        def train(self,
        			batch_size = 16,
        			epochs = 20,
        			continue_training_at_step = 0):
        	self.model.train()

        	self.dataloader = torch.utils.data.DataLoader(self.dataset,
        											 batch_size = batch_size,
        											 shuffle = True,
        											 num_workers = 8,
        											 pin_memory=False,
        											 drop_last = True)


        	step = continue_training_at_step
        	for current_epoch in range(epochs):
        		print 'epoch ',current_epoch
        		tic = time.time()
        		for (x,target) in iter(self.dataloader):
        			x = Variable(x.type(self.dtype))
        			target = Variable(target.view(-1,50).type(self.dtype))
        			output = self.model(x).view(-1,50)
        			loss = F.binary_cross_entropy(output,target)
        			#loss = self.model.loss(output,target,alpha=1.1)            
        			self.optimizer.zero_grad()
        			loss.backward()
        			loss = loss.data[0]

        			if self.clip is not None:
        				torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clip)
        			self.optimizer.step()
        			step+=1

        			if step == 100:
        				toc = time.time()
        				print 'one training step does take approximately' + str((toc-tic)*0.01) +' seconds'

        			if step%self.snapshot_interval==0 and step>self.snapshot_thresh:
        				if self.snapshot_path is None:
        					continue
        				time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
        				torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + str(step))

        			self.logger.log(step,loss)


        def validate(self):
        	self.model.eval()

        	self.validloader.train = False

        	total_loss = 0

        	for (x,target) in iter(self.validloader):
        		x = Variable(x.type(self.dtype)).view(-1,1,96,1366)
        		target = Variable(target.view(-1,50).type(self.dtype))

        		output = self.model(x)
        		loss = F.binary_cross_entropy(output,target)
        		#loss = self.model.loss(output,target,alpha=1.1)
        		total_loss += loss.data[0]

        	avg_loss = total_loss / len(self.validloader)
        	self.dataset.train = True
        	self.model.train()
        	return avg_loss
