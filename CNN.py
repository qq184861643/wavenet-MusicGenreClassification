import torch
from torch import nn
import torch.nn.functional as F
from layer_utils import SELayer2d

class parallel_CNN(nn.Module):
	def __init__(self,in_channels=1,out_channels=64,bias=False):
		super(parallel_CNN,self).__init__()
		self.out_channels = out_channels        
		self.main = nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
									nn.BatchNorm2d(64),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Conv2d(64,128,3,padding=1),
									nn.BatchNorm2d(128),
									#SELayer2d(128),                                  
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Conv2d(128,128,3,padding=1),
									nn.BatchNorm2d(128),
									#SELayer2d(128,8),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Conv2d(128,128,3,padding=1),
									nn.BatchNorm2d(128),
									#SELayer2d(128,8),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(3,5)),
									)

		self.conv_end = nn.Sequential(nn.Conv2d(128,out_channels,3,padding=1),
										nn.BatchNorm2d(out_channels)
										)
		#max_pool is better than avg_pool here        
		self.maxpool=nn.MaxPool2d(kernel_size=(4,4))      
	def forward(self,inputs):
		if inputs.size(2)!=96:
			inputs.transpose(2,3)
		x = self.main(inputs)
		x = self.conv_end(x)
		x = self.maxpool(x).contiguous().view(-1,self.out_channels)    
		return x

