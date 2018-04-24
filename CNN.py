import torch
from torch import nn
import torch.nn.functional as F

class parallel_CNN(nn.Module):
	def __init__(self,in_channels=1,out_channels=64,bias=False,linear=False):
		super(parallel_CNN,self).__init__()
		self.main = nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
									nn.BatchNorm2d(64),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Dropout(p=0.3),
									nn.Conv2d(64,128,3,padding=1),
									nn.BatchNorm2d(128),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Dropout(p=0.3),
									nn.Conv2d(128,128,3,padding=1),
									nn.BatchNorm2d(128),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Dropout(p=0.3),
									nn.Conv2d(128,128,3,padding=1),
									nn.BatchNorm2d(128),
									nn.ELU(),
									nn.MaxPool2d(kernel_size=(3,5)),
									nn.Dropout(p=0.4)
									)

		self.conv_end = nn.Sequential(nn.Conv2d(128,out_channels,3,padding=1),
										nn.BatchNorm2d(out_channels),
										nn.ELU(),
										nn.MaxPool2d(kernel_size=(4,4)))
		self.fc = nn.Linear(1024,64)
		self.linear = linear
	def forward(self,inputs):
		if inputs.size(2)!=96:
			inputs.transpose(2,3)
		x = self.main(inputs)
		if self.linear:
			x = x.view(-1,1024)
			x = F.elu(self.fc(x))
		else:
			x = self.conv_end(x)
		return x

