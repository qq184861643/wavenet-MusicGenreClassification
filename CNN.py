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
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Conv2d(64,128,3,padding=1),
									nn.BatchNorm2d(128),
									#SELayer2d(128),                                  
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Conv2d(128,128,3,padding=1),
									nn.BatchNorm2d(128),
									#SELayer2d(128,8),
									nn.ReLU(),
									nn.MaxPool2d(kernel_size=(2,4)),
									nn.Conv2d(128,128,3,padding=1),
									nn.BatchNorm2d(128),
									#SELayer2d(128,8),
									nn.ReLU(),
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

class GroupConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,cardinality,stride=1,downsample=False):
        super(GroupConv2d,self).__init__()

        self.downsample = downsample
        C = in_channels/2

        self.conv_pre = nn.Conv2d(in_channels,C,1)
        self.bn_pre = nn.BatchNorm2d(C)
        self.conv = nn.Conv2d(C,C,kernel_size=3,padding=1,stride=stride,groups=cardinality)
        self.bn = nn.BatchNorm2d(C)
        self.conv_expand = nn.Conv2d(C,out_channels,1)
        self.bn_expand=nn.BatchNorm2d(out_channels)
        self.conv_down = nn.Conv2d(in_channels,out_channels,1)

    def forward(self,x):
        residual = x

        h = self.conv_pre(x)
        h = F.relu(self.bn_pre(h),inplace=True)

        h = self.conv(h)
        h = F.relu(self.bn(h),inplace=True)

        h = self.conv_expand(h)
        h = self.bn_expand(h)

        if self.downsample:
            residual = self.conv_down(x)

        #if SEnet, should insert it here before h+residual

        return h+residual
