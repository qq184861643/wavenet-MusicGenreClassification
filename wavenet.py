import torch
from torch import nn
import torch.nn.functional as F
from layer_utils import ResidualBlock,CausalConv1d
from utils import *

class WaveNet(nn.Module):
	def __init__(self,in_depth=96,
					dilation_channels=32,
					res_channels=32,
					skip_channels=256,
					end_channels=64,
					kernel_size=2,
					bias=False,
					dilation_depth=7,n_blocks=4):
		super(WaveNet,self).__init__()
		self.n_blocks = n_blocks
		self.dilation_depth = dilation_depth

		self.pre_conv = nn.Conv1d(in_depth,res_channels,kernel_size,bias=bias)

		self.dilations = []
		self.resblocks = nn.ModuleList()
		init_dilation=1
		receptive_field = 2
		for i in range(n_blocks):
			addition_scope = kernel_size-1
			new_dilation = 1
			for i in range(dilation_depth):
				self.dilations.append((new_dilation,init_dilation))
				self.resblocks.append(ResidualBlock(dilation_channels,res_channels,
														skip_channels,kernel_size,bias))
				receptive_field+=addition_scope
				addition_scope*=2
				init_dilation = new_dilation
				new_dilation*=2


		self.post = nn.Sequential(nn.ELU(),
								  nn.Conv1d(skip_channels,skip_channels,1,bias=True),
								  nn.ELU(),
								  nn.Conv1d(skip_channels,end_channels,1,bias=True))
		self.receptive_field = receptive_field
	def forward(self,inputs):
		x = self.pre_conv(inputs)
		#print x.size()
		skip = 0

		for i in range(self.n_blocks*self.dilation_depth):
			(dilation,init_dilation) = self.dilations[i]
			x,s = self.resblocks[i](x,dilation,init_dilation)
			try:
				skip = skip[:,:,-s.size(2):]
			except:
				skip = 0
			#if not isinstance(skip,int):
				#print 'skip',skip.size(),'s',s.size()
			skip = skip+s

		outputs = self.post(skip)

		return outputs