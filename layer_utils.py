import torch
from torch import nn
import torch.nn.functional as F
from utils import dilate

class CausalConv1d(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=2,stride=1,
					dilation=1,bias=True):
		super(CausalConv1d,self).__init__()

		self.pad = (kernel_size - 1) * dilation
		self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,
							stride=stride,padding=self.pad,dilation=dilation,bias=bias)

	def forward(self,inputs):
		
		outputs = self.conv(inputs)
		return outputs[:,:,:-self.pad]


class ResidualBlock(nn.Module):
	def __init__(self,dilation_channels,res_channels,skip_channels,kernel_size,bias):
		super(ResidualBlock,self).__init__()
		self.filter_conv = nn.Conv1d(in_channels=res_channels,out_channels=dilation_channels,kernel_size=kernel_size,bias=bias)
		self.filter_norm = nn.BatchNorm1d(dilation_channels)        
		self.gate_conv = nn.Conv1d(in_channels = res_channels,out_channels = dilation_channels,kernel_size=kernel_size,bias=bias)
		self.gate_norm = nn.BatchNorm1d(dilation_channels)
		self.skip_conv = nn.Conv1d(in_channels = dilation_channels,out_channels = skip_channels,kernel_size=1,bias=bias)
		self.skip_norm = nn.BatchNorm1d(skip_channels)        
		self.res_conv = nn.Conv1d(in_channels = dilation_channels,out_channels = res_channels,kernel_size=1,bias=bias)
		self.res_norm = nn.BatchNorm1d(res_channels)       
		self.kernel_size = kernel_size
	def forward(self,inputs,dilation,init_dilation):
		inputs = dilate(inputs,dilation,init_dilation)
		sigmoid_out = self.gate_conv(inputs)
		sigmoid_out = self.gate_norm(sigmoid_out)
		sigmoid_out = F.sigmoid(sigmoid_out)        
		tanh_out = self.filter_conv(inputs)
		tanh_out = self.filter_norm(tanh_out)
		tanh_out = F.tanh(tanh_out)      
		hidden = sigmoid_out*tanh_out
		skip = hidden
		if hidden.size(2)!=1:
			skip = dilate(hidden,1,init_dilation=dilation)
		skip_out = self.skip_conv(skip)
		skip_out = self.skip_norm(skip_out)        
		res_out = self.res_conv(hidden)
		res_out = self.res_norm(res_out)
		outputs = res_out+inputs[:,:,(self.kernel_size-1):]
		return outputs,skip_out

class SELayer2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        bottleneck = int(channel / reduction)
        self.conv = nn.Sequential(
                nn.Conv2d(channel, bottleneck,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck, channel,1),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y).view(b, c, 1, 1)
        return x * y

class SELayer1d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        bottleneck = int(channel / reduction)
        self.conv = nn.Sequential(
                nn.Conv1d(channel, bottleneck,1),
                nn.ReLU(inplace=True),
                nn.Conv1d(bottleneck, channel,1),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).contiguous()
        y = self.conv(y).contiguous().view(b, c, 1)
        return x * y