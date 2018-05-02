import torch
import torch.nn as nn
from torch.autograd import Variable
from CNN import parallel_CNN
from wavenet import WaveNet
import torch.nn.functional as F 
from layer_utils import SELayer1d

class ParallelModel(nn.Module):
	def __init__(self):
		super(ParallelModel,self).__init__()
		
		self.cnn = parallel_CNN(out_channels = 128)
		self.wavenet = CascadeModel(L_trans_channels=512)
		self.drop = nn.Dropout(p=0.3)
		self.fc1 = nn.Linear(256,512)
		self.fc2 = nn.Linear(512,256) 
		self.fc3 = nn.Linear(256,50)        

	def forward(self,inputs):
		out1 = self.cnn(inputs).contiguous().view(-1,128)
		#print out1.size()
		out2 = self.wavenet.dilate_conv(inputs).contiguous().view(-1,128)
		#print out2.size()
		hidden = torch.cat((out1,out2),dim=1)
		hidden = F.relu(self.fc1(hidden))#relu is to be tested
		hidden = self.drop(hidden)
		hidden = F.relu(self.fc2(hidden))
		hidden = self.drop(hidden)        
		outputs = F.sigmoid(self.fc3(hidden))        
		return outputs

	def loss(self,outputs,target,alpha = 1.2):
		#check = F.binary_cross_entropy(outputs,target)
		x = outputs
		y = target
		epsilon = 1e-5
		one_to_zero = -torch.log(x+epsilon)*y
		zero_to_one = -torch.log(1-x+epsilon)*(1-y)
		loss = torch.mean(torch.sum(one_to_zero*alpha+zero_to_one,dim=1))
		return loss


class CascadeModel(nn.Module):
    def __init__(self,input_C=96,input_L=1366,L_trans_channels=256):
        super(CascadeModel,self).__init__()
        self.input_C = input_C
        self.input_L = input_L
        self.first_block = nn.Sequential(nn.Conv1d(input_L,L_trans_channels,1),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,3),
					 nn.BatchNorm1d(L_trans_channels),
					 nn.ReLU(),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,3),
                                         nn.BatchNorm1d(L_trans_channels),
                                         nn.ELU(),
                                         nn.MaxPool1d(2),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,3),
					 nn.BatchNorm1d(L_trans_channels),
					 nn.ReLU(),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,3),
                                         nn.BatchNorm1d(L_trans_channels),
                                         nn.ELU(),
                                         nn.MaxPool1d(2),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,3),
					 nn.BatchNorm1d(L_trans_channels),
					 nn.ReLU(),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,3,padding=1),
                                         nn.BatchNorm1d(L_trans_channels),
                                         nn.ELU(),
                                         nn.MaxPool1d(2),
                                         nn.Conv1d(L_trans_channels,L_trans_channels,1),
                                         nn.BatchNorm1d(L_trans_channels),
                                         nn.ELU(),
                                         nn.Conv1d(L_trans_channels,input_L,1)                                    
                                         )
        
        self.wavenet = WaveNet(in_depth = 9,
                               dilation_channels=32,
                               res_channels=32,
                               skip_channels=256,
                               end_channels = 128,
                               dilation_depth = 6,
                               n_blocks = 5)

        self.post = nn.Sequential(nn.Dropout(p=0.2),
                                  nn.Linear(128,256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(256,50),
                                  nn.Sigmoid(), 
                                  )

    def dilate_conv(self,inputs):
        inputs = inputs.contiguous().view(-1,self.input_C,self.input_L)
        assert inputs.size(1) == self.input_C
        inputs = inputs.transpose(1,2)
        hidden = self.first_block(inputs)
        hidden = hidden.contiguous().transpose(1,2)
        assert hidden.size(2) == self.input_L
        outputs = self.wavenet(hidden)[:,:,-1]
        return outputs

    def forward(self,inputs):
        wave_out = self.dilate_conv(inputs)
        outputs = self.post(wave_out)
        return outputs


