import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self,seed,nS,nA,device,hidden_dims=(200,100)):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        self.device = device
        
        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0]+nA,hidden_dims[1])
        self.fc1_bn = nn.BatchNorm1d(hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[1],1)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state,action):
        xs = self.input_bn(F.relu(self.input_layer(state)))
        x = torch.cat((xs,action),dim=1)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        return self.output_layer(x)

class Actor(nn.Module):
    def __init__(self,seed,nS,nA,device,hidden_dims=(400,300)):
        super(Actor,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        self.device = device

        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0],hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[1],nA)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.output_layer.weight.data.uniform_(-3e-3,3e-3)
        
    def forward(self,state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.fc1(x))
        return torch.tanh(self.output_layer(x))

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)