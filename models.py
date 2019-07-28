import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
TODO: Make a continuous space Reinforce policy, where we output a gaussian and calculate the probs based on the mu and sigma.
"""
class ReinforcePolicy(nn.Module):
    def __init__(self,seed,nS,nA,hidden_dims=(64,64)):
        super(Policy,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_dims = hidden_dims
        self.nA = nA
        self.nS = nS
        self.size = hidden_dims[-1] * hidden_dims[-2]
        self.std = nn.Parameter(torch.zeros(1, nA))
        
        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1,len(self.hidden_dims)):
            hidden_layer = nn.Linear(hidden_dims[i-1],hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],nA)
            
    def forward(self,state,action=None):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32) #device = self.device,
            x = x.unsqueeze(0)
        mean = torch.tanh(self.output_layer(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).detach()
        return action, log_prob, dist.entropy()

# PPO
class Policy(nn.Module):
    def __init__(self,seed,nS,nA,hidden_dims=(64,64)):
        super(Policy,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_dims = hidden_dims
        self.nA = nA
        self.nS = nS
        self.size = hidden_dims[-1] * hidden_dims[-2]
        self.std = nn.Parameter(torch.zeros(1, nA))
        
        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1,len(self.hidden_dims)):
            hidden_layer = nn.Linear(hidden_dims[i-1],hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1],nA)
            
    def forward(self,state,action=None):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32) #device = self.device,
            x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        mean = torch.tanh(self.output_layer(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob#, dist.entropy()
    
    # Return the action along with the probability of the action. For weighting the reward garnered by the action.
    def act(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
#         print('action',action)
        return action.item(),m.log_prob(action)

class Critic(nn.Module):
    def __init__(self,seed,nS,nA,hidden_dims=(400,300)):
        super(Critic,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA
        
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
    def __init__(self,seed,nS,nA,hidden_dims=(64,64)):
        super(Actor,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.nS = nS
        self.nA = nA

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