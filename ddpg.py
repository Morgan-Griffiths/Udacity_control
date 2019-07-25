import copy
import numpy as np
import torch

from replay_buffer import ReplayBuffer
from models import Critic,Actor
import torch.optim as optim

class DDPG(object):
    def __init__(self,nA,nS,BUFFER_SIZE,min_buffer_size,batch_size,seed,L2,TAU,gamma=1.0,n_step=0.95):
        self.seed = seed
        self.nA = nA
        self.nS = nS
        self.Buffer_size = BUFFER_SIZE
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.L2 = L2
        self.tau = TAU
        self.gamma = gamma
        self.n_step = n_step
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.R = ReplayBuffer(nA,BUFFER_SIZE,batch_size,seed)
        self.local_critic = Critic(seed,nS,nA,self.device)
        self.target_critic = Critic(seed,nS,nA,self.device)
        self.local_actor = Actor(seed,nS,nA,self.device)
        self.target_actor = Actor(seed,nS,nA,self.device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr = 1e-3,weight_decay=L2)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr = 1e-4)
        
    def add(self,state,action,reward,next_state,done):
        self.R.add(state,action,reward,next_state,done)
    
    def step(self,state,action,reward,next_state,done):
        # Sample memory if len > minimum
        if len(self.R) > self.min_buffer_size:
            # Get experience tuples
            samples = self.R.sample()
            # Learn from them and update local networks
            self.learn(samples)
            # Update target networks
            self.update_networks()
            
    def learn(self,samples):
#         print('learn')
        states,actions,rewards,next_states,dones = samples
#         print('actions',actions)
        target_actions = self.target_actor(states)
#         print('target critic')
        Q_targets = self.target_critic(states,target_actions)
    
        # GAE rewards
        GAE_rewards = torch.tensor(self.GAE(rewards.numpy()))
        target_y = GAE_rewards + (self.gamma*Q_targets*(1-dones))
        
        # update critic
#         print('local critic')
        current_y = self.local_critic(states,actions)
        L = 1/self.batch_size * sum(target_y - current_y)**2
        self.critic_optimizer.zero_grad()
#         print('L',L)
        L.backward()
        self.critic_optimizer.step()
        # update actor
        local_actions = self.local_actor(states)
        J = 1/self.batch_size * sum(self.local_critic(states,local_actions))
        self.actor_optimizer.zero_grad()
#         print('J',J)
        J.backward()
        self.actor_optimizer.step()
        
    def GAE(self,rewards):
        """
        Generalized Advantage Estimate.
        N_step discounted returns
        """
        return np.sum([sum(rewards[:i+1])*((1-self.n_step)*self.n_step**i) for i in range(rewards.shape[0])])
         
        
        
    def act(self,state,N):
        
        state = torch.from_numpy(state).float().to(self.device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()
        # Act with noise
#         action = np.clip(action + N,-1,1)
        return action
    
    def update_networks(self):
        self.target_critic = DDPG.soft_update_target(self.local_critic,self.target_critic,self.tau)
        self.target_actor = DDPG.soft_update_target(self.local_actor,self.target_actor,self.tau)
        
    @staticmethod
    def soft_update_target(local,target,tau):
        for local_param,target_param in zip(local.parameters(),target.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
        return target