import copy
import numpy as np
import torch
import torch.nn.functional as F
import os

# print(os.getcwd())
from Buffers.replay_buffer import ReplayBuffer
from Networks.models import Critic,Actor,hard_update
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
        self.local_critic = Critic(seed,nS,nA).to(self.device)
        self.target_critic = Critic(seed,nS,nA).to(self.device)
        self.local_actor = Actor(seed,nS,nA).to(self.device)
        self.target_actor = Actor(seed,nS,nA).to(self.device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr = 1e-3,weight_decay=L2)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr = 1e-4)
        
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)

    def add(self,state,action,reward,next_state,done):
        self.R.add(state,action,reward,next_state,done)
        
    def step(self):
        # Sample memory if len > minimum
        # Get experience tuples
        samples = self.R.sample()
        # Learn from them and update local networks
        self.learn(samples)
        # Update target networks
        self.update_networks()
            
    def learn(self,samples):
        states,actions,rewards,next_states,dones = samples

        target_actions = self.target_actor(next_states)
        Q_targets = self.target_critic(next_states,target_actions)
        target_y = rewards + (self.gamma*Q_targets*(1-dones))
    
        # GAE rewards
        # GAE_rewards = torch.tensor(self.GAE(rewards.cpu().numpy()))
        # target_y = GAE_rewards + (self.gamma*Q_targets*(1-dones))

        # update critic
        self.critic_optimizer.zero_grad()
        current_y = self.local_critic(states,actions)
        critic_loss = (target_y - current_y).mean()**2
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.local_critic.parameters(),1)
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()
        local_actions = self.local_actor(states)
        actor_loss = self.local_critic(states, local_actions)
        actor_loss = -actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        # if sum(rewards) > 0:
        #     print('finally')
        
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
            action = self.local_actor(state).data.cpu().numpy()
        self.local_actor.train()
        # Act with noise
        action = np.clip(action + N,-1,1)
        return action
    
    def update_networks(self):
        self.target_critic = DDPG.soft_update_target(self.local_critic,self.target_critic,self.tau)
        self.target_actor = DDPG.soft_update_target(self.local_actor,self.target_actor,self.tau)
        
    @staticmethod
    def soft_update_target(local,target,tau):
        for local_param,target_param in zip(local.parameters(),target.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
        return target