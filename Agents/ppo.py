import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

# from noise import initialize_N
from Networks.models import Policy
from unity_env import UnityEnv
import torch.optim as optim

class PPO(object):
    def __init__(self,env,nA,nS,seed,trace_decay=0.95,num_agents=20,batch_size=32,gradient_clip=10,SGD_epoch=10,tmax = 320, epsilon=0.2, beta=0.01,gamma=0.99):
        self.seed = seed
        self.env = env
        self.nA = nA
        self.nS = nS
        self.trace_decay = trace_decay
        self.num_agents = num_agents
        self.batch_size= int(batch_size * num_agents)
        self.tmax = tmax
        self.start_epsilon = self.epsilon = epsilon
        self.start_beta = self.beta = beta
        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.SGD_epoch = SGD_epoch
        self.trajectory = namedtuple('trajectory', field_names=('state','next_state','action','log_prob','value','done'))
        
        if torch.cuda.is_available():
            self.device2 = torch.device("cuda:0")
            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device('cpu')

        self.policy = Policy(self.device,seed,nS,nA).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-4)

    def reset_hyperparams(self):
        self.discount = self.start_discount
        self.epsilon = self.start_epsilon
        self.beta = self.start_beta

    def step_hyperparams(self):
        self.epsilon *= 0.999
        self.beta *= 0.995
    
    def step(self):
        states,next_states,actions,log_probs,dones,values,rewards,total_rewards = self.collect_trajectories()
        returns,advantages = self.GAE_rewards(values,dones,next_states[-1],rewards)
        # Learn on batches
        for _ in range(self.SGD_epoch):
            self.learn_good(states,next_states,actions,log_probs,dones,values,returns,advantages)
        return total_rewards

    def collect_trajectories(self):
        states,next_states,actions,log_probs,dones,values = [],[],[],[],[],[]
        rewards = []
        total_rewards = []
        state = self.env.reset()
        self.policy.eval()
        for t in range(self.tmax):
            action,log_prob,dist,value = self.policy(state)
            action,log_prob,value = action.detach().cpu().numpy(), log_prob.detach().cpu().numpy(),value.detach().cpu().squeeze(-1)
            next_state,reward,done = self.env.step(action.squeeze(0))
            # For ease of multiplication later
            inverse_dones = torch.from_numpy(np.logical_not(done).astype(int)).float()

            rewards.append(torch.from_numpy(reward).float())
            states.append(np.expand_dims(state,axis=0))
            next_states.append(next_state)
            actions.append(action)
            log_probs.append(log_prob)
            dones.append(inverse_dones)
            values.append(value)
            total_rewards.append(np.sum(reward))

            state = next_state
            if done.any():
                break
        self.policy.train()
        return states,next_states,actions,log_probs,dones,values,rewards,total_rewards

    def GAE_rewards(self,values,dones,last_state,rewards):
        ### Not vectorized ###
        # trajectories look like (state,next_state,action,log_prob,value,inverse_dones)
        # rewards = [torch.from_numpy(reward) for reward in rewards]
        length = len(values)
        returns = np.zeros((length,self.num_agents))
        advantages = torch.zeros(length,self.num_agents)
        R = self.policy(last_state)[-1].squeeze(0).detach().cpu().squeeze(-1)
        advantage = 0
        next_value = R
        index = 1
        for r,v in zip(reversed(rewards),reversed(values)):
            R = r + R * self.gamma
            TD_residue = r + next_value * dones[-index] * self.gamma - v
            advantage = TD_residue + advantage * dones[-index] * self.gamma * self.trace_decay
            next_value = v
            returns[-index,:] = R
            advantages[-index,:] = advantage
            index += 1
        return returns,advantages

    def learn_good(self,states,next_states,actions,log_probs,dones,values,returns,advantages):
        N = len(states)
        # Iterate through random batch generator
        for indicies in self.minibatch(N):
            states_b = torch.from_numpy(np.vstack([states[i] for i in indicies])).float().to(self.device)
            actions_b = torch.from_numpy(np.vstack([actions[i] for i in indicies])).float().to(self.device)
            log_probs_b = torch.from_numpy(np.vstack([log_probs[i] for i in indicies])).float().to(self.device)
            values_b = torch.from_numpy(np.vstack([values[i] for i in indicies])).float().to(self.device)
            returns_b = torch.from_numpy(np.vstack([returns[i] for i in indicies])).float().to(self.device)
            advantages_b = torch.from_numpy(np.vstack([advantages[i] for i in indicies])).unsqueeze(-1).float().to(self.device)

            _,new_log_probs,entropy,new_values = self.policy(states_b,actions_b)

            ratio = (new_log_probs - log_probs_b).exp()

            clip = torch.clamp(ratio,1-self.epsilon,1+self.epsilon)
            clipped_surrogate = torch.min(ratio*advantages_b, clip*advantages_b)

            actor_loss = -torch.mean(clipped_surrogate) - self.beta * entropy.mean()
            critic_loss = F.smooth_l1_loss(values_b,returns_b)
            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
            self.optimizer.step()
            self.step_hyperparams()

    def minibatch(self,N):
        indicies = np.arange(N)
        for _ in range(self.SGD_epoch):
            np.random.shuffle(indicies)
            yield indicies[:self.batch_size]

    # def clipped_surrogate(self,trajectories):
        
    #     states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
    #     actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
    #     old_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(self.device)
    #     rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
    #     next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)

    #     # discount and take future rewards
    #     GAE_rewards = np.sum([sum(rewards[:i+1])*((1-self.n_step)*self.n_step**i) for i in range(rewards.shape[0])])

    #     # discounts = self.discount**np.arange(len(rewards))
    #     # future_r = [rewards[i:]*discounts[:-i] if i>0 else rewards*discounts for i in range(len(rewards))]
    #     # rewards_future = [sum(future_r[i]) for i in range(len(future_r))]
    #     # mean = np.mean(rewards_future)
    #     # std = np.std(rewards_future) + 1.0e-10

    #     # rewards_normalized = (rewards_future - mean)/std
    #     # convert states to policy (or probability)
    #     new_probs = policy(states)
    #     # slice both according to the actions taken
    #     index = np.arange(new_probs.shape[0])
    #     new_probs = new_probs[index,np.array(actions)]
    #     old_probs = old_probs[index,np.array(actions)]
    #     # convert everything into pytorch tensors and move to gpu if available
    #     # old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    #     # actions = torch.tensor(actions, dtype=torch.int8, device=device)
    #     # rewards = torch.tensor(rewards_future, dtype=torch.float, device=device)
    #     ratio = new_probs/old_probs

    #     clip = torch.clamp(ratio,1-epsilon,1+epsilon)
    #     clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
        
    #     # entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
    #     #     (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
            
    #     return torch.mean(clipped_surrogate)# + entropy*beta)

    def train(self):
        index = 0
        # N = initialize_N(n_episodes)
        total_rewards = []
        for e in range(1,self.n_episodes):
            # reset the environment
            state = self.env.reset()
            # get trajectories
            trajectories = self.collect_trajectories()
            for t in range(1,self.SGD_epoch):
                # Surrogate
                self.step(trajectories)

            total_rewards.append(sum(rewards))
            print('\rEpisode {}\t Average Score: {:.2f}, Last Score: {:.2f}'.format(e, np.mean(total_rewards),total_rewards[-1]),end="")
            if total_rewards[-1] >= 30.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'ppo_checkpoint.pth')
                break
        return scores