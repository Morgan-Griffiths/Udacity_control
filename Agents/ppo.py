import numpy as np
import copy
import math
import torch
from collections import namedtuple

from OUnoise import initialize_N
from models import Policy
from unity_env import UnityEnv
import torch.optim as optim

class PPO(object):
    def __init__(self,env,nA,nS,seed,SGD_epoch=3,n_trajectories=4,n_episodes=100,tmax = 700,discount = 0.995, epsilon=0.1, beta=0.01,gamma=1.0,n_step=0.95):
        self.seed = seed
        self.env = env
        self.nA = nA
        self.nS = nS
        self.n_trajectories = n_trajectories
        self.n_episodes = n_episodes
        self.tmax = tmax
        self.discount = discount
        self.start_epsilon = self.epsilon = epsilon
        self.start_beta = self.beta = beta
        self.gamma = gamma
        self.n_step = n_step
        self.trajectory = namedtuple('trajectory', field_names=('state','action','log_prob','reward','next_state','done'))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(seed,nS,nA)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-2)

    def reset_hyperparams(self):
        self.discount = self.start_discount
        self.epsilon = self.start_epsilon
        self.beta = self.start_beta

    def step_hyperparams(self):
        self.epsilon *= 0.999
        self.beta *= 0.995
    
    def step(self,trajectories):
        L = -clipped_surrogate(trajectories)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        self.step_hyperparams()

    def collect_trajectories(self):
        trajectories = []
        # for _ in range(self.n_trajectories):
        state = self.env.reset()
        for t in range(self.tmax):
            action, log_prob = self.policy(state)
            next_state,reward,done,_ = self.env.step(action)
            trajectory = self.trajectory(state,action,reward,log_prob,next_state,done)
            trajectories.append(trajectory)
            state = next_state
            if done:
                break
        return trajectories


    def clipped_surrogate(self,trajectories):
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        old_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)

        # discount and take future rewards
        GAE_rewards = np.sum([sum(rewards[:i+1])*((1-self.n_step)*self.n_step**i) for i in range(rewards.shape[0])])

        # discounts = self.discount**np.arange(len(rewards))
        # future_r = [rewards[i:]*discounts[:-i] if i>0 else rewards*discounts for i in range(len(rewards))]
        # rewards_future = [sum(future_r[i]) for i in range(len(future_r))]
        # mean = np.mean(rewards_future)
        # std = np.std(rewards_future) + 1.0e-10

        # rewards_normalized = (rewards_future - mean)/std
        # convert states to policy (or probability)
        new_probs = policy(states)
        # slice both according to the actions taken
        index = np.arange(new_probs.shape[0])
        new_probs = new_probs[index,np.array(actions)]
        old_probs = old_probs[index,np.array(actions)]
        # convert everything into pytorch tensors and move to gpu if available
        # old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        # actions = torch.tensor(actions, dtype=torch.int8, device=device)
        # rewards = torch.tensor(rewards_future, dtype=torch.float, device=device)
        ratio = new_probs/old_probs

        clip = torch.clamp(ratio,1-epsilon,1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
        
        # entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        #     (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
            
        return torch.mean(clipped_surrogate)# + entropy*beta)

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