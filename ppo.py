import numpy as np
import copy
import math

from models import Policy
from unity_env import UnityEnv
import torch.optim as optim

class PPO(object):
    def __init__(self,env,nA,nS,batch_size,seed,tmax = 700,discount = 0.995, epsilon=0.1, beta=0.01,gamma=1.0,n_step=0.95):
        self.seed = seed
        self.env = env
        self.nA = nA
        self.nS = nS
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_step = n_step
        self.discount = discount
        self.start_epsilon = self.epsilon = epsilon
        self.start_beta = self.beta = beta

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(seed,nS,nA)
        self.optimizer = optim.Adam(policy.parameters(), lr = 1e-2)

    # convert states to probability, passing through the policy
    def states_to_prob(policy, states,actions):
        policy_input = torch.tensor(states,dtype=torch.float32)
        return policy(policy_input)

    def collect_trajectories(self,env,policy):
        rewards = []
        dones = []
        states = []
        actions = []
        a_probs = []
    #     state,env = initialize_env(env)
        state = env.reset()
        for t in range(self.tmax):
            probs = policy(state).cpu().detach().numpy()[0]
            action = np.random.choice([0,1],p=probs)
            state,reward,done,_ = env.step(action)

            actions.append(action)
            a_probs.append(probs)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            if done:
                break
        return actions,np.vstack(a_probs),states,rewards,dones

    def reset_hyperparams(self):
        self.discount = self.start_discount
        self.epsilon = self.start_epsilon
        self.beta = self.start_beta

    def step_hyperparams(self):
        self.epsilon *= 0.999
        self.beta *= 0.995
    
    def step(a_probs,states,actions,rewards):
        L = -clipped_surrogate(old_probs, states, actions, rewards)
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        self.step_hyperparams()


def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      ):
    # discount and take future rewards
    discounts = self.discount**np.arange(len(rewards))
    future_r = [rewards[i:]*discounts[:-i] if i>0 else rewards*discounts for i in range(len(rewards))]
    rewards_future = [sum(future_r[i]) for i in range(len(future_r))]
    mean = np.mean(rewards_future)
    std = np.std(rewards_future) + 1.0e-10

    rewards_normalized = (rewards_future - mean)/std
    # convert states to policy (or probability)
    new_probs = states_to_prob(policy,states,actions)
    # slice both according to the actions taken
    index = np.arange(new_probs.shape[0])
    new_probs = new_probs[index,np.array(actions)]
    old_probs = old_probs[index,np.array(actions)]
    # convert everything into pytorch tensors and move to gpu if available
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    rewards = torch.tensor(rewards_future, dtype=torch.float, device=device)
    ratio = new_probs/old_probs

    clip = torch.clamp(ratio,1-epsilon,1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
    
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
        
    return torch.mean(clipped_surrogate + entropy*beta)