from unityagents import UnityEnvironment
import numpy as np

from ddpg import DDPG
from train_one import train

BUFFER_SIZE = 10000
MIN_BUFFER_SIZE = 200
BATCH_SIZE = 25
ALPHA = 0.6 # 0.7 or 0.6
START_BETA = 0.5 # from 0.5-1
END_BETA = 1
QLR = 0.001
ALR = 0.0001
EPSILON = 1
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.001
L2 = 0.01
N_STEP = 0.95
UPDATE_EVERY = 4
CLIP_NORM = 10
    
def main():
    seed = 7
    env = UnityEnvironment(file_name='/Users/morgan/Code/deep-reinforcement-learning/p2_continuous-control/Reacher.app')
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    
    agent = DDPG(action_size,state_size,BUFFER_SIZE,MIN_BUFFER_SIZE,BATCH_SIZE,seed,L2,TAU,GAMMA,N_STEP)
    scores = train(agent,env,UPDATE_EVERY)
    return scores

if __name__ == "__main__":
    scores = main()