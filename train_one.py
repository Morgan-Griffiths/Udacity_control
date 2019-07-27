import numpy as np
from collections import deque

from OUnoise import initialize_N
from replay_buffer import ReplayBuffer

def train(agent,env,UPDATE_EVERY,n_episodes=100, tmax=1000,gamma=0.99):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        tmax (int): maximum number of timesteps per episode
        UPDATE_EVERY: how many trajectories to get between updates
        Instead of updating target every (int) steps, using 'soft' updating of .1 to gradually merge the networks
        gamma: discounts the noise over time so that the noise trends to 0
    """
    scores = []
    scores_window = deque(maxlen=100)
    index = 0
    
    N = initialize_N(n_episodes)
    for e in range(1,n_episodes):
        # reset the environment
        gamma *= gamma
        state = env.reset()
        score = 0
        for t in range(1,tmax):
            noise = N[e] * gamma
            action = agent.act(state,noise)[0]
            # print('action',action)
            next_state,reward,done = env.step(action)
            score += reward
            # agent.add(state,action,reward,next_state,done)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
                
            
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\t Average Score: {:.2f}'.format(e, np.mean(scores_window)),end="")
        if e % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))
#             torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break