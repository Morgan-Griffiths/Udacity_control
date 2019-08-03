import numpy as np
from collections import deque
import time
import pickle

from plot import plot

def train(agent,episodes,path):
    tic = time.time()
    means = []
    stds = []
    rewards_sum = deque(maxlen=10)
    for i_episode in range(1,episodes):#,episodes+1):
        # get trajectories
        rewards = agent.step()
        # get the average reward of the parallel environments
        rewards_sum.append(np.sum(rewards))
        means.append(np.mean(rewards_sum))
        stds.append(np.std(rewards_sum))

        if i_episode % 10 == 0:
            toc = time.time()
            r_mean = np.mean(rewards_sum)
            r_max = max(rewards_sum)
            r_min = min(rewards_sum)
            r_std = np.std(rewards_sum)
            plot(means,stds)
            print("\rEpisode: {} out of {}, Steps {}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(i_episode,episodes,int(i_episode*agent.tmax*20),r_mean,r_min,r_max,r_std,(toc-tic)/60))
            if r_mean > 30:
                print('Env solved!')
                # save scores
                pickle.dump([means,stds], open('ppo_scores.p', 'wb'))
                # save policy
                agent.save_weights(path)
                break