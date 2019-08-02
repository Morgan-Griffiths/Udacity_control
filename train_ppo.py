import numpy as np
from collections import deque
import time
import pickle

from plot import plot

def train(agent,episodes,path):
    tic = time.time()
    scores = []
    rewards_mean = deque(maxlen=10)
    rewards_max = deque(maxlen=10)
    rewards_min = deque(maxlen=10)
    rewards_sum = deque(maxlen=10)
    for i_episode in range(1,episodes):#,episodes+1):
        # get trajectories
        rewards = agent.step()
        # get the average reward of the parallel environments
        rewards_mean.append(np.mean(rewards))
        rewards_max.append(min(rewards))
        rewards_min.append(max(rewards))
        rewards_sum.append(np.sum(rewards))
        scores.append((np.sum(rewards),max(rewards),np.mean(rewards),min(rewards)))

        if i_episode % 10 == 0:
            toc = time.time()
            r_sum = np.mean(rewards_sum)
            r_mean = np.mean(rewards_mean)
            r_max = max(rewards_mean)
            r_min = min(rewards_mean)
            print("\rEpisode: {} out of {}, Steps {} Rewards: total {:.2f}, mean {:.2f}, min {:.2f}, max {:.2f}, Elapsed {:.2f}".format(i_episode,episodes,int(i_episode*agent.tmax*20),r_sum,r_mean,r_min,r_max,(toc-tic)/60))
            if r_sum > 30:
                print('Env solved!')
                # save plot of rewards
                plot('ppo',scores)
                # save scores
                pickle.dump(scores, open('ppo_scores.p', 'wb'))
                # save policy
                agent.save_weights(path)
                break
    return total_rewards