import numpy as np
from collections import deque

from plot import plot

def train(agent,episodes):
    total_rewards = deque(maxlen=100)
    for i_episode in range(1,episodes):#,episodes+1):
        for _ in range(100):
            # get trajectories
            rewards = agent.step()
            # get the average reward of the parallel environments
            total_rewards.append((np.mean(rewards),min(rewards),max(rewards)))

        r_mean,r_min,r_max = total_rewards[-1]
        print("\rEpisode: {}, Number of steps {} \tScores: mean {:.2f}, min {:.2f}, max {:.2f}".format(i_episode,int(i_episode*agent.batch_size),r_mean,r_min,r_max),end="")
        if r_mean > 30:
            print('Env solved!')
            # save plot of rewards
            plot(total_rewards)
            # save policy
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                os.mkdir(directory)
            torch.save(agent.policy.state_dict(), 'model_checkpoints/ppo.ckpt')
    return total_rewards