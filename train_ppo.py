import numpy as np

# Single agent
def train(env,agent,episodes,discount,epsilon,beta,tmax,SGD_epoch):
    total_rewards = []
    for i_episode in range(1,episodes+1):
        # get trajectories
        actions,a_probs,states,rewards,dones = agent.collect_trajectories(env,policy,tmax)
        for _ in range(SGD_epoch):
            # Surrogate
            agent.step(a_probs,states,actions,rewards,discount,epsilon,beta)
        
        # the clipping parameter reduces as time goes on
        epsilon*=.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995

        # get the average reward of the parallel environments
        total_rewards.append(sum(rewards))

        # display some progress every 20 iterations
        if i_episode % 50 == 0:
            print("Episode: {0:d}, score: {1:f}".format(i_episode,total_rewards[-1]))