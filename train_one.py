import numpy as np
from collections import deque

from OUnoise import initialize_N
from replay_buffer import ReplayBuffer

def train(agent,env,UPDATE_EVERY,n_episodes=1800, tmax=1000):
    """Deep Deterministic Policy Gradients.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        Instead of updating target every (int) steps, using 'soft' updating of .1 to gradually merge the networks
    """
    scores = []
    scores_window = deque(maxlen=100)
    index = 0
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    N = initialize_N(n_episodes)
    for e in range(1,n_episodes):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = 0
        for t in range(1,tmax):
            action = agent.act(state,N[e])[0]
#             print('action',action)
            
            env_info = env.step(action)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations         # get next state (for each agent)
#             print('env_info.rewards',env_info.rewards)
            reward = env_info.rewards                         # get reward (for each agent)
            if np.sum(reward) > 0:
                print('reward',reward)
            done = env_info.local_done[0]
            score += env_info.rewards[0]                         # update the score (for each agent)
            state = next_state                                # roll over states to next time step
            agent.add(state,action,reward,next_state,done)   # store memory and learn
            if UPDATE_EVERY % t == 0:
                agent.step(state,action,reward,next_state,done)   # store memory and learn
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