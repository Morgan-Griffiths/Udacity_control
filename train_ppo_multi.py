def collect_multi_trajectories(env,policy,tmax):
    n = 8
    nrand = 2
    # number of parallel instances
    n=len(envs.ps)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]

    envs.reset()
    
    # start all parallel agents
    envs.step([1]*n)
    
    # perform nrand random steps
    for _ in range(nrand):
        fr1, re1, _, _ = envs.step(np.random.choice([RIGHT, LEFT],n))
        fr2, re2, _, _ = envs.step([0]*n)
    
    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into 
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = torch.tensor([fr1,fr2],dtype=torch.float32)
        
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()
        
        ###
        print('probs',probs.shape)
        print('np.random.rand(n)',np.random.rand(n).shape)
        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)
        
        
        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2
        
        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)
        
        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, \
        action_list, reward_list

def clipped_surrogate_multi(policy, old_probs, states, actions, rewards,
                      discount = 0.995, epsilon=0.1, beta=0.01):
    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)
    
    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
    
    print('entropy',entropy)
    print('clipped_surrogate',clipped_surrogate)
    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)

# Single agent
def train_multi(env,policy,optimizer,episodes,discount,epsilon,beta,tmax,SGD_epoch):
    mean_rewards = []
    for i_episode in range(1,episodes+1):
        # get trajectories
        actions,a_probs,states,rewards,dones = collect_multi_trajectories(env,policy,tmax)
        for _ in range(SGD_epoch):
            # Surrogate
            L = -clipped_surrogate_multi(policy,a_probs,states,actions,rewards,discount,epsilon,beta)

            optimizer.zero_grad()
            # for batch loss
            print('L',L)
            L.backward()
            optimizer.step()
            del L
        
        # the clipping parameter reduces as time goes on
        epsilon*=.999

        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.995

        # get the average reward of the parallel environments
        total_rewards.append(sum(rewards))

        # display some progress every 20 iterations
        if i_episode % 20 == 0:
            print("Episode: {0:d}, score: {1:f}".format(i_episode,total_rewards[-1]))
    
def main_multi():
    seed = 1234
    envs = parallelEnv('CartPole-v0', n=8, seed=1234)
    env = gym.make('CartPole-v0')
    nA = env.action_space.n
#     nA = 1
    nS = env.observation_space.shape[0]
    del env
    policy = Policy(seed,nS,nA).to(device)
    optimizer = optim.Adam(policy.parameters(), lr = 1e-2)
    
    discount_rate = .995
    epsilon = 0.1
    beta = .01
    tmax = 200
    SGD_epoch = 4
    episodes = 500
    
    train_multi(envs,policy,optimizer,episodes,discount_rate,epsilon,beta,tmax,SGD_epoch)