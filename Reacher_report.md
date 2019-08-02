# Reacher Report

At first i tried to solve the environment with a DDPG agent. Funnily enough, initially i didn't get any rewards. Which was due to not using any exploration noise (Thats the thing about acting deterministically! No stochasticity, no exploration). However even after i fixed that, i found DDPG to be extremely unreliable in performance. Even with regards to simple environments like MountainCarContinuous and Pendulum. I was further disheartened by the fact that when i checked other people's implementations of DDPG, they were ALSO extremely unreliable in performance. Sometimes failing to learn at all. From reading the paper on D4PG, this seems like the necessary improvement on the algorithm, but also would take some time to implement myself, since i hadn't implemented the C51 distributional DQN algorithm in the previous project. 

One thing that became very obvious to me, is just how important the hyperparameters are. During my quest for DDPG heaven, i found a lot of github 'solutions' that failed to work on their default parameters. Or failed to run without errors. I throughly appreciate the reviewer's job on these projects now :) And keeping code clean and modulurized which i am guilty of fudging as well. All this is to say that i discarded the DDPG agent in favor of PPO.

The PPO model itself is deceptively simple:
1. Collect trajectories
2. calculate the returns and Generalized advantage estimates
3. Learn on the samples
Repeat steps 1-3

However i managed to make it into quite an ordeal!

Some issues i ran into:
I tried to use both of my GPUs, but since all tensors must be calculated on the same device that runs the policy, 2 seems redundant. I suppose if i had a distributed setup then 2 would work.

Initially kept the shapes of the batches as [batch_size,num_agents], however this means that each agents experience is only trained on by itself and the samples are not shared amongst the others. By flattening the experiences, then they can be shared across all agents. 

Strangely i had a hell of a time getting things to learn. I kept running into the same problem where it would appear that everything is fine, but the agent would staunchly refuse to learn. This was the cause of much (so much, oh so much) aggrevation. As debugging RL agents can be really quite difficult. Ultimately i ended up completely reworking calculating the advantages from scratch. As that seemed to be the only possible place where i had screwed things up. And indeed after i did that, the agent finally could learn. 

During the course of the agent not learning, i learned a lot about the need to modularize your functions so you can unit test them. And i think going forward i will make sure to build the agents in such a way as to be able to test all the algorithms seperately. And in addition be able to test the network's capability to learn on a simplier environment. This last point seems more difficult as it requires the flexibility to be able to handle many environments. But from my last experience it would have been oh so worth it!

There also technically should be the VF coefficient for when you use a dual head network. However it seems to train fine without? donno what thats all about.

## Experiments:

**batch non-continuity**
I experimented with non-continuity in the batches. So instead of getting a sequence, get a series of random spots. I assumed this would improve performance because it might reduce the bias of how it has been behaving. However in my small sample size it took longer to converge. This is probably because it is necessary to learn sequences to get the arm to the goal. The goal is solved normally in 68 minutes. And without sequences it was solved in ~2x that time.

**Model weight initialization**
I tried using the ddpg weight initialization, because while extensively debugging the program i noticed that the state value outputs were often quite large (either pos or neg). And given that the rewards are quite small (0.1), it will usually be way off on the value function. So i thought it might make sense to initialize the network with very small weights to cap the value projections. This didn't seem to make any difference however. In the same vein i thought about scaling the reward which i mention down below in Future Work.

Hyperparams:

Parameter | Value | Description
------------ | ------------- | -------------
SGD_epoch | 10 | Number of training epoch per iteration
Batch size | 32*20 | batch size updated for all agents
tmax | 320 | Number of steps per trajectory 
Gamma | 0.99 | Discount rate 
Epsilon | 0.2 | Ratio used to clip r = new_probs/old_probs during training
Gradient clip | 10.0 | Maximum gradient norm 
Learning rate | 1e-4 | Learning rate 
Beta | 0.01 | Entropy coefficient 
Trace decay | 0.95 | Lambda for Generalized Advantage Estimate

Future work
One thing i noticed while combing through my program in debug mode, was the the initial state estimates can be quite large (neg or pos). Whereas the rewards are 0.1 or 0. which made me wonder about 2 things:
- scaling the rewards so that the signal is louder.
- normalizing the state input

I also had a question:
Why does using the squared difference for the ratio work instead of division. And when should you normalize the state space?

Attributions:
Udacity, @github - Ostamand (using his hyperparams), ikostrikov, ShangtongZhang, bentrevett

