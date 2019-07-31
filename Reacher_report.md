# Reacher Report

At first i tried to solve the environment with a DDPG agent. Funnily enough, initially i didn't get any rewards. Which was due to not using any exploration noise (Thats the thing about acting deterministically! No stochasticity, no exploration). However even after i fixed that, i found DDPG to be extremely unreliable in performance. Even with regards to simple environments like MountainCarContinuous and Pendulum. I was further disheartened by the fact that when i checked other people's implementations of DDPG, they were ALSO extremely unreliable in performance. Sometimes failing to learn at all. From reading the paper on D4PG, this seems like the necessary improvement on the algorithm, but also would take some time to implement myself, since i hadn't implemented the C51 distributional DQN algorithm in the previous project. 

One thing that became very obvious to me, is just how important the hyperparameters are. During my quest for DDPG heaven, i found a lot of github 'solutions' that failed to work on their default parameters. Or failed to run without errors. I throughly appreciate the reviewer's job on these projects now :) And keeping code clean and modulurized which i am guilty of fudging as well. All this is to say that i discarded the DDPG agent in favor of PPO.

Using the squared difference for the ratio instead of division.
Normalize the state space or not?


Attributions:
I could not have completed the project without the following:
Udacity, Ostamand, ikostrikov, ShangtongZhang, bentrevett

