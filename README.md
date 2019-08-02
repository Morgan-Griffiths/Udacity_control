# Udacity_control

Submission for completing the Udacity Project

## Implementation

PPO

- Generalized Advantage Estimation (GAE)
- Network - shared torso, dual head.

## There are two Environments:

Reacher:

The agent controls a robotic arm with a goal to have the point of the arm reside in the goal zone. Each tick that the arm resides in the goal zone, the agent gets a 0.1 reward. Otherwise the agent gets 0 reward.

There are two types of Reacher environments. One with a single agent, and one with 20 agents. The observations are unique to each agent. The action space is continuous.

- State space = Array of size (33)
- Action space (Continuous) = Array of size (4), each action between -1,1

---

Crawler:


- Set-up: A creature with 4 arms and 4 forearms.
- Goal: The agents must move its body toward the goal direction without falling.
    - CrawlerStaticTarget - Goal direction is always forward.
    - CrawlerDynamicTarget- Goal direction is randomized.
- Agents: The environment contains 3 agent linked to a single Brain.
- Agent Reward Function (independent):
-    +0.03 times body velocity in the goal direction.
-    +0.01 times body direction alignment with goal direction.
- Brains: One Brain with the following observation/action space.
-     Vector Observation space: 117 variables corresponding to position, rotation, velocity, and angular velocities of each limb plus the acceleration and angular acceleration of the body.
-     Vector Action space: (Continuous) Size of 20, corresponding to target rotations for joints.
-     Visual Observations: None.
- Reset Parameters: None

The environment is considered solved under the following conditions:
- Benchmark Mean Reward for CrawlerStaticTarget: 2000
- Benchmark Mean Reward for CrawlerDynamicTarget: 400


## Installation

Clone the repository.

```
git clone git@github.com:MorGriffiths/Udacity_Navigation.git
cd Udacity_Navigation
```

Create a virtual environment and activate it.

```
python -m venv banana
source banana/bin/activate
```

Install Unity ml-agents.

```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/.
```

## Download the Reacher Unity Environment which matches your operating system

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

## Download the Crawler Unity Environment

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

Place the environment into the Environments folder.
If necessary, inside main.py, change the path to the unity environment appropriately


## Project Layout

### Agents

PPO, The following are either unfinished or are in need of learning help (D4PG,DDPG,REINFORCE)

### Buffers

ReplayBuffer, PriorityReplayBuffer (Uses Priority tree)

### Networks

Policy (PPO), Actor/Critic (DDPG), ReinforcePolicy (REINFORCE)

### Main files

main.py
checkpoint.pth


## Run the project

Make sure the env path is set correctly in the main.py file and run
```
python main.py
```

## Watch a trained agent

load the model checkpoint from model_checkpoints
