"""
Config file for loading hyperparams
"""

import argparse

class Config(object):
    def __init__(self):
        self.gae_lambda=0.95
        self.num_agents=20
        self.batch_size=32
        self.gradient_clip=10
        self.SGD_epoch=10
        self.tmax = 320
        self.epsilon=0.2
        self.beta=0.01
        self.gamma=0.99