#3.12.10
#import AutoPyPack
import pybullet as p
import pybullet_data
from dataclasses import dataclass

import os
import time
import random
import math
import numpy as np
from scipy.spatial.transform import Rotation

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

seed = 42 
 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super().__init__()


        # X depends on the state_dim and actio_dim
        # X = 32
        # X input -> 256 hidden neurone -> 256 hidden neurone -> 256 hidden neurone
        # this is a common part and divided to actor and to critic
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim ),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim  ),
            nn.ReLU()
        )

        # 256 hidden neurone -> 128 hidden neurone -> 64 hidden neurone -> X probability with the softmax
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim // 2 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, sum(action_dim)),
            #nn.Softmax(dim=-1)
        )

        # 256 hidden neurone -> 128 hidden neurone -> 64 hidden neurone -> 32 hidden neurone -> 1 estimation of how many point the future ia can win
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim // 2 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 8 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 8, 1) 
        )

    def forward(self, x):
        x = self.shared_layers(x)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

nvec = [3] * 12
print(nvec)

model1 = ActorCritic(state_dim=2, action_dim=nvec)
optimizerAdam = optim.Adam(model1.parameters(), lr=0.001)

x = torch.tensor([1.0, -1.0])
softmax = nn.Softmax(dim=-1)

# calculate the probability | state_value = value_pred
action_probs, _ = model1(x)
#print(f"action probs : {action_probs}")
#print(f"action_probs shape : {action_probs.shape}")

split_action = torch.split(action_probs, nvec)
#print(f"split : {split_action}")
#print(f"taille d'un groupe : {split_action[0].shape}")

action = []
log_prob = 0

for groups in split_action:
    #print(groups)
    groupsActionProbalities = softmax(groups)
    action_dist = torch.distributions.Categorical(groupsActionProbalities)  
    
    sample_action = action_dist.sample()

    action.append(sample_action.item()) # action = a

    # use later for the ppo 
    #(to know at which point the policy what thinking if it was right)
    log_prob += action_dist.log_prob(sample_action).item()

print(action)
print(log_prob)

