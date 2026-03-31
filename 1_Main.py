#3.12.10
import pybullet as p
import pybullet_data
from dataclasses import dataclass

import time
import random
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)

# ---[SET UP PyBullet]---
print("hello PYBullet")

#or p.DIRECT for non-graphical version
# connexion a la simulation
physicsClient = p.connect(p.GUI)


# charger les fichiers de base
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# gravité
p.setGravity(0,0,-9.81)

startOrientation = p.getQuaternionFromEuler([0,0,0])

planeId = p.loadURDF("plane.urdf") #charger le sol

# --charger the cube--
cube = p.loadURDF("cube_small.urdf", [0,0,0.5])

aabb = p.getAABB(cube)
#z min - z max
height = aabb[1][2] - aabb[0][2]
p.resetBasePositionAndOrientation(cube,[0,0,height/2],startOrientation)
"""
# -- charger the objectif --
x = random.uniform(-1.0,1.0)
y = random.uniform(-1.0,1.0)
"""
x,y = 0.5,0.5
GoalId = p.loadURDF("cube_small.urdf", [x,y,0.5]) #charger l'objectif a une position random

#cam variable
cameraDistance = 3
cameraYaw      = 45
cameraPitch    = -30
cameraPosition = [0,0,0]

p.resetDebugVisualizerCamera(
    cameraDistance,
    cameraYaw,
    cameraPitch,
    cameraPosition
)
# ---[HYPERPARAMETERS]---

num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000

sub_batch_size = 64
num_epochs = 10
clip_epsilon = (0.2) # value of the PPO loss
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


class agent:
    def choose_action(state):
        print("Choose action")

    def learn(state, action, reward):
        print("Learn")


def observe(cube, goal):
    cubePosition, cubeRotation = p.getBasePositionAndOrientation(cube)
    cubeVelocity = p.getBaseVelocity(cube)

    goalPosition, goalRotation = p.getBasePositionAndOrientation(goal)

    # CollectObservation
    # Input layer
    state = [
            cubePosition[0], cubePosition[1], cubePosition[2],
            goalPosition[0], goalPosition[1], goalPosition[2]
            ]
    #return np.array(state, dtype=np.float32)
    return np.array(state, dtype=np.float64)

def apply_action(ObjectId, action):
    # object ID | linkindex | vecteur [x, y, z] | where force is gived [0,0,0] | repere of the coordinates

    if action == "Forward":
        # object ID | linkindex | vecteur [x, y, z] | where force is gived [0,0,0] | repere of the coordinates
        p.applyExternalForce(ObjectId, -1, [1 ,0,0], [0,0,0], p.WORLD_FRAME)

    elif action == "Backward":
        p.applyExternalForce(ObjectId, -1, [-1,0,0], [0,0,0], p.WORLD_FRAME)

    elif action == "Left":
        p.applyExternalForce(ObjectId, -1, [0,1 ,0], [0,1,0], p.WORLD_FRAME)

    elif action == "Right":
        p.applyExternalForce(ObjectId, -1, [0,-1,0], [0,0,0], p.WORLD_FRAME)
 
    elif action == "NO":
        p.applyExternalForce(ObjectId, -1, [0,0,0], [0,0,0], p.WORLD_FRAME)    

def compute_reward(objectId,GoalId):
    
    Data = observe(objectId, GoalId)
    Data = np.round(Data,3)

    distanceX = Data[0] - Data[3]
    distanceY = Data[1] - Data[4]
    distanceZ = Data[2] - Data[5]

    distance = math.sqrt(distanceX**2 + distanceY**2 + distanceZ**2)
    print(round(distance, 3))
    return round(distance, 3)
    #print(f"dx : {distanceX} | dy : {distanceY} | dz : {distanceZ}") 
    
for i in range (10000):

    # Choose what to do
    #action = agent.choose_action(state)
    action = "NO"

    # apply it
    apply_action(cube,action)

    #avanced the simulation

    # observe the data after the action
    next_state = observe(cube, GoalId) 

    #then give a reward
    reward = compute_reward(cube, GoalId)

    # finally backpropagation (learn)
    #agent.learn(state, action, reward)

    state = next_state

    time.sleep(1./240.)

p.disconnect()