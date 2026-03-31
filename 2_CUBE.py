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

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import visual
"""
SEED = 7
torch.manual_seed(SEED)
"""
# ---[SET UP PyBullet]---
print("hello PYBullet")

# p.GUI
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
cube = p.loadURDF("cube.urdf", [0,0,0.5])

GoalId = p.loadURDF("cube.urdf", [0.5,0.5,0.5]) #charger l'objectif a une position random

p.changeDynamics(planeId, -1, lateralFriction=0.5)  # default 0.5
p.changeDynamics(cube, -1, lateralFriction=2.0) 

size = 10
border = 2

def reload(cube,GoalId):
    aabb = p.getAABB(cube)
    height = aabb[1][2] - aabb[0][2]
    p.resetBasePositionAndOrientation(cube,[0,0,height/2],startOrientation)

    # -- charger the objectif --
    if random.random() > 0.5: 
        x = random.uniform(-size + border, -border)
    else: 
        x = random.uniform(border, size - border)

    if random.random() > 0.5:
         y = random.uniform(-size + border,-border)
    else:
         y = random.uniform(border, size - border)

    aabb = p.getAABB(GoalId)
    height = aabb[1][2] - aabb[0][2]
    p.resetBasePositionAndOrientation(GoalId,[x,y,height/2],startOrientation)

reload(cube,GoalId)

#cam variable
cameraDistance = 20
cameraYaw      = 0
cameraPitch    = -50
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

total_frames = 50_000

max_training_frames = 1024

num_epochs = 4
clip_epsilon = (0.2) # value of the PPO loss

gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-3

strengh = 30

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 32):
        super().__init__()

        # 6 input -> 16 hidden neurone -> 8 hidden neurone
        # this is a common part and divided to actor and to critic
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2 ),
            nn.ReLU()
        )

        # 64 hidden neurone -> 32 hidden neurone -> 5 probability with the softmax
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Softmax(dim=-1)
        )

        # 64 hidden neurone -> 32 hidden neurone -> 1 estimation of how many point the future ia can win
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4 ),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        x = self.shared_layers(x)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

# 0 state | 1 action | 2 reward | 3 next_state | 4 done | 5 state_value | 6 log_prob
class buffer:
    def __init__(self):
        self.buffer = []
    # store info | clear all | return
    def store_buffer(self, state, action, reward, next_state, done, state_value, log_prob):
        self.buffer.append(( state, action, reward, next_state, done, state_value, log_prob))

    def clear_buffer(self):
        self.buffer = []

    def return_buffer(self):
        return self.buffer

    def sample(self, batch_size):
        random.shuffle(self.buffer)

        mini_batch = self.buffer[0:batch_size]

        return mini_batch

def observe(cube, goal):
    cubePosition, cubeRotation = p.getBasePositionAndOrientation(cube)
    linearVelocity, _ = p.getBaseVelocity(cube)

    goalPosition, goalRotation = p.getBasePositionAndOrientation(goal)

    # CollectObservation
    # Input layer

    dx = cubePosition[0] - goalPosition[0]
    dy = cubePosition[1] - goalPosition[1]
    dz = cubePosition[2] - goalPosition[2]

    state = [
            cubePosition[0]  , cubePosition[1]  , cubePosition[2],
            linearVelocity[0], linearVelocity[1], linearVelocity[2],
            goalPosition[0]  , goalPosition[1]  , goalPosition[2],
            dx, dy, dz
            ]

    """
    distance = compute_distance(state)
    state.append(distance)
    """

    #round_state = np.round(state, decimals=3)
    return np.array(state, dtype=np.float32)

def huber_loss(x, y, delta = 1.5):
    loss = x - y
    if abs(loss) <= delta:
        return 0.5 * loss **2
    else:
        return delta * (abs(loss)- 0.5 * delta)

# set to 1 for debug
def apply_action(ObjectId, action):
    # object ID | linkindex | vecteur [x, y, z] | where force is gived [0,0,0] | repere of the coordinates

    #action = 4

    if action == 0: #Forward
        # object ID | linkindex | vecteur [x, y, z] | where force is gived [0,0,0] | repere of the coordinates
        p.applyExternalForce(ObjectId, -1, [strengh ,0,0], [0,0,0], p.WORLD_FRAME)

    elif action == 1: #Backward
        p.applyExternalForce(ObjectId, -1, [-strengh,0,0], [0,0,0], p.WORLD_FRAME)

    elif action == 2: #Left
        p.applyExternalForce(ObjectId, -1, [0,strengh ,0], [0,0,0], p.WORLD_FRAME)

    elif action == 3: #Right
        p.applyExternalForce(ObjectId, -1, [0,-strengh,0], [0,0,0], p.WORLD_FRAME)
 
    elif action == 4: #NO
        p.applyExternalForce(ObjectId, -1, [0,0,0], [0,0,0], p.WORLD_FRAME)    

def touch_border(Data):
    reset = False
    if Data[0] > size:
        reset = True
    elif Data[0] < -size:
        reset = True
    elif Data[1] > size:
        reset = True
    elif Data[1] < -size:
        reset = True
    else:
        reset = False
    return reset

#take subdata and extract info 
def extract_data_buffer(subdata, dataNumber):
        data = []
        for transcription in subdata:
            data.append(transcription[dataNumber])

        return data

def compute_distance(Data):
    
    #Data = np.round(Data,3)

    distanceX = Data[0] - Data[6]
    distanceY = Data[1] - Data[7]
    distanceZ = Data[2] - Data[8]

    distance = math.sqrt(distanceX**2 + distanceY**2 + distanceZ**2)

    return distance
    #print(f"dx : {distanceX} | dy : {distanceY} | dz : {distanceZ}") 
    
def compute_reward(distance1,distance2):
    reward = 0
    direction = 0
    distance = 1
    done = False
    #reward = (size - distance2) / size

    delta = (distance1 - distance2) 
    invertDistance = size - distance2


    # trop de bruit besoin de courbe plus douce
    """
    if delta > 0:
        direction = 1
    else:
        direction = -0

    distance = invertDistance * direction
    """

    if distance2 < 1.4:
        reward = reward + 5
        done = True

    reward = reward + delta

    return reward, done

def compute_returns(subdata):
    G = 0
    G_t = []
    for transcription in reversed(subdata):
        reward = transcription[2]
        done = transcription[4]

        G = reward + gamma * G * (1 - done)
        G_t.insert(0,G)

    return G_t

def compute_advantages(subdata, values, dones, gamma, lambda_):
    #Data = D_buffer.return_buffer()
    
    advantages = []
    last_advantage = 0

    rewards = extract_data_buffer(subdata, 2)
    values = extract_data_buffer(subdata, 5)
    values.append(0)
    dones = extract_data_buffer(subdata, 4)

    for t in reversed(range(len(rewards))):

        mask = 1 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_advantage = delta + gamma * lambda_ * mask * last_advantage
        advantages.insert(0, last_advantage)
    
    return advantages


    """
    advantages = []

    for transcription in subdata:

        reward = transcription[2]

        next_state_value = transcription[3]

        with torch.no_grad():
            next_state_value = model1(torch.tensor(next_state_value, dtype=torch.float32).unsqueeze(0))[1].item()

        state_value = transcription[5]
        done = transcription[4]
        
        A = reward + gamma * next_state_value * (1 - done) - state_value

        advantages.append(A)

    return advantages
    """
    #gt - state value
    #rt+1​+γV(st+1​)−V(st​)
    # reward + gamma * next state value - state_value

def compute_value_loss(subdata, returns, model1):
    states = extract_data_buffer(subdata, 0)
    states = np.array(states)
    states_tensor = torch.tensor(states, dtype=torch.float32)

    returns_tensor = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

    action_probs, value_pred = model1(states_tensor)


    loss_fn = nn.SmoothL1Loss()
    value_loss = loss_fn(value_pred, returns_tensor).mean()

    return value_loss

def PPO_loss(subdata, advantages, model1):
    states = extract_data_buffer(subdata, 0)
    action = extract_data_buffer(subdata, 1)
    old_log_prob = extract_data_buffer(subdata, 6)

    states = np.array(states) # converting to np.array to be faster
    action = np.array(action) # and removre the warning :D

    action_tensor = torch.tensor(action, dtype=torch.int64)
    state_tensor = torch.tensor(states, dtype=torch.float32)  # everything in tensor for the next math

    action_probs, _ = model1(state_tensor)
    action_dist = torch.distributions.Categorical(action_probs)

    new_log_prob = action_dist.log_prob(action_tensor)
    old_log_prob = torch.tensor(old_log_prob, dtype=torch.float32)

    policy_ratio = torch.exp(new_log_prob - old_log_prob)

    surrogate_loss_1 = policy_ratio * advantages.detach()

    #print(surrogate_loss_1)

    surrogate_loss_2 = torch.clamp(policy_ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages

    #print(surrogate_loss_2)

    entropy = action_dist.entropy().mean()

    entropy_bonus = entropy_eps * entropy

    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    policy_loss = -torch.mean(surrogate_loss) - entropy_bonus
    
    return policy_loss

def training(frames_per_batch, sub_batch_size, model1, max_training_frames):

    D_buffer = buffer()

    graphReward = []
    episodeLength = []
    optimizer = optim.Adam(model1.parameters(), lr=lr)

    count_frame = 0
    count_max_training_frames = 0

    for i in range (frames_per_batch):    

        keys = p.getKeyboardEvents()
        if ord('s') in keys:
            print('save')
            break



        # take info
        Data = observe(cube, GoalId)          # data = state = s

        addReward = 0

        touchBorder = touch_border(Data)

        distance1 = compute_distance(Data)

        if count_max_training_frames >= max_training_frames:
            reload(cube,GoalId)

            episodeLength.append(count_max_training_frames)

            count_max_training_frames = 0
            print(f"Over {max_training_frames} frames : reset")

        
        if touchBorder:
            episodeLength.append(count_max_training_frames)
            count_max_training_frames = 0
            addReward = -20

            reload(cube,GoalId)
            print("Border touch : reset ")
        
        # put in tensor for the model
        state_tensor = torch.tensor(Data, dtype=torch.float32)

        # calculate the probability | state_value = value_pred
        action_probs, state_value = model1(state_tensor)
        state_value = state_value.detach().item()

        # choose randomly one action while taking the probability
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()        # action = a

        apply_action(cube,action) # apply the action

        p.stepSimulation()
        #time.sleep(1./240.)

        # use later for the ppo 
        #(to know at which point the policy what thinking if it was right)
        log_prob = action_dist.log_prob(torch.tensor(action, dtype=torch.int64)).item()

        next_state = observe(cube, GoalId)      # next_state = s'

        distance2 = compute_distance(next_state)
        
        reward, done = compute_reward(distance1,distance2)     # reward = r

        reward = reward + addReward

        if done:
            episodeLength.append(count_max_training_frames)
            count_max_training_frames = 0
            reload(cube,GoalId)
            print(" - - GOALLL reset - - ")
        
        graphReward.append(reward)

        D_buffer.store_buffer(Data, action, reward, next_state, done, state_value, log_prob)

        # trainig loop
        if len(D_buffer.buffer) >= sub_batch_size:

            optimizer.zero_grad()

            subdata = D_buffer.sample(sub_batch_size)


            # calculate the amount of reward the model should get
            # use to compare the model thus to update the model
            returns = compute_returns(subdata)
            
            advantages = compute_advantages(subdata, model1,done, gamma, lmbda)
            
            advantages = torch.tensor(advantages, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


            # calculate the loss of the policy for the actor
            policy_loss =  PPO_loss(subdata, advantages, model1)

            # calculate the loss of the policy for the critic
            value_loss = compute_value_loss(subdata, returns, model1)

            
            

            loss = policy_loss + value_loss 
            #print(loss)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_grad_norm)

            optimizer.step()

            D_buffer.clear_buffer()

        count_max_training_frames = count_max_training_frames +1
        count_frame = count_frame + 1

        if count_frame % max_training_frames*4 == 0:
            print(f"frame count : {count_frame}")


        #p.stepSimulation()
        #time.sleep(1./240.)
    visual.tensorGraphic(graphReward, episodeLength)


def test(frames_per_batch, model1, max_training_frames):

    graphReward = []
    episodeLength = []

    count_frame = 0
    count_max_training_frames = 0

    for i in range (frames_per_batch):    
        # take info
        Data = observe(cube, GoalId)          # data = state = s

        addReward = 0

        touchBorder = touch_border(Data)

        distance1 = compute_distance(Data)

        if count_max_training_frames >= max_training_frames:
            reload(cube,GoalId)

            episodeLength.append(count_max_training_frames)

            count_max_training_frames = 0
            print(f"Over {max_training_frames} frames : reset")

        if touchBorder:
            reload(cube,GoalId)
            episodeLength.append(count_max_training_frames)
            count_max_training_frames = 0
            addReward = -20
            print("Border touch : reset ")

        # put in tensor for the model
        state_tensor = torch.tensor(Data, dtype=torch.float32)

        # calculate the probability | state_value = value_pred
        action_probs, _ = model1(state_tensor)

        # choose randomly one action while taking the probability
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()        # action = a

        apply_action(cube,action) # apply the action

        p.stepSimulation()
        #time.sleep(1./240.)

        next_state = observe(cube, GoalId)      # next_state = s'

        distance2 = compute_distance(next_state)
        
        reward, done = compute_reward(distance1,distance2)     # reward = r

        reward = reward + addReward

        if done:
            episodeLength.append(count_max_training_frames)
            count_max_training_frames = 0
            reload(cube,GoalId)
            print(" - - GOALLL reset - - ")
        
        graphReward.append(reward)

        count_max_training_frames = count_max_training_frames +1
        count_frame = count_frame + 1

        if count_frame % max_training_frames == 0:
            print(f"frame count : {count_frame}")


        #p.stepSimulation()
        #time.sleep(1./240.)
    visual.tensorGraphic(graphReward, episodeLength)

path = r"/home/fish/FISH/prod/code/python/PyBullet/Model"

if not os.path.exists(path):
    os.makedirs(path)

model1 = ActorCritic(state_dim=12, action_dim=4)

while True:
    print("\n[0] Exit | Train [1] | Load [2] | Use [3] | Delete [4]")
    choice = input().strip()

    if not choice.isdigit():
        print("Invalid input")
        continue

    QuestionAction = int(choice)

    #exit
    if QuestionAction == 0:
        break

    # train
    elif QuestionAction == 1:

        print("\nGo for training")
        print("Number of frames per batch [default = 122880] :")
        choice = input()
        if choice.isdigit():
            frames_per_batch = int(choice)
        else:
            frames_per_batch = 122880

        print("Number of frames per sub batch [default = 128] :")
        choice = input()
        if choice.isdigit():
            sub_batch_size = int(choice)
        else:
            sub_batch_size = 128

        print("Max frames per episode [default = 2048] :")
        choice = input()
        if choice.isdigit():
            max_training_frames = int(choice)
        else:
            max_training_frames = 2048
        


        print(f"training model with {frames_per_batch} frames per batch ")
        print(f"training model with {sub_batch_size} frames per sub batch ")
        print(f"training model with {max_training_frames} max frames per episode")

        training(frames_per_batch, sub_batch_size, model1, max_training_frames)


        print("Do you want to save the model : yes [1] | no [2] ")
        choiceSave = input().strip()
        Save = choiceSave

        if Save == 2:
            print("model not saved :(")
        else:
            print("Choose model name :")
            MODEL_NAME = input()
            MODEL_NAME += ".pth"
            MODEL_SAVE_PATH = os.path.join(path, MODEL_NAME)

            torch.save(model1.state_dict(), MODEL_SAVE_PATH)
            print(f"save model : {MODEL_SAVE_PATH}")

    # load
    elif QuestionAction == 2:
        print("Choose model name :")

        MODEL_NAME = input()
        MODEL_NAME += ".pth"
        MODEL_LOAD_PATH = os.path.join(path, MODEL_NAME)

        model1.load_state_dict(torch.load(MODEL_LOAD_PATH))
        model1.eval()
        print("Model loaded successfully!")

    # test
    elif QuestionAction == 3:

        print("\nGo for a test")
        print("Number of frames [default = 2048] :")
        choice = input()
        if choice.isdigit():
            frames_per_batch = int(choice)
        else:
            frames_per_batch = 2048
        
        print("Max frames per episode [default = 1024] :")
        choice = input()
        if choice.isdigit():
            max_training_frames = int(choice)
        else:
            max_training_frames = 1024

        print(f"Number of frame : {frames_per_batch}")
        print(f"training model with {max_training_frames} max frames per episode")


        test(frames_per_batch, model1, max_training_frames)

    elif QuestionAction == 4:
        print("Exit [0] | Choose a model name :")
        choiceDelete = input().strip()
        
        if choiceDelete == "0":
            print("no files has been deleted")

        else :
            MODEL_DELETE_PATH = os.path.join(path, choiceDelete)
            MODEL_DELETE_PATH += ".pth"
            os.remove(MODEL_DELETE_PATH)

p.disconnect()