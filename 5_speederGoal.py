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

import visual

def radian(x):
    return 180 * x / np.pi
"""
saveFile = open("save.txt")    
saveFile.close()
"""
path = r"/home/fish/FISH/prod/code/python/PyBullet/Drone"
graphPath = r"/home/fish/FISH/prod/code/python/PyBullet/Graph"
if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(graphPath):
    os.makedirs(graphPath)

# --- PYBULLET ---

class Env():
    def __init__(self):
        self.step_count = 0
        self.position = 0.0
        self.stepMouvement = 0.02
        self.maxAcceptRotation = 0.2
        self.maxRotation = 0.8
        self.penaltie = 0.2
        self.goalReward = 0.5
        self.checkpoint = 0
        self.checkpointMultiplier = 1

        self.secondDistanceGoal = 0
        self.secondSoftDelta = 0

        self.size = 10
        self.border = 8

        self.joint_ids_base_legs =   [1, 4, 7, 10] # -90 // 90 | -1.5708 // 1.5708
        self.joint_ids_first_legs =  [2, 5, 8, 11] # -45 // 70 | -0.7853 // 1.2217
        self.joint_ids_second_legs = [3, 6, 9, 12] # -65 // 90 | -1.1344 // 1.5708

        self.clip_base_legs =   [-1.5708 , 1.5708]
        self.clip_first_legs =  [-1.2217 , 0.7854]
        self.clip_second_legs = [-1.5708 , 1.1344]

        self.joint_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12]

        self.len_joint_ids = len(self.joint_ids)
        self.target = [self.position] * self.len_joint_ids

        self.state = self.build_env()

    def build_env(self):
        # p.GUI or p.DIRECT for non-graphical version
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0,0,-9.81)

        # charger les fichiers de base
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #not mandatory

        planeId = p.loadURDF("plane.urdf" , [ 0 , 0 , 0 ] , [ 0 , 0 , 0 , 1 ] ) 
        self.goal = p.loadURDF("cube.urdf", [ 0 , 2 , 1 ] , [ 0 , 0 , 0 , 1 ])

        urdf_folder = r"/home/fish/FISH/prod/code/python/PyBullet/urdf/speeder_description/robot.urdf"
        

        sizeFactor = 20
        #spider = p.loadURDF(urdf_folder, [ 0 , 0 , 0 ] , [ 0 , 0 , 0 , 1 ] , globalScaling=sizeFactor , useFixedBase = True)
        self.spider = p.loadURDF(urdf_folder, [ 0 , 0 , 0.14 ] , [ 0 , 0 , 0 , 1 ] , globalScaling=sizeFactor)
        self.obj_of_focus = self.spider

    def reload(self):


        # --- GOAL RESET ---
        aabb = p.getAABB(self.goal)
        goalHeight = aabb[1][2] - aabb[0][2]

        goalX = 0
        goalY = 0

        number = 0
        # goal spawn at a distance from the center 
        # avoid the goal to spawn on the spider
        while number == 0:
            #print("random")
            temporaryX = random.uniform(-self.size, self.size)
            temporaryY = random.uniform(-self.size, self.size)

            if temporaryY < -self.border or temporaryY > self.border or temporaryX < -self.border or temporaryX > self.border:
                #print("graph")
                goalX = temporaryX
                goalY = temporaryY
                number += 1

        p.resetBasePositionAndOrientation(self.goal,[ goalX , goalY , goalHeight / 2 ],[ 1 , 0 , 0 , 0 ])


        # --- SPIDER RESET ---

        EulerSpider = [0,0,90]

        rotation = Rotation.from_euler('xyz', EulerSpider, degrees=True)
        rotation_quaternion = rotation.as_quat()

        p.resetBasePositionAndOrientation(self.spider , [ 0 , 0 , 1 ] , rotation_quaternion)

        p.resetBaseVelocity(self.spider, [0, 0, 0], [0, 0, 0])
        
        p.setJointMotorControlArray(
        self.spider,
        self.joint_ids,
        controlMode=p.VELOCITY_CONTROL,
        forces=[0]*self.len_joint_ids
        )
        
        for i in range(p.getNumJoints(self.spider)):
            p.resetJointState(self.spider, i, targetValue=0, targetVelocity=0)

        for i in range(self.len_joint_ids):
            self.target[i] = 0 # radian
        """
        for _ in range(50):
            p.stepSimulation()

        """
        p.setJointMotorControlArray(
            self.spider,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=self.target
        )

        self.secondDistanceGoal = 0   # ← ajouter
        self.secondSoftDelta = 0

        # -- GOAL DISTANCE --
        # use after for the reward
        self.checkpoint = 0

        reload_state = self.observe()
        reload_DistanceX = reload_state[6]
        reload_DistanceY = reload_state[7]

        self.reload_Distance = math.sqrt(reload_DistanceX**2 + reload_DistanceY**2)

    def observe(self):

        self.spiderPosition, self.spiderOrientation = p.getBasePositionAndOrientation(self.spider)
        self.linearVelocity, _ = p.getBaseVelocity(self.spider)

        self.goalPosition, _ = p.getBasePositionAndOrientation(self.goal)
        

        distanceX = self.goalPosition[0] - self.spiderPosition[0]
        distanceY = self.goalPosition[1] - self.spiderPosition[1]
        distance = math.sqrt(distanceX**2 + distanceY**2)


        self.linkState = p.getLinkState(self.spider,0)

        orientation = self.linkState[1]
        euler = p.getEulerFromQuaternion(orientation)

        # variable for the fall and flip detection
        rotation_pitch = euler[1]
        rotation_roll  = euler[0]
        rotation_yaw   = euler[2]

        self.state = [self.spiderPosition[0], self.spiderPosition[1], self.spiderPosition[2],
                      self.linearVelocity[0], self.linearVelocity[1], self.linearVelocity[2],
                      distanceX             , distanceY             , distance,
                      rotation_pitch        , rotation_roll         , rotation_yaw]

        """
        for i in range(p.getNumJoints(spider)):
            print(p.getJointInfo(spider,i)[3])
        """
        # [1] position | [0] velocity
        for i in range(p.getNumJoints(self.spider)):
            
            self.state.append(p.getJointState(self.spider,i)[0]) 
            self.state.append(p.getJointState(self.spider,i)[1]) 

        #roundState = [ '%.2f' % elem for elem in state]
        #print(len(state))

        return np.array(self.state, dtype=np.float32)

    def apply_action(self,action):
        # 0 -> 23
        #action = 24
        action +=1
        # 1 -> 24

        #print(action)

        if action == 25:
            print("Debug Session")
            p.disconnect()
            pass

        elif action > 12 and action != 25:
            # 13 -> 24
            action -= 12
            # 1 -> 12
            self.target[action] -= self.stepMouvement   

        else :
            # 1 -> 12
            self.target[action] += self.stepMouvement  

        # clip all the joint depending of their joint types
        for i in range(self.len_joint_ids):

            if i in self.joint_ids_base_legs:
                self.target[i] = min(max(self.target[i], self.clip_base_legs[0]) , self.clip_base_legs[1])

            elif i in self.joint_ids_first_legs:
                self.target[i] = min(max(self.target[i], self.clip_first_legs[0]) , self.clip_first_legs[1])

            elif i in self.joint_ids_second_legs:
                self.target[i] = min(max(self.target[i], self.clip_second_legs[0]) , self.clip_second_legs[1])

    def compute_reward(self, nextState,oldState, frames):

        won = False
        done = False
        reward = 0

        resetDistanceGoal = self.reload_Distance

        firstDistanceGoal = 0
        firstSoftDelta = 0


        distance90Percent = resetDistanceGoal * (90/100)
        distance80Percent = resetDistanceGoal * (80/100)
        distance65Percent = resetDistanceGoal * (65/100)
        distance50Percent = resetDistanceGoal * (50/100)
        distance35Percent = resetDistanceGoal * (35/100)


        # get base orientation for the flip check
        self.linkState = p.getLinkState(self.spider,0)

        orientation = self.linkState[1]
        euler = p.getEulerFromQuaternion(orientation)

        # variable for the fall and flip detection
        height        = self.linkState[0][2]
        rotationPitch = euler[1]
        rotationRoll  = euler[0]


        # --- delta calculation ---
        # variable for the deplacement detection
        distance = math.sqrt(nextState[6]**2 + nextState[7]**2)

        OldDistance = math.sqrt(oldState[6]**2 + oldState[7]**2)
        
        # --- pitch/roll/height check ---
        # reset for high roll
        if rotationRoll > self.maxRotation or rotationRoll < -self.maxRotation:
            reward -= self.penaltie
            done = True

        # reset for high pitch
        if rotationPitch > self.maxRotation or rotationPitch < -self.maxRotation:  
            reward -= self.penaltie
            done = True

        if height < 0.4 :
            reward -= self.penaltie
            done = True

        # --- distance check ---
        if not done :
            """
            print(f"distance90 : {distance90Percent}")
            print(f"distance80 : {distance80Percent}")
            print(f"distance65 : {distance65Percent}")
            print(f"distance50 : {distance50Percent}")
            print(f"distance35 : {distance35Percent}")
            
            print(f"distance : {distance}")
            """
            if distance < distance90Percent and self.checkpoint == 0:
                self.checkpoint = 1
                reward += self.goalReward / 4

                self.checkpointMultiplier = 2
                print(f"Reward Multiplier : {self.checkpointMultiplier}")
                print("checkpoint 1")

            if distance < distance80Percent and self.checkpoint == 1:
                self.checkpoint = 2
                reward += self.goalReward * 1

                self.checkpointMultiplier = 1.5
                print(f"Reward Multiplier : {self.checkpointMultiplier}")
                print("checkpoint 2")

            if distance < distance65Percent and self.checkpoint == 2:
                self.checkpoint = 3
                reward += self.goalReward * 2

                self.checkpointMultiplier = 1.75
                print(f"Reward Multiplier : {self.checkpointMultiplier}")
                print("checkpoint 3")

            if distance < distance50Percent and self.checkpoint == 3:
                self.checkpoint = 4
                reward += self.goalReward * 3

                self.checkpointMultiplier = 2
                print(f"Reward Multiplier : {self.checkpointMultiplier}")
                print("checkpoint 4")
            
            if distance < distance35Percent and self.checkpoint == 4:
                self.checkpoint = 5
                reward += self.goalReward * 4

                self.checkpointMultiplier = 2.5
                print(f"Reward Multiplier : {self.checkpointMultiplier}")
                print("checkpoint 5")

            # win
            if distance < 2.5 :
                reward += self.goalReward * 5
                won = True

        framesCheck = 30
        if frames%framesCheck == 0:
            #print(frames)
            firstDistanceGoal = resetDistanceGoal - distance
            
            # Calculate DELTA
            firstSoftDelta = firstDistanceGoal - self.secondDistanceGoal # delta

            self.secondDistanceGoal = firstDistanceGoal # shift of distance

            # calculate Delta interval to limit big reward when falling
            deltaInterval = firstSoftDelta - self.secondSoftDelta

            self.secondSoftDelta = firstSoftDelta
            #print(round(deltaInterval,4))

            if deltaInterval < 0.05:
                #print()
                if firstSoftDelta > 0.005:
                    reward += firstSoftDelta * 2 

                else :
                    reward += firstSoftDelta

            """
            else : 
                print(f"firstDistanceGoal  : {firstDistanceGoal}")
                print(f"secondDistanceGoal : {self.secondDistanceGoal}")
                print(f"firstSoftDelta : {firstSoftDelta}")
                print(f"secondSoftDelta : {self.secondSoftDelta}")
                print(deltaInterval)
            """

            """
            firstDeltaClip = firstSoftDelta - self.secondDistanceGoal

            self.secondSoftDelta = firstSoftDelta
            """

            #print(f"reward : {reward}")
        
        return reward, done, won

# --- PPO ---

# ---[HYPERPARAMETERS]---
lr_actor  = 0.0003 # not implemented
lr_critic = 0.0001 # not implemented
lr = 0.0001
max_grad_norm = 1.0

clip_epsilon = (0.2) # value of the PPO loss
gamma = 0.99
lmbda = 0.95
entropy_eps = 0.001

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
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Softmax(dim=-1)
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
        """
        random.shuffle(self.buffer)

        mini_batch = self.buffer[0:batch_size]
        """
        mini_batch = self.buffer[0:batch_size]
        return mini_batch

#take subdata and extract info 
def extract_data_buffer(subdata, dataNumber):
        data = []
        for transcription in subdata:
            data.append(transcription[dataNumber])

        return data
 
def compute_returns(subdata):
    G = 0
    G_t = []
    for transcription in reversed(subdata):
        reward = transcription[2]
        done = transcription[4]

        G = reward + gamma * G * (1 - done)
        G_t.insert(0,G)

    return G_t

def compute_advantages(subdata, gamma, lambda_):
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

# reset() is link to the ppo | env.reload() is the reset for the agent/environnement
def reset(episodeLength, episode_count_frames, D_buffer, env, graphRewards, graphEpisodeReward):
    episodeLength.append(len(graphRewards))
    episode_count_frames = 0
    graphEpisodeReward.append(graphRewards)
    #print(graphEpisodeReward)
    graphRewards = []

    env.reload()

    return episode_count_frames,graphEpisodeReward,graphRewards
    #print(f"Over {max_training_frames} frames : reset")


def training(frames_per_batch, sub_batch_size, optimizerAdam, model1, max_training_frames, env, buffer_Collect_Size, num_epochs):
    D_buffer = buffer()
    env.reload()
    optimizer = optimizerAdam

    graphRewards = []
    episodeLength = []
    graphValueLoss = []
    graphPolicyLoss = []
    graphEpisodeReward = []

    Won = 0

    total_count_frame = 0
    episode_count_frames = 0

    for i in range (frames_per_batch+1):   
        keys = p.getKeyboardEvents()
        if ord('c') in keys:
            print('save')
            break
    
        if ord('x') in keys:
            env.reload()

        # limit episode length
        if episode_count_frames >= max_training_frames:
            D_buffer.clear_buffer()
            episode_count_frames, graphEpisodeReward,graphRewards = reset(episodeLength,episode_count_frames, D_buffer, env, graphRewards,graphEpisodeReward)
        
        # take info
        Data = env.observe()          # data = state = s

        # put in tensor for the model
        state_tensor = torch.tensor(Data, dtype=torch.float32)

        # calculate the probability | state_value = value_pred
        action_probs, state_value = model1(state_tensor)
        state_value = state_value.detach().item()

        # choose randomly one action while taking the probability
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample().item()        # action = a
        env.apply_action(action)

        # update the joint position
        p.setJointMotorControlArray(env.spider,
                                    env.joint_ids,
                                    p.POSITION_CONTROL,
                                    env.target)
        
        
        focus_position , _ = p.getBasePositionAndOrientation(env.spider)
        """
        p.resetDebugVisualizerCamera(cameraDistance=15,
                                    cameraYaw=0,
                                    cameraPitch=-40,
                                    cameraTargetPosition = focus_position)
        """

        for i in range(5):
            p.stepSimulation()
        #time.sleep(1./240.)
        # --- after action ---

        # use later for the ppo 
        #(to know at which point the policy what thinking if it was right)
        log_prob = action_dist.log_prob(torch.tensor(action, dtype=torch.int64)).item()

        next_state = env.observe()      # next_state = s'

        reward, done, won = env.compute_reward(next_state,Data,episode_count_frames)     # reward = r


        graphRewards.append(reward)

        
        D_buffer.store_buffer(Data, action, reward, next_state, done, state_value, log_prob)
        if done:
            #D_buffer.clear_buffer()
            episode_count_frames, graphEpisodeReward,graphRewards = reset(episodeLength,episode_count_frames, D_buffer, env, graphRewards,graphEpisodeReward)

        elif won:
            #D_buffer.clear_buffer()
            episode_count_frames, graphEpisodeReward,graphRewards = reset(episodeLength,episode_count_frames, D_buffer, env, graphRewards,graphEpisodeReward)

            Won += 1
            print(" - - DONE - - ")

        #print(f"reward : {reward}")
        #print(f"episode reward : {graphEpisodeReward}")

        # learning loop
        if len(D_buffer.buffer) >= buffer_Collect_Size :
            print("ppo")
            subdata = D_buffer.return_buffer()

            # calculate the amount of reward the model should get
            # use to compare the model thus to update the model
            returns = compute_returns(subdata)
    
            advantages = compute_advantages(subdata, gamma, lmbda)
            advantages = torch.tensor(advantages, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for epoch in range(num_epochs):
                indices = list(range(len(subdata))) # list 1,2 ... X | X = length of the subdata
                random.shuffle(indices)             # shuffle the indicies

                for start in range(0, len(subdata), sub_batch_size): # divide the subdata in smaller size
                    idx              = indices[start : start + sub_batch_size]                   # PPO main core is here
                    mini_batch       = [D_buffer.buffer[i] for i in idx] # = subdata
                    advantages_batch = advantages[idx]                   # = advantages
                    returns_batch    = [returns[i] for i in idx]         # = returns

                    optimizer.zero_grad() 

                    # calculate the loss of the policy for the actor
                    policy_loss =  PPO_loss(mini_batch, advantages_batch, model1)
                    graphPolicyLoss.append(policy_loss) # only for the graph

                    # calculate the loss of the policy for the critic
                    value_loss = compute_value_loss(mini_batch, returns_batch, model1)
                    graphValueLoss.append(value_loss) # only for the graph

                    loss = policy_loss + value_loss * 0.5
                    #print(loss)
                    
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model1.parameters(), max_grad_norm)

                    optimizer.step()

            D_buffer.clear_buffer()

        episode_count_frames = episode_count_frames +1
        total_count_frame = total_count_frame + 1

        if total_count_frame % max_training_frames*2 == 0:
            print(f"frame count : {total_count_frame}")
            


        #p.stepSimulation()
        #time.sleep(1./240.)
    print(f"Number of Finish Episode : {Won}")

    #graphRewards         = [round(num, 3) for num in graphRewards]
    episodeLength        = [round(num, 3) for num in episodeLength]

    numpyGraphPolicyLoss = torch.stack(graphPolicyLoss).cpu().detach().tolist()
    numpyGraphPolicyLoss = [round(num, 3) for num in numpyGraphPolicyLoss]

    numpyGraphValueLoss  = torch.stack(graphValueLoss).cpu().detach().tolist()
    numpyGraphValueLoss  = [round(num, 3) for num in numpyGraphValueLoss]



    with open("save.txt", "a") as saveFile:

        saveFile.write("----\n")

        #saveFile.write(f"{graphRewards}\n")
        saveFile.write(f"{episodeLength}\n")
        saveFile.write(f"{numpyGraphPolicyLoss}\n")
        saveFile.write(f"{numpyGraphValueLoss}\n")
        saveFile.write(f"{graphEpisodeReward}\n")
        saveFile.write(f"{num_epochs}\n")

    #saveFile.close()
    
    import graph

    # reset all variable for the next training

    graphRewards = []
    episodeLength = []
    graphValueLoss = []
    graphPolicyLoss = []
    graphEpisodeReward = []

    Won = 0

    total_count_frame = 0
    episode_count_frames = 0


def DEBUG(env):
    env.reload()
    cameraPitch = 40
    reward = 0

    # [0] number | [1] name | [2] type | [3] first position index | [4] first velocity index | [5] flags |
    # [6] damping | [7] friction | [8] lower limit | [9] upper limit | [10] max force | [11] max velocity |
    # [12] link name | [13] joint axis | [14] parent position | [15] parent orientation | [16] parent index


    for i in range(p.getNumJoints(env.spider)):

        name = p.getJointInfo(env.spider,i)[1]
        number = p.getJointInfo(env.spider,i)[0]
        print(f"name : {name} number : {number}")
        jlower = p.getJointInfo(env.spider , i)[8]
        jupper = p.getJointInfo(env.spider , i)[9]
        jlowerDegree = round( radian(p.getJointInfo(env.spider,i)[8]), 3)
        jupperDegree = round( radian(p.getJointInfo(env.spider,i)[9]), 3)

        print(f"{jlower} -> {jlowerDegree}")
        print(f"{jupper} -> {jupperDegree}")
        print()

    while True:
        
        
        keys = p.getKeyboardEvents()

        if ord('x') in keys:
            env.reload()
            

        if ord('e') in keys :
            env.apply_action(1)
            env.apply_action(4)  
            env.apply_action(7)  
            env.apply_action(10)      

        if ord('d') in keys :
            env.apply_action(13)
            env.apply_action(16)  
            env.apply_action(19)  
            env.apply_action(22)    

        if ord('o') in keys : 
            cameraPitch += 0.2
        
        if ord('l') in keys : 
            cameraPitch -= 0.2

        if ord('f') in keys:
            linkState = p.getLinkState(env.spider,0)
            print()
            for i in range(4):
                print(i)
                print(round(linkState[1][i],2))

        linkState = p.getLinkState(env.spider,0)

        orientation = linkState[1]
        height = linkState[0][2]
        minHeight = 0.4

        if minHeight > height:
            height_penalty = -abs(minHeight - height) * 3

        euler = p.getEulerFromQuaternion(orientation)
        #print(f"roll  : {round(radian(euler[1]), 1)}")
        #print(f"pitch : {round(radian(euler[0]), 1)}")
        #print(f"yaw   : {round(radian(euler[2]), 1)}")
        """
        if linkState[0][2] > 0.5 and linkState[0][2] < 0.8 :
            print(linkState[0][2])
        """

        # update the joint position
        p.setJointMotorControlArray(env.spider,
                                    env.joint_ids,
                                    p.POSITION_CONTROL,
                                    env.target)
        
        
        focus_position , _ = p.getBasePositionAndOrientation(env.spider)
            
        p.resetDebugVisualizerCamera(cameraDistance=10,
                                    cameraYaw=0,
                                    cameraPitch=-cameraPitch,
                                    cameraTargetPosition = focus_position)


        p.stepSimulation()
        time.sleep(1./240.)


# 3 * 12 action | ia choose for each joint if it wants to go increase, decrease the joint angle or do nothing
model1 = ActorCritic(state_dim=38, action_dim=24)
optimizerAdam = optim.Adam(model1.parameters(), lr=lr)
env1 = Env()

#env2 = Env()
"""
frames_per_batch = 15000
buffer_Collect_Size = 1500
num_epochs = 10
sub_batch_size = 750
max_training_frames = 3000
training(frames_per_batch, sub_batch_size,optimizerAdam, model1, max_training_frames, env1, buffer_Collect_Size, num_epochs)
"""
while True:
    print("\n[0] Exit | Train [1] | Load [2] | DEBUG [3] | multi [4] ")
    
    choice = input().strip()

    if not choice.isdigit():
        print("Invalid input")
        continue


    QuestionAction = int(choice)
    
    #QuestionAction = 5

    #exit
    if QuestionAction == 0:
        break

    # train
    elif QuestionAction == 1:

        frames_per_batch = 1_000_000
        buffer_Collect_Size = 8192
        num_epochs = 8
        sub_batch_size = 4096
        max_training_frames = 8192

        print("\nGo for training")

        print(f"Number of frames per batch [default = {frames_per_batch}] :")
        print("Input is multiplie by 1000")
        choice = input()
        if choice.isdigit():
            frames_per_batch = int(choice) * 1000
        else:
            frames_per_batch = frames_per_batch

        print(f"Size of the buffer for the PPO [default = {buffer_Collect_Size}] :")
        choice = input()
        if choice.isdigit():
            buffer_Collect_Size = int(choice)
        else:
            buffer_Collect_Size = buffer_Collect_Size

        print(f"Number of epochs [default = {num_epochs}] :")
        choice = input()
        if choice.isdigit():
            num_epochs = int(choice)
        else:
            num_epochs = num_epochs

        print(f"Number of frames per sub batch [default = {sub_batch_size}] :")
        choice = input()
        if choice.isdigit():
            sub_batch_size = int(choice)
        else:
            sub_batch_size = sub_batch_size

        print(f"Max frames per episode [default = {max_training_frames}] :")
        choice = input()
        if choice.isdigit():
            max_training_frames = int(choice)
        else:
            max_training_frames = max_training_frames

        print(f"training model with {frames_per_batch} frames per batch ")
        print(f"training model with {buffer_Collect_Size} data per epochs ")
        print(f"training model with {num_epochs} epochs")
        print(f"training model with {sub_batch_size} frames per sub batch ")
        print(f"training model with {max_training_frames} max frames per episode")
        print()

        training(frames_per_batch, sub_batch_size, optimizerAdam, model1, max_training_frames, env1, buffer_Collect_Size, num_epochs)

        MODEL_NAME = "tempo"
        MODEL_NAME += ".pth"
        MODEL_SAVE_PATH = os.path.join(path, MODEL_NAME)

        torch.save(model1.state_dict(), MODEL_SAVE_PATH)
        print(f"save model : {MODEL_SAVE_PATH}")
        """

        print("Do you want to save the model : yes [1] | no [2] ")
        choiceSave = input().strip()
        Save = choiceSave


        if int(Save) == 2:
            print("model not saved :(")
        else:
            print("Choose model name :")
            MODEL_NAME = input()
            MODEL_NAME += ".pth"
            MODEL_SAVE_PATH = os.path.join(path, MODEL_NAME)

            torch.save(model1.state_dict(), MODEL_SAVE_PATH)
            print(f"save model : {MODEL_SAVE_PATH}")
        """

    # load
    elif QuestionAction == 2:
        print("Choose model name :")

        MODEL_NAME = input()
        MODEL_NAME += ".pth"
        MODEL_LOAD_PATH = os.path.join(path, MODEL_NAME)

        model1.load_state_dict(torch.load(MODEL_LOAD_PATH))
        model1.eval()
        print("Model loaded successfully!")


    elif QuestionAction == 3:
        print("DEBUG")
        DEBUG(env1)

    elif QuestionAction == 4:
        print("Choose model name :")

        MODEL_NAME = input()
        MODEL_NAME += ".pth"
        MODEL_LOAD_PATH = os.path.join(path, MODEL_NAME)

        model1.load_state_dict(torch.load(MODEL_LOAD_PATH))
        model1.eval()
        print("Model loaded successfully!")


        frames_per_batch = 10_000_000
        buffer_Collect_Size = 2048
        num_epochs = 15
        sub_batch_size = 512
        max_training_frames = 8192

        chunk = 80
        divide = frames_per_batch//chunk
        print(divide)
        print(chunk)

        for i in range(divide):
            training(divide, sub_batch_size, optimizerAdam, model1, max_training_frames, env1, buffer_Collect_Size, num_epochs)
            
            MODEL_NAME = str(i)
            MODEL_NAME += ".pth"
            MODEL_SAVE_PATH = os.path.join(path, MODEL_NAME)

            torch.save(model1.state_dict(), MODEL_SAVE_PATH)
            print(f"save model : {MODEL_SAVE_PATH}")

            MODEL_NAME = str(i)
            MODEL_NAME += ".pth"
            MODEL_LOAD_PATH = os.path.join(path, MODEL_NAME)

            model1.load_state_dict(torch.load(MODEL_LOAD_PATH))
            model1.eval()
            print("Model loaded successfully!")


p.disconnect()