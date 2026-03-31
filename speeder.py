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

def radian(x):
    return 180 * x / np.pi

# p.GUI or p.DIRECT for non-graphical version
physicsClient = p.connect(p.GUI)
p.setGravity(0,0,-9.81)

# charger les fichiers de base
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #not mandatory

urdf_folder = r"/home/fish/FISH/prod/code/python/PyBullet/urdf/speeder_description/robot.urdf"
planeId = p.loadURDF("plane.urdf" , [ 0 , 0 , 0 ] , [ 0 , 0 , 0 , 1 ] ) #charger le sol

sizeFactor = 20
#spider = p.loadURDF(urdf_folder, [ 0 , 0 , 0 ] , [ 0 , 0 , 0 , 1 ] , globalScaling=sizeFactor , useFixedBase = True)
spider = p.loadURDF(urdf_folder, [ 0 , 0 , 0 ] , [ 0 , 0 , 0 , 1 ] , globalScaling=sizeFactor)
obj_of_focus = spider

# [0] number | [1] name | [2] type | [3] first position index | [4] first velocity index | [5] flags |
# [6] damping | [7] friction | [8] lower limit | [9] upper limit | [10] max force | [11] max velocity |
# [12] link name | [13] joint axis | [14] parent position | [15] parent orientation | [16] parent index

for i in range(p.getNumJoints(spider)):

    name = p.getJointInfo(spider,i)[1]
    number = p.getJointInfo(spider,i)[0]
    print(f"name : {name} number : {number}")
    jlower = p.getJointInfo(spider , i)[8]
    jupper = p.getJointInfo(spider , i)[9]
    jlowerDegree = round( radian(p.getJointInfo(spider,i)[8]), 3)
    jupperDegree = round( radian(p.getJointInfo(spider,i)[9]), 3)

    print(f"{jlower} -> {jlowerDegree}")
    print(f"{jupper} -> {jupperDegree}")
    print()

# robot | type of mouvement , target position

position = 0.0
stepMouvement = 0.05

joint_ids_base_legs =   [0, 3, 6, 9] # -90 // 90 | -1.5708 // 1.5708
joint_ids_first_legs =  [1, 4, 7, 10] # -45 // 70 | -0.7853 // 1.2217
joint_ids_second_legs = [2, 5, 8, 11] # -65 // 90 | -1.1344 // 1.5708

clip_base_legs =   [-1.5708 , 1.5708]
clip_first_legs =  [-0.7853 , 1.2217]
clip_second_legs = [-1.1344 , 1.5708]

print(clip_base_legs[0] , clip_base_legs [1])
print(clip_first_legs[0] , clip_first_legs [1])
print(clip_second_legs[0] , clip_second_legs [1])

joint_ids = joint_ids_base_legs + joint_ids_first_legs + joint_ids_second_legs
#joint_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12]
print(joint_ids)

len_joint_ids = len(joint_ids)
print(len_joint_ids)
target = [position] * len_joint_ids
cliped = [position] * len_joint_ids 

while True:
    keys = p.getKeyboardEvents()

    if ord('x') in keys:
        target[1] += stepMouvement # radian
        target[2] += stepMouvement # radian

    
    if ord('c') in keys:
        target[1] -= stepMouvement # radian
        target[2] -= stepMouvement # radian
    
    if ord('e') in keys:
        target[3] += stepMouvement # radian
        target[4] += stepMouvement # radian

    if ord('d') in keys:
        target[3] -= stepMouvement # radian
        target[4] -= stepMouvement # radian
    
    for i in range(len_joint_ids):

        if i in joint_ids_base_legs:
            x = min(max(target[i], clip_base_legs[0]) , clip_base_legs[1])
            cliped[i] = x

        elif i in joint_ids_first_legs:
            x = min(max(target[i], clip_first_legs[0]) , clip_first_legs[1])
            cliped[i] = x

        elif i in joint_ids_second_legs:
            x = min(max(target[i], clip_second_legs[0]) , clip_second_legs[1])
            cliped[i] = x

    #print(cliped)
    p.setJointMotorControlArray(spider,
                                joint_ids,
                                p.POSITION_CONTROL,
                                target)
    

    focus_position , _ = p.getBasePositionAndOrientation(spider)
        
    p.resetDebugVisualizerCamera(cameraDistance=4,
                                cameraYaw=0,
                                cameraPitch=-40,
                                cameraTargetPosition = focus_position)


    p.stepSimulation()
    time.sleep(1./240.)