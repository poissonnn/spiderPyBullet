import matplotlib.pyplot as plt
import numpy as np
import random


red_light   = [220/255, 100/255, 100/255]
red_dark    = [97 /255, 0  /255, 6  /255]
green_light = [100/255, 200/255, 100/255]
green_dark  = [80 /255, 180/255, 80 /255]
blue_light  = [100/255, 100/255, 220/255]
orange      = [232/255, 127/255, 36 /255]
black       = [50 /255, 51 /255, 57 /255]
true_black  = [0.1,0.1,0.1]
gray        = [150/255, 150/255, 150/255]
white       = [1, 1, 1]

teal_light  = [176/255, 228/255, 204/255]
teal_medium = [64 /255, 138/255, 113/255]
teal_dark   = [40 /255, 90 /255, 72 /255]
teal_black  = [9  /255, 20 /255, 19 /255]

Size = (20,20)

fig, axs = plt.subplots(1,1,figsize=Size)

SizeGraph = 2.5
N = 1000

size = 2
border = 1



x = []
y = []
number = 0
while number == 0:
    print("random")
    temporaryX = random.uniform(-2, 2)
    temporaryY = random.uniform(-2, 2)

    if temporaryY < -border or temporaryY > border or temporaryX < -border or temporaryX > border:
        print("graph")
        x.append(temporaryX)
        y.append(temporaryY)
        number += 1






axs.set_xlim(-SizeGraph, SizeGraph)
axs.set_ylim(-SizeGraph, SizeGraph)

axs.scatter(x ,y ,s = 50.0, color = red_light)

axs.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)



plt.show()