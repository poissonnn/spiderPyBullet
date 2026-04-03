import numpy as np
import matplotlib.pyplot as plt

Size = (14,7)
Reward = []
NoReward = []
a = 1.5
b = 4

maxSize = 100
Step = 0.1

red_light   = [220/255, 100/255, 100/255]
green_light = [100/255, 200/255, 100/255]
green_dark  = [80/255,  180/255, 80/255 ]
blue_light  = [100/255, 100/255, 220/255]
orange      = [241/255, 108/255, 52/255 ]
black       = [50/255,  51/255,  57/255 ]
true_black  = [0.1,0.1,0.1]
gray        = [150/255, 150/255, 150/255]
white       = [1, 1, 1]

for i in np.arange(0, maxSize, Step):
    x = a * np.sin(i/b)
    Reward.append(x)
    NoReward.append(-x)



print(len(Reward))

fig, axs = plt.subplots(1,1,figsize=Size)

axs.set_xlim(0,maxSize*(1/Step))
axs.set_ylim(-4,4)

axs.plot(Reward, color = orange)
axs.plot(NoReward, color = orange)
plt.show()