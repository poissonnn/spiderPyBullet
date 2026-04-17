import matplotlib.pyplot as plt
import numpy as np
import pickle

red_dark    = [97 /255, 0  /255, 6  /255]
red_light   = [220/255, 100/255, 100/255]
orange_dark = [214/255, 40 /255, 40 /255]
orange      = [255/255, 127/255, 17 /255]
peach       = [255/255, 176/255, 144/255]
yellow      = [252/255, 191/255, 73 /255]

green_dark  = [80 /255, 180/255, 80 /255]
green_light = [100/255, 200/255, 100/255]

teal_dark   = [40 /255, 90 /255, 72 /255]
teal_medium = [64 /255, 138/255, 113/255]
teal_light  = [176/255, 228/255, 204/255]

blue_dark   = [56 /255, 82 /255, 180/255]
blue_light  = [100/255, 100/255, 220/255]

purple      = [93 /255, 28 /255, 106/255]
pink        = [202/255, 89 /255, 149/255]

black       = [50 /255, 51 /255, 57 /255]
true_black  = [0.1,0.1,0.1]
gray        = [150/255, 150/255, 150/255]
white       = [1, 1, 1]

size = (14,12)

graphPath = r"Graph/"

def Reward(graphRewards):
    print("reward graph")
    
    alpha = 1
    zorder = 2
    linewidth = 3.5

    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Reward")

    for graphRewards in allGraphRewards:
        
        alpha -= 0.20 
        linewidth -= 0.5
        zorder += 1
        print(zorder)

        ax.plot(graphRewards, color = black, zorder = 3, alpha = alpha,linewidth = linewidth)

        #ax.fill_between(range(len(graphRewards)), np.array(graphRewards), where= np.array(graphRewards) > 0, color = yellow, alpha = alpha/3, interpolate = True,zorder = 1)
        #ax.fill_between(range(len(graphRewards)), np.array(graphRewards), where= np.array(graphRewards) < 0, color = orange_dark  , alpha = alpha/3, interpolate = True, zorder = 1)

    ax.axhline(0, color = true_black, linestyle = (0,(2,2.5)), linewidth = 1.5, alpha = alpha)
    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Graph/Reward.png")
    plt.close()


linesCount = 0

with open("save.txt") as SaveFile:
    
    lines = SaveFile.readlines()

counter = len(lines)
#print(counter)



# -- read all the data from the save file --
num_epochsIndicies          = -1
graphRewardsIndices         = -2
episodeLengthIndices        = -3
numpyGraphPolicyLossIndices = -4
numpyGraphValueLossIndices  = -5
graphEpisodeRewardIndices   = -6

allGraphRewards = []

batch = counter//6

batchSize = counter//batch
print(batchSize)
print()

for i in range(batch):
    num_epochs = float(lines[num_epochsIndicies-batchSize*i].strip())

    graphEpisodeReward = lines[graphRewardsIndices-batchSize*i].strip() 
    graphEpisodeReward = [float(i) for i in graphEpisodeReward[1:-1].split(", ")] 


    numpyGraphValueLoss = lines[episodeLengthIndices-batchSize*i].strip()
    numpyGraphValueLoss = [float(i) for i in numpyGraphValueLoss[1:-1].split(", ")] 

    numpyGraphPolicyLoss = lines[numpyGraphPolicyLossIndices-batchSize*i].strip()
    numpyGraphPolicyLoss = [float(i) for i in numpyGraphPolicyLoss[1:-1].split(", ")]

    episodeLength = lines[numpyGraphValueLossIndices-batchSize*i].strip()
    episodeLength = [float(i) for i in episodeLength[1:-1].split(", ")] 

    graphRewards = lines[graphEpisodeRewardIndices-batchSize*i].strip()
    graphRewards = [float(i) for i in graphRewards[1:-1].split(", ")] # .split(", ") supp the useless part

    allGraphRewards.append(graphRewards)
    print("a")


Reward(allGraphRewards)

