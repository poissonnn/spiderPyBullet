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

# Helper function used for visualization in the following examples
def identify_axes(ax_dict, fontsize=48):
    """
    Helper to identify the Axes in the examples below.

    Draws the label in a large font in the center of the Axes.

    Parameters
    ----------
    ax_dict : dict[str, Axes]
        Mapping between the title / label and the Axes.
    fontsize : int, optional
        How big the label should be.
    """
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

def cumulative_reward(graphRewards):
    print("cumulative reward graph")


    alpha = 1
    zorder = 20
    linewidth = 3.0

    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Cumulative reward")

    for graphRewards in allGraphRewards:
        total = 0
        cumulative = []
        
        for i in graphRewards:
            total += i
            cumulative.append(total)

        ax.plot(cumulative, color = black, zorder = zorder, alpha = alpha,linewidth = linewidth)

        ax.fill_between(range(len(cumulative)), np.array(cumulative), where= np.array(cumulative) > 0, color = green_light, alpha = alpha/4, interpolate = True,zorder = zorder-3)
        ax.fill_between(range(len(cumulative)), np.array(cumulative), where= np.array(cumulative) < 0, color = orange_dark  , alpha = alpha/4, interpolate = True, zorder = zorder-3)
       
        alpha -= 0.10 
        linewidth -= 0.5
        zorder += 1

    ax.axhline(0, color = true_black, linestyle = (0,(2,2.5)), linewidth = 1.5)
    
    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)
    
    plt.tight_layout()
    plt.savefig("Graph/cumulative_reward.png")
    plt.close()

def Reward(allGraphRewards):
    print("reward graph")
    
    alpha = 1
    zorder = 20
    linewidth = 3.5

    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Reward")

    for graphRewards in allGraphRewards:
        


        ax.plot(graphRewards, color = black, zorder = zorder, alpha = alpha,linewidth = linewidth)

        alpha -= 0.20 
        linewidth -= 0.5
        zorder += 1

        #ax.fill_between(range(len(graphRewards)), np.array(graphRewards), where= np.array(graphRewards) > 0, color = yellow, alpha = alpha/4, interpolate = True,zorder = zorder-3)
        #ax.fill_between(range(len(graphRewards)), np.array(graphRewards), where= np.array(graphRewards) < 0, color = orange_dark  , alpha = alpha/4, interpolate = True, zorder = zorder-3)

    ax.axhline(0, color = true_black, linestyle = (0,(2,2.5)), linewidth = 1.5, alpha = alpha)
    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Graph/Reward.png")
    plt.close()

def episode_length(allEpisodeLength):
    print("episode length graph")

    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Episode length")

    alpha = 1
    zorder = 20
    linewidth = 5.5
    i = 0

    for episodeLength in allEpisodeLength:
        episodeLengthMean = np.mean(episodeLength)
        
        if i == 0:
            ax.axhline(episodeLengthMean, color = purple, linestyle = (0,(5,1)), linewidth = 2, zorder = zorder+1)
            i = 1

        ax.plot(episodeLength, color = peach, zorder = zorder,alpha = alpha, linewidth = linewidth)
        #ax.axhline(episodeLengthMean, color = pink, linestyle = (0,(5,1)), linewidth = 1, zorder = 2)

        alpha -= 0.20 
        linewidth -= 0.8
        zorder += 1

    #ax.plot(episodeLength, color = peach, zorder = 1)
    #ax.axhline(episodeLengthMean, color = pink, linestyle = (0,(5,1)), linewidth = 1, zorder = 2)
    #ax.axhline(episodeLengthMean, color = purple, linestyle = (0,(5,1)), linewidth = 2, zorder = 3)

    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Graph/episodeLength.png")
    plt.close()

def policy_loss(PolicyLoss,num_epochs):
    print("policy loss graph")

    batch = len(PolicyLoss)//num_epochs

    PolicyLossBatchMean = []
    PolicyLossCopy = PolicyLoss.copy()

    batch = int(batch)
    num_epochs =  int(num_epochs)
    

    for i in range(batch):
        
        PolicyLossBatchMean.append(np.mean(PolicyLossCopy[:num_epochs]))
        del PolicyLossCopy[0:num_epochs]


    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Policy loss")

    PolicyLossMean = np.mean(PolicyLoss)

    ax.plot(PolicyLossBatchMean, color = blue_light, linewidth = 2, zorder = 1)
    ax.axhline(PolicyLossMean, color = blue_dark, linestyle = (0,(5,1)), linewidth = 2, zorder = 3)

    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Graph/policyLoss.png")
    plt.close()
    
def value_loss(ValueLoss,num_epochs):
    print("value loss graph")

    #ValueLoss = ValueLoss.tolist()

    batch = len(ValueLoss)//num_epochs

    ValueLossBatchMean = []
    ValueLossCopy = ValueLoss.copy()

    batch = int(batch)
    num_epochs =  int(num_epochs)

    for i in range(batch):

        ValueLossBatchMean.append(np.mean(ValueLossCopy[:num_epochs]))
        del ValueLossCopy[0:num_epochs]


    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Value loss")

    ValueLossMean = np.mean(ValueLoss)

    ax.plot(ValueLossBatchMean, color = blue_light, linewidth = 2, zorder = 1)
    ax.axhline(ValueLossMean, color = blue_dark, linestyle = (0,(5,1)), linewidth = 2, zorder = 3)

    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Graph/ValueLoss.png")
    plt.close()

def episodeReward(allEpisodeReward,num_epochs):
    print("episodeReward graph")

    alpha = 1
    zorder = 20
    linewidth = 2.5

    fig, ax = plt.subplots(figsize=size)
    ax.set_title("Episode Reward")

    for episodeReward in allEpisodeReward:
        total = 0
        cumulative = []
        #episodeRewardMin  = min(episodeReward)
        #episodeRewardMax  = max(episodeReward)
        #episodeRewardMean = sum(episodeReward)/len(episodeReward)
        #print(len(allEpisodeReward))
        
        ax.plot(episodeReward, color = orange_dark, zorder = zorder, alpha = alpha,linewidth = linewidth)
        ax.fill_between(range(len(episodeReward)), np.array(episodeReward), where = np.array(episodeReward) > 0, color = yellow, alpha = alpha/4, interpolate = True, zorder = zorder-3)
        ax.fill_between(range(len(episodeReward)), np.array(episodeReward), where = np.array(episodeReward) < 0, color = orange_dark  , alpha = alpha/4, interpolate = True ,zorder = zorder-3)

        alpha -= 0.30 
        linewidth -= 1.5
        zorder += 1

    ax.axhline(0, color = true_black, linestyle = (0,(2,2.5)), linewidth = 1.5)
    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Graph/episodeReward.png")
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


allNum_epochs = []
allGraphEpisodeReward = []
allNumpyGraphValueLoss = []
allNumpyGraphPolicyLoss = []
allEpisodeLength = []
allGraphRewards = []

batch = counter//6

batchSize = counter//batch

for i in range(batch):
    num_epochs = float(lines[num_epochsIndicies-batchSize*i].strip())
    allNum_epochs.append(num_epochs)

    graphEpisodeReward = lines[graphRewardsIndices-batchSize*i].strip() 
    graphEpisodeReward = [float(i) for i in graphEpisodeReward[1:-1].split(", ")] 
    allGraphEpisodeReward.append(graphEpisodeReward)    


    numpyGraphValueLoss = lines[episodeLengthIndices-batchSize*i].strip()
    numpyGraphValueLoss = [float(i) for i in numpyGraphValueLoss[1:-1].split(", ")] 
    allNumpyGraphValueLoss.append(numpyGraphValueLoss)    

    numpyGraphPolicyLoss = lines[numpyGraphPolicyLossIndices-batchSize*i].strip()
    numpyGraphPolicyLoss = [float(i) for i in numpyGraphPolicyLoss[1:-1].split(", ")]
    allNumpyGraphPolicyLoss.append(numpyGraphPolicyLoss)

    episodeLength = lines[numpyGraphValueLossIndices-batchSize*i].strip()
    episodeLength = [float(i) for i in episodeLength[1:-1].split(", ")] 
    allEpisodeLength.append(episodeLength)

    graphRewards = lines[graphEpisodeRewardIndices-batchSize*i].strip()
    graphRewards = [float(i) for i in graphRewards[1:-1].split(", ")] # .split(", ") supp the useless part
    allGraphRewards.append(graphRewards)


#cumulative_reward(allGraphRewards)
#Reward(allGraphRewards)
#episode_length(allEpisodeLength)

#policy_loss(numpyGraphPolicyLoss, num_epochs)

#value_loss(numpyGraphValueLoss, num_epochs)

episodeReward(allGraphEpisodeReward, allNum_epochs)
