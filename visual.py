import matplotlib.pyplot as plt
import numpy as np

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

def tensorGraphic(reward, episodeLength):
    print("graph")

    fig, axs = plt.subplots(2,1,figsize=(Size))

    axs[0].set_title("reward")
    axs[0].plot(reward, color = green_light)
    
    axs[1].set_title("episodeLength")
    axs[1].plot(episodeLength, color = red_light)

    plt.savefig("training.png")


graphRewardMean        = []
specificGraphReward10  = [] 
specificGraphReward4   = [] 
specificGraphReward2   = [] 
specificGraphReward1_3 = []
specificGraphReward1_1 = []
XGraphRewardMean       = []

def third_Tensor_Graphic(reward,
                        episodeLength,
                        episodes_rewards):



    print("graph")


    mosaic = """
        AB
        DD
        """
    

    for i in range(len(episodes_rewards)):
        graphRewardMean.append(float(np.mean(episodes_rewards[i])))

        specificGraphReward10.append(episodes_rewards[i][len(episodes_rewards[i])//10]) 
        specificGraphReward4.append(episodes_rewards[i][len(episodes_rewards[i])//7]) 
        specificGraphReward2.append(episodes_rewards[i][len(episodes_rewards[i])//2]) 
        specificGraphReward1_3.append(episodes_rewards[i][int(len(episodes_rewards[i])//1.333)])  
        specificGraphReward1_1.append(episodes_rewards[i][int(len(episodes_rewards[i])//1.111)])  




    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic(mosaic)
    #identify_axes(axs)


    axs["D"].set_title("Reward Specific")
    #axs["A"].plot(specificGraphReward10,  color = [100/255, 200/255, 100/255], alpha = 0.70, linestyle = (0,(2,2.5)))
    axs["D"].plot(specificGraphReward4,   color = teal_medium , alpha = 0.75,)
    axs["D"].plot(specificGraphReward2,   color = teal_dark, alpha = 0.80, )
    axs["D"].plot(specificGraphReward1_3, color = teal_black  , alpha = 0.85, )
    #axs["A"].plot(specificGraphReward1_1, color = [100/255, 200/255, 80/255 ], alpha = 0.90, linestyle = (0,(2,2.5)))

    # superposer différent reward a différent moment

    #halfEpisodeLength = np.array(episodeLength) / 2


    totalLength = 0
    for Length in episodeLength:
        totalLength += Length
        XGraphRewardMean.append(totalLength)


    axs["B"].set_title("Reward")
    axs["B"].plot(reward, color = orange, zorder = 1)
    
    axs["B"].plot(XGraphRewardMean, graphRewardMean, color = red_dark, zorder = 2)
    
    axs["A"].set_title("EpisodeLength")
    axs["A"].plot(episodeLength, color = red_light)

    axs["A"].grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)
    axs["B"].grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)
    axs["D"].grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

   

    plt.savefig("training.png")


def cumulative_reward(rewards):
    cumulative = []
    total = 0
    for r in rewards:
        total += r
        cumulative.append(total)

a = []
x = []

def PPO_Graphic(X):

    y = []
    fig, ax = plt.subplots(1,1,figsize=(14,7))

    a.append(X)

    min_a = min(a)
    max_a = max(a)+1

    x = range(min_a,max_a+1)

    for i in range(min_a,max_a+1):
        y.append(a.count(i))
        print(y)

    plt.bar(x,y)

    plt.savefig("training.png")
