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
green_light = [100/255, 200/255, 100/255]
green_dark  = [80/255,  180/255, 80/255 ]
blue_light  = [100/255, 100/255, 220/255]
orange      = [241/255, 108/255, 52/255 ]
black       = [50/255,  51/255,  57/255 ]
true_black  = [0.1,0.1,0.1]
gray        = [150/255, 150/255, 150/255]
white       = [1, 1, 1]

Size = (14,7)

def tensorGraphic(reward, episodeLength):
    print("graph")

    fig, axs = plt.subplots(2,1,figsize=(Size))

    axs[0].set_title("reward")
    axs[0].plot(reward, color = green_light)
    
    axs[1].set_title("episodeLength")
    axs[1].plot(episodeLength, color = red_light)

    plt.savefig("training.png")

"""
def third_Tensor_Graphic(reward, episodeLength, rewardSpe):
    print("graph")

    fig, axs = plt.subplots(2,2,figsize=(Size))

    axs[0][0].set_title("Reward")
    axs[0][0].plot(reward, color = green_light)

    axs[0][1].set_title("Reward on 1/4 step")
    axs[0][1].plot(rewardSpe, color = orange)
    
    axs[1][1].set_title("EpisodeLength")
    axs[1][1].plot(episodeLength, color = red_light)



    plt.savefig("training.png")
"""
def third_Tensor_Graphic(reward, episodeLength, rewardSpe):
    print("graph")

    mosaic = """
        AB
        DD
        """
    
    fig = plt.figure(constrained_layout=True)
    axs = fig.subplot_mosaic(mosaic)
    #identify_axes(axs)


    axs["A"].set_title("Reward")
    axs["A"].plot(reward, color = green_light)

    axs["B"].set_title("Reward on 1/4 step")
    axs["B"].plot(rewardSpe, color = orange)
    
    axs["D"].set_title("EpisodeLength")
    axs["D"].plot(episodeLength, color = red_light)



    plt.savefig("training.png")


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
