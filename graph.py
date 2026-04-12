import matplotlib.pyplot as plt
import numpy as np


red_light   = [220/255, 100/255, 100/255]
red_dark    = [97 /255, 0  /255, 6  /255]
green_light = [100/255, 200/255, 100/255]
green_dark  = [80 /255, 180/255, 80 /255]
blue_light  = [100/255, 100/255, 220/255]
purple      = [93 /255, 28 /255, 106/255]
peach       = [255/255, 176/255, 144/255]
orange      = [255/255, 127/255, 17 /255]
yellow      = [252/255, 191/255, 73 /255]
orange_dark = [214/255, 40 /255, 40 /255]
black       = [50 /255, 51 /255, 57 /255]
true_black  = [0.1,0.1,0.1]
gray        = [150/255, 150/255, 150/255]
white       = [1, 1, 1]

teal_light  = [176/255, 228/255, 204/255]
teal_medium = [64 /255, 138/255, 113/255]
teal_dark   = [40 /255, 90 /255, 72 /255]

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
    cumulative = []
    total = 0

    for i in graphRewards:
        total += i
        cumulative.append(total)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Cumulative reward")

    ax.axhline(0, color = true_black, linestyle = (0,(2,2.5)), linewidth = 1.5)

    ax.plot(cumulative, color = black, zorder = 2)
    
    ax.fill_between(range(len(cumulative)), np.array(cumulative), where= np.array(cumulative) > 0, color = green_light, alpha = 0.25, interpolate = True)
    ax.fill_between(range(len(cumulative)), np.array(cumulative), where= np.array(cumulative) < 0, color = red_light  , alpha = 0.25, interpolate = True)

    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)
    

    plt.tight_layout()
    plt.savefig("cumulative_reward.png")
    plt.close()


def Reward(graphRewards):
    print("reward graph")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Reward")

    ax.plot(graphRewards, color = orange_dark, zorder = 1)

    ax.axhline(0, color = true_black, linestyle = (0,(2,2.5)), linewidth = 1.5)
    ax.fill_between(range(len(graphRewards)), np.array(graphRewards), where= np.array(graphRewards) > 0, color = yellow, alpha = 0.25, interpolate = True)
    ax.fill_between(range(len(graphRewards)), np.array(graphRewards), where= np.array(graphRewards) < 0, color = orange_dark  , alpha = 0.25, interpolate = True)

    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("Reward.png")
    plt.close()




graphRewards = [1,2,6,5,8,7,40,80,90,80,-70,-10,-10,12,55,-154,-785]
Reward(graphRewards)
cumulative_reward(graphRewards)

def episode_length(episodeLength):
    print("episode length graph")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title("Episode length")


    ax.plot(episodeLength, color = purple, zorder = 1)
    ax.axhline(0, color = purple, linestyle = (0,(2,2.5)), linewidth = 1.5, zorder = 2)

    ax.grid(which = "major", alpha = 0.25, linestyle = "--", linewidth = 0.8,color = gray, zorder = 0)

    plt.tight_layout()
    plt.savefig("episodeLength.png")
    plt.close()


def policy_loss():
    print("policy loss graph")

def value_loss():
    print("value loss graph")