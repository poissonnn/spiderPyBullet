import matplotlib.pyplot as plt

def tensorGraphic (graphReward, episodeLength):
    print("hello")

    red_light   = [220/255, 100/255, 100/255]
    green_light = [100/255, 200/255, 100/255]
    green_dark  = [80/255,  180/255, 80/255 ]
    blue_light  = [100/255, 100/255, 220/255]
    orange      = [241/255, 108/255, 52/255 ]
    black       = [50/255,  51/255,  57/255 ]
    gray        = [150/255, 150/255, 150/255]
    white       = [1, 1, 1]
    
    numberGraph = 2

    fig, axs = plt.subplots(numberGraph, 1)

    axs[0].plot(graphReward,
            color = 'red')
    axs[0].set_title("Reward")

    axs[1].plot(episodeLength,
            color = 'red')

    axs[1].set_title("Episode Length")



    plt.tight_layout()
    plt.savefig("training.png")
