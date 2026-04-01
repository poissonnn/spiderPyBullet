import matplotlib.pyplot as plt

red_light   = [220/255, 100/255, 100/255]
green_light = [100/255, 200/255, 100/255]
green_dark  = [80/255,  180/255, 80/255 ]
blue_light  = [100/255, 100/255, 220/255]
orange      = [241/255, 108/255, 52/255 ]
black       = [50/255,  51/255,  57/255 ]
true_black  = [0.1,0.1,0.1]
gray        = [150/255, 150/255, 150/255]
white       = [1, 1, 1]

Size = 14,7

def tensorGraphic(reward, episodeLength):
    print("graph")

    fig, axs = plt.subplots(2,1,figsize=(Size))

    axs[0].set_title("reward")
    axs[0].plot(reward, color = green_light)
    
    axs[1].set_title("episodeLength")
    axs[1].plot(episodeLength, color = red_light)

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
