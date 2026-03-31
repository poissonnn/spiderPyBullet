size = 10

def compute_reward(distance1,distance2):
    reward = 0
    direction = 0
    distance = 1
    done = False
    #reward = (size - distance2) / size

    delta = (distance1 - distance2) 
    invertDistance = size - distance2


    # trop de bruit besoin de courbe plus douce

    if delta > 0:
        direction = 1
    else:
        direction = -0

    distance = invertDistance * direction

    if distance2 < 1.4:
        reward = reward + 5
        done = True

    reward = reward + distance

    return reward, done

reward = compute_reward(6,5)
print(reward)
reward = compute_reward(5,4)
print(reward)
reward = compute_reward(4,4)
print(reward)
reward = compute_reward(4,5)
print(reward)