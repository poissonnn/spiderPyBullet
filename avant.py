reward = 0

firstDistanceGoal  = 20
checkpoint         = 0
distance           = 20



distance80Percent = firstDistanceGoal * (80/100)
distance65Percent = firstDistanceGoal * (65/100)
distance50Percent = firstDistanceGoal * (50/100)
distance35Percent = firstDistanceGoal * (35/100)


print(f"distance80 : {distance80Percent}")
print(f"distance65 : {distance65Percent}")
print(f"distance50 : {distance50Percent}")
print(f"distance35 : {distance35Percent}")


print(f"distance : {distance}")

print()

for i in range(40):

    if distance < distance80Percent and checkpoint == 0:
        checkpoint = 1
        reward += 5 
        
        print("checkpoint 1")

    if distance < distance65Percent and checkpoint == 1:
        checkpoint = 2
        reward += 10 

        print("checkpoint 2")

    if distance < distance50Percent and checkpoint == 2:
        checkpoint = 3
        reward += 15 
        print("checkpoint 3")

    if distance < distance35Percent and checkpoint == 3:
        checkpoint = 4
        reward += 20
        print("checkpoint 4")


    print(f"distance : {distance}")
    distance -= 0.5
    


print(f"reward : {reward}")