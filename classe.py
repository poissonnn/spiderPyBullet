import numpy

position = 0.0
stepMouvement = 0.2

joint_ids_base_legs =   [1, 4, 7, 10] # -90 // 90 | -1.5708 // 1.5708
joint_ids_first_legs =  [2, 5, 8, 11] # -45 // 70 | -0.7853 // 1.2217
joint_ids_second_legs = [3, 6, 9, 12] # -65 // 90 | -1.1344 // 1.5708

clip_base_legs =   [-1.5708 , 1.5708]
clip_first_legs =  [-0.7853 , 1.2217]
clip_second_legs = [-1.1344 , 1.5708]

print(clip_base_legs[0] , clip_base_legs [1])
print(clip_first_legs[0] , clip_first_legs [1])
print(clip_second_legs[0] , clip_second_legs [1])

joint_ids = joint_ids_base_legs + joint_ids_first_legs + joint_ids_second_legs
print(joint_ids)
len_joint_ids = len(joint_ids)
print(len_joint_ids)
target = [position] * len_joint_ids
#cliped = [position for  i in range(len_joint_ids)]
cliped = [position] * len_joint_ids
print(target)
print(cliped)

for i in range(10):
    for j in range(len_joint_ids):
        target[j] = 2 


for i in range(len_joint_ids):

    if i in joint_ids_base_legs:
        print(f"{i} : {target[i]} : 1")
        cliped[i-1] = numpy.clip(target[i] , clip_base_legs[0] , clip_base_legs[1] )
        print(f"{i} : {cliped[i-1]} : 1.5")

    if i in joint_ids_first_legs:
        print(f"{i} : {target[i]} : 2")
        cliped[i-1] = numpy.clip(target[i] , clip_first_legs[0] , clip_first_legs[1] )
        print(f"{i} : {cliped[i]} : 2.5")

    if i in joint_ids_second_legs:
        print(f"{i} : {target[i]} : 3")
        cliped[i-1] = numpy.clip(target[i] , clip_second_legs[0] , clip_second_legs[1] )
        print(f"{i} : {cliped[i]} : 3.5")

    #target[i]
print(cliped)