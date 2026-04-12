import matplotlib.pyplot as plt
import numpy as np

graphRewards = [0,1,1.5,5,8,9,10,6,5,4,-5,-18,-9,-52]

cumulative = []
total = 0

for i in graphRewards:
    total += i
    cumulative.append(total)



fig, ax = plt.subplots()

ax.plot(cumulative, color='black')
ax.axhline(0, color='black')

ax.fill_between(range(len(cumulative)), np.array(cumulative), where= np.array(cumulative) > 0, facecolor='green', alpha=.5, interpolate=True)
ax.fill_between(range(len(cumulative)), np.array(cumulative), where= np.array(cumulative) < 0, facecolor='red'  , alpha=.5, interpolate=True)

plt.show()