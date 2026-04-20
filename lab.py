from gymnasium.spaces import MultiDiscrete
import numpy as np

observation_space = MultiDiscrete(np.array([3] * 12))
for i in range(10):

    print(observation_space.sample())
