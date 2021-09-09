import numpy as np
from gym.spaces import Discrete, Box

action_space = Discrete(4)
observation_space = Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
