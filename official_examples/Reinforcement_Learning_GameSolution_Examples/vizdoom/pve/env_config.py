import numpy as np
from gym.spaces import Discrete, Box

action_space = Discrete(8)
observation_space = Box(-np.inf, np.inf, shape=(240, 320, 1), dtype=np.float32)
