import numpy as np
from gym.spaces import Discrete, Box

action_space = Discrete(2)
observation_space = Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
