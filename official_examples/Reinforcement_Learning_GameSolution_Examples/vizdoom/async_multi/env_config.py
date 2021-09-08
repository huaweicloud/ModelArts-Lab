import numpy as np
from gym.spaces import Discrete,Box,Dict,MultiBinary


action_space=Discrete(3)
observation_space=Dict({
    "obs":Box(-np.inf,np.inf,shape=(6,9,3),dtype=np.float32),
    "valid_action":MultiBinary(3)
})

