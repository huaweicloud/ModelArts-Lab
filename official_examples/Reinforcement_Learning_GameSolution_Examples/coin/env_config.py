from gym.spaces import Discrete, Tuple, MultiBinary

action_space = {
    0: Discrete(2),
    1: Discrete(3)}
observation_space = {
    0: Discrete(4),
    1: Tuple((Discrete(4), MultiBinary(3)))}
