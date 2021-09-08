import os
import gym
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind


os.system('pip install gym[atari]')


def create_env(name='PongNoFrameskip-v4'):
    env = gym.make(name)
    env = wrap_deepmind(
        env,
        dim=42,
        framestack=False,
        framestack_via_traj_view_api=False)
    return env
