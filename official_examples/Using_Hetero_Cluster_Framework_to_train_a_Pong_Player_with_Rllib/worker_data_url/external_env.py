import os
import subprocess


def system_cmd(cmd_list):
    cmd = subprocess.run(cmd_list, shell=False)
    if cmd.returncode != 0:
        print('cannot exec cmd: {}, exit with {}'.format(cmd_list, cmd.returncode))
        raise EnvironmentError
    return


def install_gym(mode=None):
    system_cmd(['pip', 'uninstall', '-y', 'enum34'])
    system_cmd(['pip', 'install', 'gym[atari]'])
    if mode is not None:
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'atari_roms')
        p_path = os.environ.get('SITE_PACKAGES_PATH', '/home/ma-user/anaconda/lib/python3.6/site-packages')
        target = os.path.join(p_path, 'atari_py/')
        system_cmd(['cp', '-rf', file_path, target])
    return


install_gym(mode='atari')


import gym
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind


def create_env(name='PongNoFrameskip-v4'):
    env = gym.make(name)
    env = wrap_deepmind(
        env,
        dim=42,
        framestack=False,
        framestack_via_traj_view_api=False)
    return env
