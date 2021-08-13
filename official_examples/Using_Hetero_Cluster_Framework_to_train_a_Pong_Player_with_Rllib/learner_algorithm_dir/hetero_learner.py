from multiprocessing import Process
import argparse
import ray
import os
import time
import sys

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.env import ExternalEnv
from ray.tune.logger import pretty_print

import moxing as mox

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=1, help='GPU')
parser.add_argument('--data_url', type=str, default='')
parser.add_argument('--train_url', type=str)
parser.add_argument('--iter', type=int, default=50)
FLAGS, unparsed = parser.parse_known_args()

sys.path.append(FLAGS.data_url)
print(os.listdir(FLAGS.data_url))
from rl_config import ACTION_SPACE, OBSERVATION_SPACE, CONFIG_PPO, CONFIG_DQN

ADDRESS = os.environ.get('VC_LEARNER_HOSTS', '127.0.0.1')
PORT = 9999


class ExternalAtari(ExternalEnv):
    def __init__(self, config):
        ExternalEnv.__init__(self, action_space=ACTION_SPACE, observation_space=OBSERVATION_SPACE)

    def run(self):
        print('start to run fake eternal env...')
        time.sleep(999999)


def ray_server(run='PPO', address=ADDRESS, port=PORT):
    print(ray.init(log_to_driver=False))

    connector_config = {
        "input": (
            lambda ioctx: PolicyServerInput(ioctx, address, port)
        ),
        "num_workers": 0,
        "input_evaluation": [],
        "create_env_on_driver": False,
        "num_gpus": FLAGS.num_gpus,
    }

    if run == "DQN":
        trainer = DQNTrainer(
            env=ExternalAtari,
            config=dict(
                connector_config, **CONFIG_DQN))
    elif run == "PPO":
        trainer = PPOTrainer(
            env=ExternalAtari,
            config=dict(
                connector_config, **CONFIG_PPO))
    else:
        raise ValueError("--run must be DQN or PPO")

    i = 0
    while i < FLAGS.iter:
        i += 1
        print(pretty_print(trainer.train()))
    ray.shutdown()

    checkpoint = trainer.save("{}/ckpts".format(FLAGS.train_url.rstrip('/')))
    print("checkpoint saved at", checkpoint)

    del trainer


def upload_event(src, dest):
    while True:
        time.sleep(30)
        if os.path.exists(src):
            mox.file.copy_parallel(src, dest)


if __name__ == '__main__':
    run = 'PPO'
    tb_path = '{}/ray_results'.format('/'.join(os.environ.get('PWD', ' /home/work/user-job-dir').split('/')[0:3]))
    print('ray results stored in: ', tb_path)
    upload_proc = Process(target=upload_event,
                          args=(tb_path, FLAGS.train_url,))
    upload_proc.start()
    ray_server(run)
    upload_proc.terminate()
