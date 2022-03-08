import time
import os
import argparse
from multiprocessing import Process
import sys

from ray.rllib.env.policy_client import PolicyClient

SERVER_ADDRESS = os.environ.get('VC_LEARNER_HOSTS', "127.0.0.1")  # "127.0.0.1"
SERVER_PORT = 9999

parser = argparse.ArgumentParser()
parser.add_argument("--inference-mode", type=str, default="remote", choices=["local", "remote"])
parser.add_argument("--off-policy", type=str, default=False,
                    help="Whether to take random instead of on-policy actions.")
parser.add_argument('--data_url', type=str, default='/home/work/modelarts/inputs/data_url_0')
parser.add_argument('--train_url', type=str, default='/home/work/modelarts/outputs/train_url_0')
parser.add_argument("--num_envs", type=int, default=4)
FLAGS, unparsed = parser.parse_known_args()

sys.path.append(FLAGS.data_url)
print(os.listdir(FLAGS.data_url))
from external_env import create_env


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def env_boost(ind=0):
    env = create_env()
    not_ready_flag = True
    while not_ready_flag:
        try:
            client = PolicyClient("http://{}:{}".format(SERVER_ADDRESS, SERVER_PORT + ind),
                                  inference_mode=FLAGS.inference_mode)

            eid = client.start_episode(training_enabled=True)
        except ConnectionError:
            print("Server not ready...")
        else:
            not_ready_flag = False
    obs = env.reset()
    rewards = 0

    while True:
        if str2bool(FLAGS.off_policy):
            action = env.action_space.sample()
            client.log_action(eid, obs, action)
        else:
            st = time.time()
            action = client.get_action(eid, obs)
            print("get action: ", eid, action)
            print('proc_time: {}'.format(time.time() - st))
        obs, reward, done, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward, info=info)
        print("log returns: ", eid, reward)
        if done:
            print("Total reward:", rewards)
            rewards = 0
            client.end_episode(eid, obs)
            obs = env.reset()
            eid = client.start_episode(training_enabled=True)


if __name__ == "__main__":
    processes = []
    time.sleep(30)
    for i in range(FLAGS.num_envs):
        processes.append(Process(target=env_boost, args=(0,)))
        processes[i].start()
