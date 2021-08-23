import os
import numpy as np
import time
import pickle
import traceback
import logging
from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import ExternalEnv
import sys
import ray

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from rl_config import ACTION_SPACE, OBSERVATION_SPACE, CONFIG_PPO, CONFIG_DQN


class ExternalAtari(ExternalEnv):
    def __init__(self, config):
        ExternalEnv.__init__(self, action_space=ACTION_SPACE, observation_space=OBSERVATION_SPACE)

    def run(self):
        print('start to run fake eternal env...')
        time.sleep(999999)


def build_bot():
    ray.init(local_mode=True)
    trainer = PPOTrainer(env=ExternalAtari, config=dict(**CONFIG_PPO))
    model_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ckpts')
    last_iter = 0
    for name in os.listdir(model_dir):
        print(name)
        it = int(name.split('_')[1])
        if it > last_iter:
            last_iter = it
    print(os.listdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ckpts/checkpoint_{}'.format(last_iter))))
    trainer.restore(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'ckpts/checkpoint_{}/checkpoint-{}'.format(last_iter, last_iter)))
    return trainer


bot = build_bot()


class PolicyServer(ThreadingMixIn, HTTPServer):
    def __init__(self, address, port):
        handler = _make_handler()
        HTTPServer.__init__(self, (address, port), handler)


def _make_handler():
    class Handler(SimpleHTTPRequestHandler):

        def do_POST(self):
            content_len = int(self.headers.get("Content-Length"), 0)
            raw_body = self.rfile.read(content_len)
            parsed_input = pickle.loads(raw_body)
            obs = np.array(parsed_input['obs'])

            try:
                act = bot.compute_action(obs, explore=True)
                response = {'action': act}
                self.send_response_only(200)
                self.end_headers()
                self.wfile.write(pickle.dumps(response, protocol=0))
            except Exception:
                logging.info("error")
                self.send_error(500, traceback.format_exc())

    return Handler


if __name__ == "__main__":
    server = PolicyServer(address="0.0.0.0", port=8080)
    print("server start...")
    server.serve_forever()
