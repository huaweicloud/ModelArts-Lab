import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from gym_spoof.envs.spoof_env import SpoofEnv


class GameInterface():
    def __init__(self, *args):
        # Create Doom instance
        self.game = SpoofEnv()
        self.player_num = 2
        # must contain current turn
        self.current_turn = 0
        self.action_0 = [0, 0, 0]

    def reset(self, *args):
        p1_obs, p2_obs = self.game.reset(initial_state=[True, False, True], shuffle_initial=True)
        self.current_turn = 0
        self.action_0 = [0, 0, 0]
        self.game.render()
        return p1_obs

    def step(self, action, *args):
        # record player 0 action
        if self.current_turn == 0:
            self.action_0 = [0, 0, 0]
            self.action_0[action] = 1
            obs, reward, done, info = self.game.step((self.action_0, None))
            obs = (obs[0], 1 * np.array(obs[1]))
            reward = reward[1]
        else:
            obs, reward, done, info = self.game.step((self.action_0, action))
            obs = obs[0]
            reward = reward[0]
        self.game.render()
        if done:
            print("winner is ", self.current_turn if reward == -1 else 1 - self.current_turn)
        self.current_turn = 1 - self.current_turn
        return obs, reward, done, info


if __name__ == '__main__':
    import random

    game_env = GameInterface()
    done = False
    game_env.reset()
    while not done:
        action = random.choice([0, 1]) if game_env.current_turn else random.choice([0, 1, 2])
        obs, reward, done, info = game_env.step(action)
