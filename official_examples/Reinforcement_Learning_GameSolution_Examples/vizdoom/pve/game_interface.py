import copy
import os
import shutil
import time
import itertools as it

import numpy as np
from vizdoom import *


class GameInterface():
    def __init__(self, *args):
        # Create Doom instance
        print("Initializing doom...")
        config_file_path = os.path.join(os.path.dirname(__file__),
                                        "game_core/basic.cfg")
        game = DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(False)

        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(ScreenResolution.RES_320X240)
        game.set_available_game_variables([GameVariable.KILLCOUNT, GameVariable.HITCOUNT])
        game.set_mode(Mode.PLAYER)
        # may be need if error "Failed to create ./_vizdoom/ directory"
        while True:
            try:
                if os.path.exists("./_vizdoom/"):
                    shutil.rmtree("./_vizdoom/")
                game.init()
            except FileExistsError:
                time.sleep(0.01)
            else:
                break
        self.game = game
        self.resolution = (240, 320)

        n = game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]

        # Other parameters
        self.frame_repeat = 12
        print("Doom initialized.")

    def reset(self, *args):
        self.game.new_episode()
        obs = self.game.get_state().screen_buffer
        obs = obs.reshape([self.resolution[0], self.resolution[1], 1]).astype(np.float32)
        self.obs = obs
        return obs

    def step(self, action, *args):
        last_obs = copy.deepcopy(self.obs)
        reward = self.game.make_action(self.actions[action], self.frame_repeat)
        done = self.game.is_episode_finished()
        obs = self.game.get_state().screen_buffer if not done else last_obs
        obs = obs.reshape([self.resolution[0], self.resolution[1], 1]).astype(np.float32)
        info = {}
        self.reward = reward
        self.obs = obs
        return obs, reward, done, info

    def close(self, *args):
        self.game.close()

    def get_winner(self):
        return 1 if self.reward > 0 else 0


if __name__ == '__main__':
    game_env = GameInterface()
    game_env.reset()
    game_env.step(0)
