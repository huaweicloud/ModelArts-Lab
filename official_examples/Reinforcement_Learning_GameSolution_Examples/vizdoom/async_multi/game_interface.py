import os
import time
from multiprocessing import Process, Queue
from random import random
from time import sleep

import numpy as np
import skimage.color
import skimage.transform
from vizdoom import *


class GameInterface():
    def __init__(self, *args):
        # Create Doom instance
        print("Initializing doom...")
        self.frame_repeat = 1
        self.resolution = (6, 9, 3)
        self.config_file_path = os.path.join(os.path.dirname(__file__), "scenarios/cig.cfg")
        self.actions_list = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]

        self.player_id_dict = {}
        self.info = {}

        self.obs_queue = Queue()
        self.transfer_obs_queue = Queue()
        self.reward_queue = Queue()
        self.done_queue = Queue()
        self.winner_queue = Queue()
        self.action_queue = Queue()
        self.step_queue_dict = {}

        self.players = 3
        for i in range(self.players):
            self.player_id_dict[i] = i
            self.step_queue_dict[i] = Queue(maxsize=1)

        Process(target=self._start_game, args=()).start()

    def reset(self, *args):
        print("reset")
        sleep(random() * 0.005 + 0.001)
        obs_dict = {}
        transfer_obs_dict = {}
        for i in range(self.players):
            transfer_obs_dict.update({i: {}})

        while self.obs_queue.qsize() < self.players:
            pass
        while self.obs_queue.qsize() > 0:
            temp = self.obs_queue.get()
            obs_dict[list(temp.keys())[0]] = list(temp.values())[0]

            temp_trans = self.transfer_obs_queue.get()
            transfer_obs_dict[list(temp_trans.keys())[0]] = list(temp_trans.values())[0]

        self.obs_dict = obs_dict
        return obs_dict, transfer_obs_dict, self.player_id_dict

    def step(self, action, *args):
        print(action)
        for i in range(self.players):
            self.action_queue.put_nowait(action)
            if self.step_queue_dict[i].qsize():
                self.step_queue_dict[i].get()
            self.step_queue_dict[i].put_nowait(True)
        obs_dict = {}
        transfer_obs_dict = {}
        reward_dict = {}
        done_dict = {}
        done = True
        while self.obs_queue.qsize() < self.players:
            pass
        while self.obs_queue.qsize() > 0:
            temp_obs = self.obs_queue.get()
            obs_dict[list(temp_obs.keys())[0]] = list(temp_obs.values())[0]

            temp_trans = self.transfer_obs_queue.get()
            transfer_obs_dict[list(temp_trans.keys())[0]] = list(temp_trans.values())[0]

            temp_r = self.reward_queue.get()
            reward_dict[list(temp_r.keys())[0]] = list(temp_r.values())[0]

            temp_done = self.done_queue.get()
            done_dict[list(temp_done.keys())[0]] = list(temp_done.values())[0]

        for d_k, d_v in done_dict.items():
            if not d_v:
                done = False

        return obs_dict, transfer_obs_dict, self.player_id_dict, reward_dict, done, self.info

    def close(self, *args):
        print("close")

    def get_winner(self):
        winner_dict = {}
        while self.winner_queue.qsize() < self.players:
            pass
        while self.winner_queue.qsize() > 0:
            temp_win = self.winner_queue.get()
            winner_dict[list(temp_win.keys())[0]] = list(temp_win.values())[0]

        return winner_dict

    def get_obs_arr(self):
        return self.obs_dict

    def _setup_player(self):
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_mode(Mode.PLAYER)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_console_enabled(False)
        game.set_window_visible(False)
        return game

    def _play(self, index):
        game_i = self._setup_player()
        if index == 0:
            game_i.add_game_args(
                "-host " + str(self.players) + " -port 12123 -netmode 0 -deathmatch +timelimit " + str(1) +
                " +sv_norespawn 0 +name Player0 +colorset 0")
        else:
            game_i.add_game_args("-join 127.0.0.1:12123 +name Player" + str(index) + " +colorset " + str(index))
        game_i.init()

        while True:
            print("player {} begin a new episode".format(index))
            while not game_i.is_episode_finished():
                # sleep(random() * 0.005 + 0.001)
                obs = self._preprocess(game_i.get_state().screen_buffer)
                self.transfer_obs_queue.put_nowait(
                    {index: {"obs": obs, "valid_action": np.array([1, 1, 1])}})
                self.obs_queue.put_nowait({index: obs})
                while True:
                    if self.step_queue_dict[index].qsize() > 0 and self.step_queue_dict[index].get(timeout=1):
                        break
                if self.step_queue_dict[index].qsize() > 0:
                    self.step_queue_dict[index].get_nowait()
                self.step_queue_dict[index].put_nowait(False)
                a_dict = self.action_queue.get()
                reward = game_i.make_action(self.actions_list[a_dict[index]], self.frame_repeat)
                self.reward_queue.put({index: reward})
                done = game_i.is_episode_finished()
                self.done_queue.put({index: done})
                if done:
                    self.transfer_obs_queue.put_nowait(
                        {index: {"obs": obs, "valid_action": np.array([1, 1, 1])}})
                    self.obs_queue.put_nowait({index: obs})

            time.sleep(1)
            game_i.new_episode()
            print("player {} end a episode".format(index))

    def _start_game(self):
        for i in range(self.players):
            Process(target=self._play, args=(i,)).start()

        print("Doom initialized.")

    def _preprocess(self, img):
        img = skimage.transform.resize(img.transpose(2, 0, 1), self.resolution)
        img = img.astype(np.float32)
        return img


if __name__ == '__main__':
    game_env = GameInterface()
    game_env.reset()
    game_env.step({0: 1, 1: 2, 2: 0})
