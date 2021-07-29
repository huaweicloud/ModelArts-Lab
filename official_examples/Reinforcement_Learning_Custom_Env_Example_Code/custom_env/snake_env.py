from collections import deque

import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding


class SnakeAction(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


class BoardColor(object):
    BODY_COLOR = np.array([0, 0, 0], dtype=np.uint8)
    FOOD_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    SPACE_COLOR = np.array([255, 255, 255], dtype=np.uint8)


class SnakeEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self, observation_mode='rgb', energy_consum=False):
        self.observation_mode = observation_mode
        self.energy_consum = energy_consum
        self.width = 42
        self.height = 42

        self.action_space = spaces.Discrete(4)

        if observation_mode == 'rgb':
            self.observation_space = spaces.Box(low=0, high=256, shape=(self.width * 2, self.height * 2, 3),
                                                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 1), dtype=np.uint8)

        self.snake = Snake()
        self.foods = []
        self.n_foods = 5
        self.viewer = None
        self.np_random = np.random

    def set_foods(self, n):
        self.n_foods = n

    def reset(self):
        self.snake.body.clear()
        self.foods.clear()
        empty_cells = self.get_empty_cells()
        empty_cells = self.snake.init(empty_cells, self.np_random)
        self.foods = [empty_cells[i] for i in self.np_random.choice(len(empty_cells), self.n_foods, replace=False)]
        import pdb
        pdb.set_trace()
        return self.get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        snake_tail = self.snake.step(action)

        self.snake.reward = 0.

        if self.energy_consum:
            self.snake.reward -= 0.01

        if self.snake.head in self.foods:
            self.snake.reward += 1.
            self.snake.body.append(snake_tail)
            self.foods.remove(self.snake.head)
            empty_cells = self.get_empty_cells()
            food = empty_cells[self.np_random.choice(len(empty_cells))]
            self.foods.append(food)

        # snake collided wall
        if self.is_collided_wall(self.snake.head):
            self.snake.reward -= 1.
            self.snake.done = True

        # snake bite itself
        if self.snake.head in list(self.snake.body)[1:]:
            self.snake.reward -= 1.
            self.snake.done = True

        self.snake.reward = np.clip(self.snake.reward, -1., 1.)

        return self.get_observation(), self.snake.reward, self.snake.done, {}

    def get_observation(self):
        if self.observation_mode == 'rgb':
            return self.get_image()
        else:
            observation = np.zeros((self.width, self.height), dtype=np.uint8)

            for x, y in self.snake.body:
                try:
                    observation[x][y] = 100
                except:
                    pass

            for food in self.foods:
                x, y = food
                observation[x][y] = 200
            return observation[:, :, None]

    def get_image(self):
        board_width = 2 * self.width
        board_height = 2 * self.height
        cell_size = 2

        board = Board(board_height, board_width)
        for x, y in self.snake.body:
            board.fill_cell((x * cell_size, y * cell_size), cell_size, BoardColor.BODY_COLOR)

        for food in self.foods:
            x, y = food
            board.fill_cell((x * cell_size, y * cell_size), cell_size, BoardColor.FOOD_COLOR)
        return board.board

    def get_empty_cells(self):
        empty_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        for cell in self.snake.body:
            if cell in empty_cells:
                empty_cells.remove(cell)
        for food in self.foods:
            if food in empty_cells:
                empty_cells.remove(food)
        return empty_cells

    def is_collided_wall(self, head):
        x, y = head
        if x < 0 or x > (self.width - 1) or y < 0 or y > (self.height - 1):
            return True
        return False

    def render(self, mode='human'):
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen


class Snake(object):

    def __init__(self):
        self.body = deque()
        self.prev_act = None
        self.done = False
        self.reward = 0.

    def step(self, action):
        if not self.done:
            if not self.is_valid_action(action):
                action = self.prev_act
            self.prev_act = action
            x, y = self.head
            if action == SnakeAction.LEFT:
                self.body.appendleft((x, y - 1))
            if action == SnakeAction.RIGHT:
                self.body.appendleft((x, y + 1))
            if action == SnakeAction.UP:
                self.body.appendleft((x - 1, y))
            if action == SnakeAction.DOWN:
                self.body.appendleft((x + 1, y))
            return self.body.pop()

    @property
    def head(self):
        return self.body[0]

    def is_valid_action(self, action):
        if len(self.body) == 1:
            return True

        horizontal_actions = [SnakeAction.LEFT, SnakeAction.RIGHT]
        vertical_actions = [SnakeAction.UP, SnakeAction.DOWN]

        if self.prev_act in horizontal_actions:
            return action in vertical_actions
        return action in horizontal_actions

    def init(self, empty_cells, np_random):
        self.body.clear()
        self.done = False
        self.reward = 0.
        self.prev_act = None
        start_head = empty_cells[np_random.choice(len(empty_cells))]
        self.body.appendleft(start_head)
        empty_cells.remove(start_head)
        return empty_cells


class Board(object):

    def __init__(self, height, weight):
        self.board = np.empty((height, weight, 3), dtype=np.uint8)
        self.board[:, :, :] = BoardColor.SPACE_COLOR

    def fill_cell(self, vertex, cell_size, color):
        x, y = vertex
        self.board[x:x + cell_size, y:y + cell_size, :] = color


class SnakeEnvMC(SnakeEnv):

    def __init__(self):
        super().__init__(observation_mode='rgb')


snake_env = SnakeEnvMC()

if __name__ == '__main__':
    ss = SnakeEnv()
    ss.reset()
