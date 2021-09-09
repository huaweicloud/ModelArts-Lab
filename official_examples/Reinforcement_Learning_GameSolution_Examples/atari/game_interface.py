import random

import gym

GameInterface = gym.make("BreakoutNoFrameskip-v4")

if __name__ == '__main__':
    game_env = GameInterface
    game_env.reset()
    for _ in range(100):
        obs, reward, done, info = game_env.step(random.choice([0, 1, 2, 3]))
        print(reward)
    game_env.close()
