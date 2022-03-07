from ray.rllib.examples.env.multi_agent import MultiAgentCartPole


class GameInterface():
    def __init__(self, *args):
        print("Initializing multiagent cartpole...")
        self.game = MultiAgentCartPole({"num_agents": 4})
        print("multiagent cartpole initialized.")

    def reset(self, *args):
        obs = self.game.reset()
        return obs

    def step(self, action_dict, *args):
        obs, reward, done, info = self.game.step(action_dict)
        return obs, reward, done, info


if __name__ == '__main__':
    game_env = GameInterface()
    game_env.reset()
    game_env.step({0: 0, 1: 1, 2: 0, 3: 1})
