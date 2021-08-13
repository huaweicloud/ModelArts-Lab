import gym


ACTION_SPACE = gym.spaces.Discrete(6)
OBSERVATION_SPACE = gym.spaces.Box(0, 255, (42, 42, 1))
CONFIG_PPO = {
    "rollout_fragment_length": 10,
    "train_batch_size": 5000,
    "framework": "tf",
    "preprocessor_pref": "rllib",
    "batch_mode": "complete_episodes",
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "clip_rewards": True,
    "clip_param": 0.1,
    "vf_clip_param": 10.0,
    "entropy_coeff": 0.01,
    "sgd_minibatch_size": 500,
    "num_sgd_iter": 10,
    "observation_filter": "NoFilter",
    "model": {
        "dim": 42,
        "framestack": False,
        "vf_share_layers": True
    }
}
CONFIG_DQN = {
    "gamma": 0.99,
    "lr": .0001,
    "learning_starts": 10000,
    "buffer_size": 100000,
    "rollout_fragment_length": 128,
    "train_batch_size": 1024,
    "exploration_config": {
        "epsilon_timesteps": 200000,
        "final_epsilon": .01
    },
    "model": {
        "dim": 42,
        "grayscale": True,
        "framestack": False,
        "zero_mean": False,
        "vf_share_layers": True
    }
}
