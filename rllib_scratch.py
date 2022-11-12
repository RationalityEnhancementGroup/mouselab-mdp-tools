"""adapted from: https://docs.ray.io/en/latest/rllib/index.html""" 
# Import the RL algorithm (Algorithm) we would like to use.
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
import gym
from mouselab.mouselab import MouselabEnvSymmetricRegistered

from ray.tune.registry import register_env
from mouselab.cost_functions import linear_depth
env = MouselabEnvSymmetricRegistered("high_increasing")
register_env("high_increasing_mouselab", lambda config: MouselabEnvSymmetricRegistered("high_increasing", cost=linear_depth(depth_cost_weight=1.0,static_cost_weight=2.0)))


# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "high_increasing_mouselab",
    "horizon": 13,
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [13, 13],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

# Create our RLlib Trainer.
algo = PPO(config=config)
# algo = DQN(env="high_increasing_mouselab", config={"framework": "tf2", "horizon":13})

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(3):
    print(algo.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# algo.evaluate()

env = MouselabEnvSymmetricRegistered("high_increasing", cost=linear_depth(depth_cost_weight=1.0,static_cost_weight=2.0))
# run until episode ends
episode_reward = 0
i_episode = 0
done = False
obs = env.reset()
while not done and i_episode <= 13:
    print(obs)
    action = algo.compute_single_action(obs)
    print(action)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    i_episode += 1

print(episode_reward)