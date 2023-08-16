import yaml
import pandas as pd
import seaborn as sns
import os.path as osp
import logger
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from env.gridworld import GridWorld
from ppo.ppo2.ppo2 import learn
from ppo.common.models import get_network_builder

class Conifg:
    """
    Configurable parameters for algorithm execution
    """
    def __init__(self):

        self.total_timesteps = 50000
        self.seed = 1
        self.max_env_step = 100
        self.dfa = []

        self.env_file_path = None
        self.network = None

        self.save_path = 'D:\project\FormalAutoRewardPPO\model'

        self.ent_coef = 0.0  # Entropy Coefficient
        self.lr = 3e-4  # Learning Rate
        self.cliprange = 0.2  # Clip Range


def build_env(config):
    """
    Custom Gym environment with DFA statements for trainable environment construction.
    """
    dfas = config.dfa
    env = GridWorld(config, dfas)
    return env

def plot_learning_curve(data, data_name):
    plt.figure(figsize=(10, 6))

    # Plot the mean rewards
    plt.plot(data, label=data_name, color='blue')
    # Add title and labels
    plt.title('Learning Curve')
    plt.xlabel('Batch')
    plt.ylabel(data_name)

    # Display the legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_heatmap(data):
    position_counts = {}
    for position in data:
        if position in position_counts:
            position_counts[position] += 1
        else:
            position_counts[position] = 1

    with open('D:\project\FormalAutoRewardPPO\env\environment\8 x 8/1.yaml', 'r') as file:
        map_data = yaml.load(file, Loader=yaml.FullLoader)

    grid = np.zeros((8, 8))
    obstacles = map_data['obstacles']
    special_points = map_data['L']
    for position, count in position_counts.items():
        grid[position] = count

    heatmap_data = pd.DataFrame(grid)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='Reds', annot=False, cbar=True, linewidths=0.5, linecolor='white')

    for obstacle in obstacles:
        plt.text(obstacle[1] + 0.5, obstacle[0] + 0.5, 'o', color='black', fontsize=12, ha='center', va='center')

    for point, label in special_points.items():
        plt.text(point[1] + 0.5, point[0] + 0.5, label, color='black', fontsize=12, ha='center', va='center')

    plt.title('Heatmap of Robot Positions')
    plt.gca().invert_yaxis()
    plt.show()

def train(config):
    total_timesteps = config.total_timesteps
    seed = config.seed

    env = build_env(config)

    if config.network:
        config.network = config.network
    else:
        config.network = 'mlp'
    network_para = {'num_layers':4}
    model, stat_data = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        network=config.network,
        nsteps=config.nsteps,
        ent_coef=config.ent_coef,  # Use ent_coef from config
        lr=config.lr,  # Use lr from config
        cliprange=config.cliprange,  # Use cliprange from config
        **network_para
    )

    if config.save_path is not None:
        save_path = osp.expanduser(config.save_path)
        model.save(save_path)

    plot_learning_curve(stat_data['mean_reward'], 'Mean Reward')
    logger.log("Running trained model")

def evaluate(model_save_path, config, network, model_fn=None, **network_kwargs):
    env = build_env(config)

    if model_fn is None:
        from ppo.ppo2.model import Model
        model_fn = Model

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        network = policy_network_fn(ob_space.shape)

    model = model_fn(ac_space=ac_space, policy_network=network, ent_coef=0, vf_coef=0.5,
                     max_grad_norm=0.5)

    model = model.load(model_save_path)
    # if model_save_path is not None:
    #     load_path = osp.expanduser(model_save_path)
    #     ckpt = tf.train.Checkpoint(model=model)
    #     manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
    #     ckpt.restore(manager.latest_checkpoint)


    n_steps = 1000
    obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
    obs[:] = env.reset()
    dones = [False]
    positions = []
    for _ in range(n_steps):
        current_obs = tf.constant(obs)
        actions, values, states, neglogpacs = model.step(current_obs)
        actions = actions._numpy()
        obs[:], rewards, dones, infos = env.step(actions)
        if dones[0]:
            env.reset()
        positions.append((obs[0][0],obs[0][1]))

    plot_heatmap(positions)
    print('heatmap has been drawn')

def one_round_evaluate(model_save_path, config, network, model_fn=None, **network_kwargs):
    env = build_env(config)

    if model_fn is None:
        from ppo.ppo2.model import Model
        model_fn = Model

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network_type = network
        policy_network_fn = get_network_builder(network_type)(**network_kwargs)
        network = policy_network_fn(ob_space.shape)

    model = model_fn(ac_space=ac_space, policy_network=network, ent_coef=0, vf_coef=0.5,
                     max_grad_norm=0.5)

    model = model.load(model_save_path)
    # if model_save_path is not None:
    #     load_path = osp.expanduser(model_save_path)
    #     ckpt = tf.train.Checkpoint(model=model)
    #     manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
    #     ckpt.restore(manager.latest_checkpoint)


    n_steps = 1000
    obs = np.zeros((1,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
    obs[:] = env.reset()
    positions = []
    for _ in range(n_steps):
        current_obs = tf.constant(obs)
        actions, values, states, neglogpacs = model.step(current_obs)
        actions = actions._numpy()
        obs[:], rewards, dones, infos = env.step(actions)
        if dones[0]:
            env.reset()
            break
        positions.append((obs[0][0],obs[0][1]))

    return positions


if __name__=='__main__':
    from dfa.dfa import DFA_1, DFA_2, sync_or, check_dfa

    config = Conifg()
    config.total_timesteps = 100000
    config.nsteps = 1024

    env_file_path = "env/environment/8 x 8/1.yaml"
    config.env_file_path = env_file_path
    PQCLTLTf_1 = sync_or(DFA_1, DFA_2)
    config.dfa = [PQCLTLTf_1]
    #
    # train(config)
    model_save_path = 'D:\project\FormalAutoRewardPPO\model\plot_model/40round_model'
    evaluate(model_save_path, config, 'mlp')
