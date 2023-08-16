import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from run import Conifg, train, one_round_evaluate
from dfa.dfa import DFA_3, DFA_4, check_dfa, DFA_1, DFA_2
from run import Conifg, train, evaluate

DFA_3 = check_dfa(DFA_1)
DFA_4 = check_dfa(DFA_2)

config = Conifg()
config.total_timesteps = 150000
config.nsteps = 1024

env_file_path = "env/environment/8 x 8/1.yaml"
config.env_file_path = env_file_path

config.dfa = [DFA_3, DFA_4]

def plot_robot_trajectory_on_seaborn_grid(trajectory):
    # trajectory = [
    #     (0, 0), (1, 1), (2, 2), (3, 3),
    #     (4, 4), (5, 5), (6, 6), (7, 7),
    #     (7, 6), (6, 5), (5, 4), (4, 3),
    #     (3, 2), (2, 1), (1, 0)
    # ]

    with open('D:\project\FormalAutoRewardPPO\env\environment\8 x 8/1.yaml', 'r') as file:
        map_data = yaml.load(file, Loader=yaml.FullLoader)
    obstacles = map_data['obstacles']
    special_points = map_data['L']

    grid = np.zeros((8, 8))
    heatmap_data = pd.DataFrame(grid)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='Reds', annot=False, cbar=False, linewidths=0.5, linecolor='white')

    for i in range(len(trajectory) - 1):
        start = trajectory[i]
        end = trajectory[i + 1]
        plt.arrow(start[1] + 0.5, start[0] + 0.5, end[1] - start[1], end[0] - start[0],
                  head_width=0.2, head_length=0.2, fc='blue', ec='blue')

    for obstacle in obstacles:
        plt.text(obstacle[1] + 0.5, obstacle[0] + 0.5, 'o', color='black', fontsize=12, ha='center', va='center')

    for point, label in special_points.items():
        plt.text(point[1] + 0.5, point[0] + 0.5, label, color='black', fontsize=12, ha='center', va='center')

    plt.title('Robot Trajectory')

    plt.gca().invert_yaxis()

    plt.show()

if __name__ == '__main__':

    model_save_path = 'D:\project\FormalAutoRewardPPO\model\plot_model/140round_model'
    positions = one_round_evaluate(model_save_path, config, 'mlp')

    # 绘制轨迹图
    plot_robot_trajectory_on_seaborn_grid(positions)
