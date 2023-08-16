from dfa.dfa import DFA_3, DFA_4, check_dfa, DFA_1, DFA_2
from run import Conifg, train, evaluate, one_round_evaluate
from exp3 import plot_robot_trajectory_on_seaborn_grid

DFA_1 = check_dfa(DFA_1)
DFA_2 = check_dfa(DFA_2)

config = Conifg()
config.total_timesteps = 400000
config.nsteps = 1024

env_file_path = "env/environment/8 x 8/1.yaml"
config.env_file_path = env_file_path

config.dfa = [DFA_2, DFA_1]

config.num_layers = 4
# config.ent_coef = 0.01  # Set Entropy Coefficient to encourage exploration
# config.lr = 5e-4        # Set Learning Rate
# config.cliprange = 0.1  # Set Clip Range

# train(config)
model_save_path = 'D:\project\FormalAutoRewardPPO\model\plot_model/390round_model'
p = one_round_evaluate(model_save_path, config, 'mlp')
print(p)
plot_robot_trajectory_on_seaborn_grid(p)
# evaluate(model_save_path, config, 'mlp')