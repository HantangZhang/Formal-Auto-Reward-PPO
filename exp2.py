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

# train(config)
model_save_path = 'D:\project\FormalAutoRewardPPO\model\plot_model/140round_model'
evaluate(model_save_path, config, 'mlp')