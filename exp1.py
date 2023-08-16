from dfa.dfa import DFA_1, DFA_2, sync_or, check_dfa
from run import Conifg, train, evaluate

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