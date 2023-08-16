import gym
import numpy as np
from gym import spaces
from env.mdp import MDP
from reward_machine.reward_machine import RewardMachine


class GridWorld(gym.Env):

    """
        A custom environment that simulates a grid world.
        This class inherits from the gym.Env class.
    """

    def __init__(self, config, dfas):
        """
        Initializes a new GridWorld instance.

        Parameters:
        config : Configuration object with environment parameters.
        dfas : A list of DFAs (Deterministic Finite Automata) constructed from formal language formulas.

        Returns:
        None
        """
        super(GridWorld, self).__init__()

        self.mdp = MDP(file_path=config.env_file_path)

        self.num_envs = 1

        self._build_reward_machine(dfas)

        self.coord_space = [8, 8]
        self.rm_sapce = [self.RM.n + 1]
        self.dfa_space = [len(self.dfa_state_mapping)]

        self.observation_space = spaces.MultiDiscrete(self.coord_space + self.rm_sapce + self.dfa_space)
        self.action_space = spaces.Discrete(4)

        self.current_state = None
        self.initial_state = [6, 6]
        self.max_env_step = config.max_env_step

        self.current_step = 0
        self.episode = 0
        self.total_reward = 0
        self.current_episode_reward = 0
        self.rm_dfa_traj = {'rm_traj': [], 'dfa_traj': []}

    def _build_reward_machine(self, dfas):
        """
        Build reward machine based on the given DFA.
        """
        self.RM = RewardMachine(dfas)

        self.dfa_state_mapping = {}
        s = 0
        for n, dfa in enumerate(self.RM.dfas):
            for state in dfa.states:
                new_state = str(n) + "".join(state)
                self.dfa_state_mapping[new_state] = s
                s += 1
        self.dfa_state_mapping[-1] = s+1

    def build_observation(self, coord, rm_state, dfa_state):
        """
        Build observations based on the given states, which can be fed into a neural network.

        Parameters:
            coord: The environment state, referring to the current coordinates of the robot.
            rm_state: The current state of the reward machine, such as 0, 1, 2.
            dfa_state：The current state of the DFA, obtained based on the mapped state, such as 0, 1, 2.

        Returns:
            state: An observation that can be input to a neural network. The observation is represented as a NumPy array.
        """
        coord = np.array(coord)
        rm_state = np.array([rm_state])
        dfa_state = self.convert_dfa_state(dfa_state)
        state = np.concatenate([coord, rm_state, dfa_state])

        self.coord_point = tuple(coord)
        return state

    def reset(self):
        self.RM.reset()
        self.current_state = self.build_observation(self.initial_state, self.RM.q0, self.RM.dfas[0].initial_state)

        self.current_step = 0

        self.episode += 1
        self.total_reward = 0

        return self.current_state

    def step(self, action):

        next_states = self.mdp.deterministic_transition(self.coord_point, int(action))

        reward = self.get_reward()

        done, done_info = self.get_done()

        self.total_reward += float(reward)
        self.current_state = self.build_observation(next_states, self.RM.rm_current_state, self.RM.dfa_current_state)

        infos = [
            {'done': done_info},
            {'episode': {'r': self.total_reward, 'l': self.episode}},
        ]

        self.current_step += 1

        self.current_episode_reward += float(reward)
        self.rm_dfa_traj['rm_traj'].append(self.RM.rm_current_state)
        self.rm_dfa_traj['dfa_traj'].append(self.RM.dfa_current_state)
        # if done:
        #     if done_info == 'b':
        #         print('current rm state：', self.RM.rm_current_state,
        #               'current dfa state：', self.RM.dfa_current_state,
        #               'current reward:', reward,
        #               'done reason:', done_info,
        #               'total reward：', self.current_episode_reward,
        #               'path traj:', self.rm_dfa_traj)
        #     else:
        #         pass
        #     self.current_episode_reward = 0
        #     self.rm_dfa_traj = {'rm_traj': [], 'dfa_traj': []}

        return self.current_state, reward, [done], infos

    def get_done(self):
        """
        Check if the environment is in a done state. The conditions for being done include:
            - The robot reaching a sink position
            - The reward machine's state being a terminal state (-1)
            - Reaching the maximum step count per episode set by the environment

        Returns:
            tuple: A tuple containing a boolean value indicating whether the environment is done (True or False),
                    and a string explaining the reason for the environment termination (done_info).
        """
        labeled_state = self.mdp.L[self.coord_point]
        if labeled_state == 'o':
            return True, 'o'
        elif self.RM.rm_current_state == -1:
            return True, labeled_state
        else:
            # reach to max step
            if self.current_step == self.max_env_step:
                return True, 'max'
            return False, 'normal'

    def labeling(self, env_state):
        """
        Labeling function, retrieve the corresponding sigma based on the current position of the robot.
        """
        return self.mdp.L[env_state]

    def get_reward(self):
        """
        Call the reward function within the reward machine to obtain a reward value.
        """
        sigma = self.labeling(self.coord_point)
        reward = self.RM.R(sigma)

        return np.array([reward])

    def convert_dfa_state(self, dfa_state):
        """
        Convert the original DFA states into states that can be input to a neural network.
        - Original DFA states, such as  "1", "("0", "1")"
        - mapped DFA states that can be used as input to a neural network. such as 0, 1, 2
        """
        n = self.RM.rm_current_state
        if dfa_state:
            convert_state = self.dfa_state_mapping[str(n) + "".join(dfa_state)]
        else:
            convert_state = self.dfa_state_mapping[-1]

        convert_state = np.array([convert_state])
        return convert_state


if __name__=='__main__':
    env = GridWorld(file_path="environment/8 x 8/1.yaml")
    state = env.reset()
    print(state)
    # state = env.step(1)
    print(env.mdp.L)





