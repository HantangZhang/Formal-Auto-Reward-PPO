from automata.fa.dfa import DFA

class RewardMachine:

    def __init__(self, dfas):

        self.Q = {}  # list of non-terminal RM states from DFA states
        self.q0 = 0  # initial state

        self.dfas = dfas  # A list of DFAs

        # Build the reward machine based on the provided DFAs
        self._build_from_dfas(dfas)

    def _build_from_dfas(self, dfas):

        # Number of DFAs
        self.n = len(dfas)

        # States of RM (Cartesian product of all DFA states)
        self.Q = {q: [state for state in dfa.states] for q, dfa in enumerate(dfas)}
        self.Q[-1] = ['end']

        # Each DFA is assigned a unique ID which will be the state in the RM
        self.dfa_ids = list(range(self.n))
        self.dfa_ids.append(-1)

        # Store DFAs by their ID
        self.id_to_dfa = {i: dfa for i, dfa in enumerate(dfas)}

        # Initial state of RM
        self.reset()

    def reset(self):

        self.rm_current_state = self.q0
        self.dfa_current_state = self.dfas[0].initial_state

    def get_dfa(self, q):
        """
        Method to get the DFA corresponding to a given RM state (DFA ID).
        """
        return self.id_to_dfa[q]

    def delta_q(self, dfa_state, sigma):
        """
        Transition function that takes a state and an input symbol and returns the next state.
        """

        q = self.rm_current_state
        # Get the current DFA based on RM state
        current_dfa = self.id_to_dfa[q]

        # Determine the next state in the current DFA
        next_dfa_state = current_dfa.transitions[dfa_state][sigma]
        # Check if the next state is a final state of the current DFA
        if next_dfa_state in current_dfa.final_states:
            # If there's a next DFA, transition to it. Otherwise, transition to terminal rm state -1.
            rm_state = q + 1 if q + 1 in self.dfa_ids else -1
        else:
            rm_state = q
        return next_dfa_state, rm_state


    def R(self, sigma):
        """
        Reward function that takes a current state, an input symbol, and returns the reward.
        """

        next_dfa_state, next_rm_state = self.delta_q(self.dfa_current_state, sigma)

        def compute_k(state, position, dfa):
            """
            Calculate the value of k for a PQCLTLf statement composed of two LTL statements.
            """

            # Get the opt value based on the position
            opt_value = dfa.opt_list[position]

            if state[position] == '1':
                return 1
            elif position + 1 < len(state) and state[position + 1] == '1':
                # Recursive call to compute n (Currently, it directly checks the satisfaction of the next position)
                n = compute_k(state, position + 1, dfa)
                return n + opt_value
            return 0

        if next_rm_state != self.rm_current_state:
            dfa = self.get_dfa(self.rm_current_state)
            if next_dfa_state in dfa.final_states:
                if dfa.op == 'ordered':
                    k = compute_k(next_dfa_state, 0, dfa)
                elif dfa.op == 'ltl':
                    k = 1

            reward = 1 - (k / (dfa.opt_value + 1))
            if next_rm_state != -1:
                self.dfa_current_state = self.get_dfa(next_rm_state).initial_state
            else:
                self.dfa_current_state = None
        else:
            reward = 0
            self.dfa_current_state = next_dfa_state

        self.rm_current_state = next_rm_state

        return reward

if __name__=='__main__':
    from dfa.dfa import DFA_1, DFA_2, sync, sync_or
    from dfa.dfa import DFA_3, DFA_4, check_dfa
    PQCLTLTf = sync_or(DFA_1, DFA_2)
    # PQCLTLTf2 = sync_or(DFA_6, DFA_7)

    DFA_3 = check_dfa(DFA_3)
    DFA_4 = check_dfa(DFA_4)
    rm = RewardMachine([DFA_3, DFA_4])
    # rm['transitions']
    test_state = (('0', '0'), ('1', '0'))
    # print(rm.R(test_state, 'b'))
    print(rm.Q)
    print(rm.dfa_ids)
    print(rm.dfa_current_state, rm.rm_current_state)
    print(rm.R('a'))
    print(rm.dfa_current_state, rm.rm_current_state)
    print(rm.R('b'))
    print(rm.dfa_current_state, rm.rm_current_state)
    print(rm.R('c'))
    print(rm.dfa_current_state, rm.rm_current_state)
    print(1111, rm.id_to_dfa[1])
    print(rm.R('a'))
    print(rm.dfa_current_state, rm.rm_current_state)
    print(rm.R('c'))
    print(rm.dfa_current_state, rm.rm_current_state)
    '''
    1. compute k only can take two ltl 
    '''

