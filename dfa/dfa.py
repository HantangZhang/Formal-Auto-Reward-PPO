from automata.fa.dfa import DFA
from collections import defaultdict
from itertools import product

# F b ->
DFA_1 = DFA(
    states={"0", "1"},
    input_symbols={"a", "b", "c", "E", "o"},
    transitions={
        "0": {"a": "0", "c": "0", "b": "1", "E": "0", "o": "0"},
        "1": {"a": "1", "c": "1", "b": "1", "E": "1", "o": "1"},
    },
    initial_state="0",
    final_states={"1"},
)

# (F a | F c)
DFA_2 = DFA(
    states={"0", "1"},
    input_symbols={"a", "E", "c", "o", "b"},
    transitions={
        "0": {"a": "1", "E": "0", "c": "1", "o": "0", "b": "0"},
        "1": {"a": "1", "E": "1", "c": "1", "o": "1", "b": "1"},
    },
    initial_state="0",
    final_states={"1"},
)

# F(a & F(b & F c))
DFA_3 = DFA(
    states={"0", "1", "2", "3"},
    input_symbols={"a", "b", "E", "c", "o"},
    transitions={
        "0": {"a": "2", "b": "0", "E": "0", "c": "0", "o": "0"},
        "1": {"a": "1", "b": "1", "E": "1", "c": "3", "o": "1"},
        "2": {"a": "2", "b": "1", "E": "2", "c": "2", "o": "2"},
        "3": {"a": "3", "b": "3", "E": "3", "c": "3", "o": "3"},
    },
    initial_state="0",
    final_states={"3"},
)

# F (a & F c) | F (b & F c)
DFA_4 = DFA(
    states={"0", "1", "2"},
    input_symbols={"a", "b", "E", "c", "o"},
    transitions={
        "0": {"a": "1", "b": "1", "E": "0", "c": "0", "o": "0"},
        "1": {"a": "1", "b": "1", "E": "1", "c": "2", "o": "1"},
        "2": {"a": "2", "b": "2", "E": "2", "c": "2", "o": "2"},
    },
    initial_state="0",
    final_states={"2"},
)

def check_dfa(dfa):
    if not hasattr(dfa, 'opt'):
        dfa.opt_value = 1
        dfa.op = 'ltl'

    return dfa


def sync(dfa1: DFA, dfa2: DFA) -> DFA:
    try:
        assert dfa1.input_symbols == dfa2.input_symbols
    except AssertionError:
        new_input_symbols = dfa2.input_symbols.union(dfa1.input_symbols)
    else:
        new_input_symbols = dfa2.input_symbols

    for dfa in (dfa1, dfa2):
        check_dfa(dfa)
        for q, a in product(dfa.states, new_input_symbols):
            if a not in dfa.input_symbols:
                dfa.transitions[q][a] = q

    new_states = {
        (a, b)
        for a, b in product(dfa1.states, dfa2.states)
    }
    new_transitions = defaultdict(dict)

    for (state_a, transitions_a), symbol, (state_b, transitions_b) in product(
            dfa1.transitions.items(), new_input_symbols, dfa2.transitions.items()
    ):
        if (
            symbol in transitions_a
            and symbol in transitions_b):
            new_transitions[state_a, state_b][symbol] = (
                transitions_a[symbol],
                transitions_b[symbol],
            )
            new_initial_state = (dfa1.initial_state, dfa2.initial_state)

    new_final_states = set(product(dfa1.states, dfa2.final_states)).union(product(dfa1.final_states, dfa2.states))

    new_dfa = DFA(
        states=new_states,
        input_symbols=new_input_symbols,
        transitions=new_transitions,
        initial_state=new_initial_state,
        final_states=new_final_states,
    )

    return new_dfa


def sync_or(dfa1: DFA, dfa2: DFA) -> DFA:
    dfa = sync(dfa1, dfa2)
    dfa.opt_value = dfa1.opt_value + dfa2.opt_value
    dfa.opt_list = [dfa1.opt_value, dfa2.opt_value]
    dfa.op = 'ordered'
    return dfa

def sync_conj(dfa1: DFA, dfa2: DFA) -> DFA:
    dfa = sync(dfa1, dfa2)
    dfa.opt = dfa1.opt * dfa2.opt
    dfa.op = 'conj'


if __name__=='__main__':
    DFA_1 = DFA(
        states={"0", "1"},
        input_symbols={"a", "b", "c", "E", "o"},
        transitions={
            "0": {"a": "0", "c": "0", "b": "1", "E": "0", "o": "0"},
            "1": {"a": "1", "c": "1", "b": "1", "E": "1", "o": "1"},
        },
        initial_state="0",
        final_states={"1"},
    )

    # (F a | F c)
    DFA_2 = DFA(
        states={"0", "1"},
        input_symbols={"a", "E", "c", "o", "b"},
        transitions={
            "0": {"a": "1", "E": "0", "c": "1", "o": "0", "b": "0"},
            "1": {"a": "1", "E": "1", "c": "1", "o": "1", "b": "1"},
        },
        initial_state="0",
        final_states={"1"},
    )
    new_dfa = sync(DFA_1, DFA_2)
    print(new_dfa)

    new_dfa = sync_or(DFA_1, DFA_2)
    print(new_dfa.opt)

