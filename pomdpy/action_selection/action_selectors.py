import random
import numpy as np
safety_threshold = 0.92
SMC = True

def setSMCflag(val):
    SMC = val
# UCB1 action selection algorithm
def ucb_action(mcts, current_node, greedy):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    N = mapping.total_visit_count
    log_n = np.log(N + 1)

    all_actions = np.array(list(mapping.entries.values()))
    actions = all_actions[:4].tolist()
    random.shuffle(actions)
    for action_entry in actions:

        # Skip illegal actions
        current_p = action_entry.mean_p_value
        # STEP find the q value with respect to p value - restrict only move actions
        if(SMC):
            if (current_p>=1-safety_threshold ):
                # print(current_p)
                continue

        if not action_entry.is_legal:
            continue

        current_q = action_entry.mean_q_value
        if(current_q == np.inf):
            current_q = 0
        # If the UCB coefficient is 0, this is greedy Q selection
        if not greedy:
            current_q += mcts.find_fast_ucb(N, action_entry.visit_count, log_n)

        if (current_q >= best_q_value):
            if current_q > best_q_value:
                best_actions = []
            best_q_value = current_q
            # best actions is a list of Discrete Actions
            best_actions.append(action_entry.get_action())
    look_actions = all_actions[4:].tolist()
    # best_actions = best_actions if len(best_actions)>0 else random.choice(look_actions)

    # assert best_actions.__len__() is not 0


    return random.choice(best_actions) if len(best_actions)>0 else random.choice(look_actions)


def e_greedy(current_node, epsilon):
    best_actions = []
    best_q_value = -np.inf
    mapping = current_node.action_map

    actions = list(mapping.entries.values())
    random.shuffle(actions)

    if np.random.uniform(0, 1) < epsilon:
        for action_entry in actions:
            if not action_entry.is_legal:
                continue
            else:
                return action_entry.get_action()
        # No legal actions
        raise RuntimeError('No legal actions to take')
    else:
        # Greedy choice
        for action_entry in actions:
            # Skip illegal actions
            if not action_entry.is_legal:
                continue

            current_q = action_entry.mean_q_value

            if current_q >= best_q_value:
                if current_q > best_q_value:
                    best_actions = []
                best_q_value = current_q
                # best actions is a list of Discrete Actions
                best_actions.append(action_entry.get_action())

        assert best_actions.__len__() is not 0

        return random.choice(best_actions)