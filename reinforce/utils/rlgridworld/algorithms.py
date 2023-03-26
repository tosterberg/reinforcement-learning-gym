from rlgridworld.grid import Grid

# from page 75 of Sutton and Barto, RL, 2nd. Ed.

def iterative_policy_evaluation(gw, policy, gamma=0.9, epsilon=0.001):
    while True:
        biggest_change = 0
        for node in gw:
            state = node.state
            if not gw.is_terminal(state) and not gw.is_barrier(state):
                # get current (old) value
                old_value = gw.get_value(state)
                # get action from policy
                action = policy[state]
                # get immediate reward for action
                reward = gw.get_reward_for_action(state, action)
                # get value at destination state
                value_at_dest = gw.get_value_at_destination(state, action)
                # compute new value
                new_value = reward + gamma*value_at_dest
                # set new value for state
                gw.set_value(state, new_value)
                # see if |new_value-old_value| is larger than biggest_change
                biggest_change = max(
                    biggest_change, abs(new_value-old_value))
        # iterated over all states, so see if biggest_change is small enough
        if biggest_change < epsilon:
            break

# from page 80 of Sutton and Barto, RL, 2nd. Ed.

def policy_iteration(gw, policy, gamma=0.9, epsilon=0.001):
    while True:
        # perform iterative policy evaluation to update values
        iterative_policy_evaluation(gw, policy, gamma, epsilon)
        # update policy from new values
        new_policy = compute_policy_from_values(gw, gamma)
        # see if policy has changed
        for action in policy:
            if policy[action] == new_policy[action]:
                policy_stable = True
            else:
                policy_stable = False
                break
        # update policy
        policy = new_policy
        # repeat until policy does not change
        if policy_stable == True:
            break

# from page 83 of Sutton and Barto, RL 2nd. Ed.

def value_iteration(gw, gamma=0.9, epsilon=0.001):
    count = 0
    while True:
        count += 1
        biggest_change_in_value = 0
        for node in gw:
            state = node.state
            if not gw.is_terminal(state) and not gw.is_barrier(state):
                old_value = gw.get_value(state)
                new_value = float('-inf')
                # valid decisions and rewards at current state
                dr = gw.valid_decisions_and_rewards(state)
                for action, reward in dr.items():
                    reward = gw.get_reward_for_action(state, action)
                    value_at_dest = gw.get_value_at_destination(state, action)
                    value = reward + gamma*value_at_dest
                    if value > new_value:
                        new_value = value
                    gw.set_value(state, new_value)
                biggest_change_in_value = max(biggest_change_in_value,
                                                  abs(new_value - old_value))
        if biggest_change_in_value < epsilon:
            break

# computes policy from values - argmax( values ) - my algorithm

def compute_policy_from_values(gw, gamma=0.9):
    # create null policy dictionary
    policy = {}
    # loop over all states
    for node in gw:
        state = node.state
        # assign 'no' policy to barrier states
        if gw.is_barrier(state):
            policy[state] = ''
        # assign 'no' policy to terminal sttes
        if gw.is_terminal(state):
            policy[state] = ''
        # for all non terminal and non barrier states
        if not gw.is_terminal(state) and not gw.is_barrier(state):
            # set candidate best action and best value
            best_action = None
            best_value = float('-inf')
            # get dictionary of all valid decisions and rewards at current state (i,j)
            decisions_rewards = gw.valid_decisions_and_rewards(state)
            # iterate over all action, reward in
            for action, reward in decisions_rewards.items():
                # get reward for current action
                reward = gw.get_reward_for_action(state, action)
                # get the value of the destination state for the current action
                value_at_dest = gw.get_value_at_destination(state, action)
                # compute candidate vale
                value = reward + gamma*value_at_dest
                # if value is better, then update best action and best value
                if value > best_value:
                    best_value = value
                    best_action = action
            # add best action to the policy dictionary
            policy[state] = best_action
    return policy
