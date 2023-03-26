def play_game(gw, policy, state=(0,0)):
    # default game starting point is state = (0,0) 
    # list of tuples that are (state, reward) pairs
    states_and_rewards = [] # list of tuples that are (state, reward) pairs
    converged = False
    while not converged:
        # get action from policy
        action = policy[state] # get action from policy
        # find reward for the action
        reward = gw.get_reward_for_action(state, action)
        # more to the new state
        stateprime = move(state,action)
        # add new state and reward to the list
        states_and_rewards.append((state,reward))
        # if you have moved to a terminal state, then stop
        if gw.is_terminal(stateprime):
            converged = True
        # update state to new state
        state = stateprime
    return states_and_rewards

def move(state, action): # only valid actions at states are sent to move
    i,j = state
    if action == 'left':
        j = j-1
    if action == 'right':
        j = j+1
    if action == 'down':
        i = i-1
    if action == 'up':
        i = i+1
    return (i,j)