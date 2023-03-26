from rlgridworld.grid import Grid

# defines a functions of a state, i.e., whose domain is a state

def init_V(gw): # V(s), the state-value function, maps states to real numbers
    V = {} 
    for node in gw:
        state = node.state
#        if not gw.is_barrier(state) and not gw.is_terminal(state):
        V[state] = 0 
    return V

def init_pi(gw): # pi(s), the policy function, maps states to actions
    pi = {} 
    for node in gw:
        state = node.state
        if not gw.is_barrier(state) and not gw.is_terminal(state):
            pi[state] = []
    return pi

# defines a function of a state and an action

def init_Q(gw): # Q, the action-value, maps states and actions tp real numbers
    Q = {}
    for node in gw:
        state = node.state
        if not gw.is_barrier(state) and not gw.is_terminal(state):
            Q[state] = {}
            all_actions = gw.valid_decisions(state)
            for a in all_actions:
                Q[state][a] = 0
    return Q

# defines a function of a state and an action

def init_C(gw):  # Q maps states and actions tp real numbers
    C = {}
    for node in gw:
        state = node.state
        if not gw.is_barrier(state) and not gw.is_terminal(state):
            C[state] = {}
            all_actions = gw.valid_decisions(state)
            for a in all_actions:
                C[state][a] = 0
    return C

def init_Returns(gw): # Returns maps states and actions to a list
    Returns = {}
    for node in gw:
        state = node.state
        if not gw.is_barrier(state) and not gw.is_terminal(state):
            Returns[state] = {}
            all_actions = gw.valid_decisions(state)
            for a in all_actions:
                Returns[state][a] = []
    return Returns
