from rlgridworld.node import Node

class Grid:
    """GridWorld's Grid of Nodes"""

    def __init__(self, numberRows, numberColumns):
        # Create Grid as list of lists with M rows and N columns
        self.M = numberRows
        self.N = numberColumns

        # create grid, i indexes M and j indexes N
        self.grid=[]
        for i in range(0,self.M):
            nodes = []
            for j in range(0,self.N):
                nodes.append(Node())
            self.grid.append(nodes)

        for i in range(0,self.M):
            for j in range(0,self.N):
                self.grid[i][j].set_state((i,j))
    # define iterator for states in grid
    def __iter__(self):
        self.i = 0
        self.j = 0
        return self
    # define next i,j pair for iterator
    def __next__(self):
        if self.i < self.M and self.j < self.N:
            v = self.grid[self.i][self.j]
            if self.i < self.M:
                if self.j < self.N:
                    self.j += 1
                if self.j == self.N:
                    self.j = 0
                    self.i += 1
            return v
        else:
            raise StopIteration

    def get_node(self,state):
        return self.grid[state[0]][state[1]]

    def print_grid_state(self):
        for i in range(0,self.M):
            for j in range(0,self.N):
                print(f"Row: {self.grid[i][j].get_state()[0]}"
                    f", Column: {self.grid[i][j].get_state()[1]}"
                    f", Value: {self.grid[i][j].get_value()}"
                    f", is_terminal: {self.grid[i][j].get_is_terminal()}"
                    f", is_barrier: {self.grid[i][j].get_is_barrier()}")     

    def print_grid_rewards(self):
        for i in range(0,self.M):
            for j in range(0,self.N):
                print(f"Row: {self.grid[i][j].get_state()[0]}"
                f", Column: {self.grid[i][j].get_state()[1]}"
                f", Left: {self.grid[i][j].get_left()}"
                f", Right: {self.grid[i][j].get_right()}" 
                f", Down: {self.grid[i][j].get_down()}"
                f", Up: {self.grid[i][j].get_up()}")

    def init_rewards(self,reward):
        # interior nodes
        for i in range(0+1,self.M-1):
            for j in range(0+1,self.N-1):
                self.grid[i][j].set_left(reward)
                self.grid[i][j].set_right(reward)
                self.grid[i][j].set_down(reward)
                self.grid[i][j].set_up(reward)
                
        # left boundary nodes
        for i in range(0+1,self.M-1):
            for j in range(0,1):
                self.grid[i][j].set_right(reward)
                self.grid[i][j].set_down(reward)
                self.grid[i][j].set_up(reward)
                
        # right boundary nodes
        for i in range(0+1,self.M-1):
            for j in range(self.N-1,self.N):
                self.grid[i][j].set_left(reward)
                self.grid[i][j].set_down(reward)
                self.grid[i][j].set_up(reward)
                
        # bottom boundary nodes
        for i in range(0,1):
            for j in range(0+1,self.N-1):
                self.grid[i][j].set_left(reward)
                self.grid[i][j].set_right(reward)
                self.grid[i][j].set_up(reward)

        # top bouindary nodes
        for i in range(self.M-1,self.M):
            for j in range(0+1,self.N-1):
                self.grid[i][j].set_left(reward)
                self.grid[i][j].set_right(reward)
                self.grid[i][j].set_down(reward)

        # left, bottom corner
        i = 0
        j = 0
        self.grid[i][j].set_right(reward)
        self.grid[i][j].set_up(reward)

        # left, top corner
        i = self.M-1
        j = 0
        self.grid[i][j].set_right(reward)
        self.grid[i][j].set_down(reward)

        # right, bottom corner
        i = 0
        j = self.N-1
        self.grid[i][j].set_left(reward)
        self.grid[i][j].set_up(reward)

        # right, top corner
        i = self.M-1
        j = self.N-1
        self.grid[i][j].set_left(reward) # changed in 2021 
        self.grid[i][j].set_down(reward)
        
    def is_valid_node_index(self,i,j):
        if i>=0 and i<=self.M-1 and j>=0 and j<=self.N-1:
            return True
        else:
            return False 

    def is_barrier(self,state):
        i, j = state[0], state[1] 
        if self.grid[i][j].get_is_barrier():
            return True
        else:
            return False

    def set_barrier(self,state):
        # set barrier flag to True
        # set rewards from adjacent nodes that lead to barrier to None
        i, j = state[0], state[1]
        self.grid[i][j].set_is_barrier(True)
        self.grid[i][j].set_left(None)
        self.grid[i][j].set_right(None)
        self.grid[i][j].set_down(None)
        self.grid[i][j].set_up(None)
        #node to left
        if self.is_valid_node_index(i,j-1):
            self.grid[i][j-1].set_right(None)
        #node to right
        if self.is_valid_node_index(i,j+1):
            self.grid[i][j+1].set_left(None)
        #node below
        if self.is_valid_node_index(i-1,j):
            self.grid[i-1][j].set_up(None)
        #node above
        if self.is_valid_node_index(i+1,j):
            self.grid[i+1][j].set_down(None)

    def is_terminal(self,state):
        i, j = state[0], state[1]
        if self.grid[i][j].get_is_terminal():
            return True
        else:
            return False

    def set_terminal(self,state):
        i, j = state[0], state[1]
        # set terminal flag to True
        self.grid[i][j].set_is_terminal(True)
        # set reward from terminal state to None
        self.set_reward(state,'left',None)
        self.set_reward(state,'right',None)
        self.set_reward(state,'down',None)
        self.set_reward(state,'up',None)

    def set_reward(self,state,direction,reward):
        i, j = state[0], state[1]
        if direction == 'left':
            self.grid[i][j].set_left(reward)
        if direction == 'right':
            self.grid[i][j].set_right(reward)
        if direction == 'down':
            self.grid[i][j].set_down(reward)
        if direction == 'up':
            self.grid[i][j].set_up(reward)

    def valid_decisions(self,state):
        i, j = state[0], state[1]
        actions = []
        if self.is_terminal(state) or self.is_barrier(state):
            actions = []
        else:
            if self.is_valid_node_index(i,j-1):
                if not self.is_barrier((i,j-1)): 
                    actions.append('left')
            if self.is_valid_node_index(i,j+1):
                if not self.is_barrier((i,j+1)):
                    actions.append('right')
            if self.is_valid_node_index(i-1,j):
                if not self.is_barrier((i-1,j)): 
                    actions.append('down')
            if self.is_valid_node_index(i+1,j):
                if not self.is_barrier((i+1,j)): 
                    actions.append('up')
        return actions

    def valid_decisions_and_rewards(self,state):
        i, j = state[0], state[1]
        dr = {}
        if self.is_terminal(state) or self.is_barrier(state):
            dr = {}
        else:
            if self.is_valid_node_index(i,j-1):
                if not self.is_barrier((i,j-1)): 
                    dr.update({'left':self.grid[i][j].get_left()})
            if self.is_valid_node_index(i,j+1):
                if not self.is_barrier((i,j+1)):
                    dr.update({'right':self.grid[i][j].get_right()})
            if self.is_valid_node_index(i-1,j):
                if not self.is_barrier((i-1,j)): 
                    dr.update({'down':self.grid[i][j].get_down()})
            if self.is_valid_node_index(i+1,j):
                if not self.is_barrier((i+1,j)): 
                    dr.update({'up':self.grid[i][j].get_up()})

        return dr

    def set_value(self,state,value):
        i, j = state[0], state[1]
        self.grid[i][j].set_value(value)

    def get_value(self,state):
        i, j = state[0], state[1]
        return self.grid[i][j].get_value()

    def get_reward_for_action(self,state,action):
        i, j = state[0], state[1]
        if action == 'left':
            return self.grid[i][j].get_left()
        if action == 'right':
            return self.grid[i][j].get_right()
        if action == 'down':
            return self.grid[i][j].get_down()
        if action == 'up':
            return self.grid[i][j].get_up()

    def get_value_at_destination(self,state,action):
        i, j = state[0], state[1]
        if action == 'left':
            return self.get_value((i,j-1))
        if action == 'right':
            return self.get_value((i,j+1))
        if action == 'down':
            return self.get_value((i-1,j))
        if action == 'up':
            return self.get_value((i+1,j))

    def print_grid_states_decisions(self):
        for i in range(0,self.M):
            for j in range(0,self.N):
                state = (i,j)
                print("")
                print(state)
                print("barrier node:", self.is_barrier(state))
                print("terminal node:", self.is_terminal(state))
                dr = self.valid_decisions_and_rewards(state)    
                print(dr)

    def print_values(self):
        for i in range(self.M-1,-1,-1):
            print("-",end="")
            for j in range(0,self.N):
                print("---------",end="")
            print("")
            print("|",end="")
            for j in range(0,self.N):
                state = (i,j)
                v = self.get_value(state)
                print(" %6.2f |" % v, end="") 
            print("")
        print("-",end="")
        for j in range(0,self.N):
            print("---------",end="")
        print("")
###
    def print_values_(self,values):
        for i in range(self.M-1,-1,-1):
            print("-",end="")
            for j in range(0,self.N):
                print("---------",end="")
            print("")
            print("|",end="")
            for j in range(0,self.N):
                state = (i,j)
                v = values[state]
                print(" %6.2f |" % v, end="") 
            print("")
        print("-",end="")
        for j in range(0,self.N):
            print("---------",end="")
        print("")   
###
    def print_policy(self,policy):
        for i in range(self.M-1,-1,-1):
            print("-", end="")
            for j in range(0, self.N):
                print("---------", end="")
            print("")
            print("|",end="")
            for j in range(0,self.N):
                p = policy[(i,j)]
                if p == 'left':
                    a = 'Left'
                elif p == 'right':
                    a = 'Right'
                elif p == 'down':
                    a = 'Down'
                elif p == 'up':
                    a = 'Up'
                else:
                    a = ' '
                print(" %6s |" % a, end="")        
            print("")
        print("-", end="")
        for j in range(0, self.N):
            print("---------", end="")
        print("") 
