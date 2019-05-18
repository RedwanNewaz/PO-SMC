from pomdpy.discrete_pomdp import DiscreteState

class RSCellType:
    """
    Rocks are enumerated 0, 1, 2, ...
    other cell types should be negative.
    """
    START = 1
    OBSTACLE = 0
    CAUTION = -1
    FREE =-2
    GOAL = -3



class CellState(DiscreteState):
    """
    The state contains the position of the robot, as well as a boolean value for each cell
    representing whether it is obstacle, caution, free .

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, grid_position, cell_states):
        if cell_states is not None:
            assert cell_states.__len__() is not 0
        self.position = grid_position
        self.cell_states = cell_states  # list

    def __copy__(self):
        return CellState(self.position,self.cell_states)
    def copy(self):
        return CellState(self.position, self.cell_states)

    def as_list(self):
        """
        Returns a list containing the (i,j) grid position indicating values
        representing the cell states (obstacle, caution, free)
        :return:
        """
        state_list = [self.position.i, self.position.j]
        for i in range(0, self.cell_states.__len__()):
            if self.cell_states[i] == RSCellType.OBSTACLE:
                state_list.append(RSCellType.OBSTACLE)
            elif self.cell_states == RSCellType.CAUTION:
                state_list.append(RSCellType.CAUTION)
            else:
                state_list.append(RSCellType.FREE)
        return state_list

    def separate_cells(self):
        """
        Used for the PyGame sim
        :return:
        """
        free_cells = []
        obs_cells = []
        caution_cells = []
        for i in range(0, self.cell_states.__len__()):
            if self.cell_states[i]==RSCellType.OBSTACLE:
                obs_cells.append(i)
            elif self.cell_states==RSCellType.CAUTION:
                caution_cells.append(i)
            else:
                free_cells.append(i)
        return free_cells, obs_cells, caution_cells

    def print_state(self):
        pass

    def to_string(self):
        data =self.as_list()
        state=[]
        for d in data:
            if d == RSCellType.OBSTACLE:
                state.append("OBS")
            elif d == RSCellType.CAUTION:
                state.append("CAUTION")
            else:
                state.append("FREE")

        return "{}".format(state)
