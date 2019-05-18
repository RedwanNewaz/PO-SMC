from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction



class ActionType(object):
    """
    Lists the possible actions and attributes an integer code.
    The pickup actins are not available before the last row.
    Each action is followed by observation - for moving east the robot needs to look east first
    """
    MOVE_NORTH = 0
    MOVE_EAST = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3

    LOOK_NORTH  = 4
    LOOK_EAST   = 5
    LOOK_SOUTH  = 6
    LOOK_WEST   = 7



class RobotAction(DiscreteAction):
    """
      -The move and pick up a cup problem Action class
      -Wrapper for storing the bin number. Also stores the obstacle number for checking actions
      -Handles pretty printing
      """

    def __init__(self, bin_number):
        super(RobotAction, self).__init__(bin_number)
        self.bin_number = bin_number


    @staticmethod
    def make_adjacent_position(pos, action_type):
        if action_type is ActionType.MOVE_NORTH:
            pos.i -= 1
        elif action_type is ActionType.MOVE_EAST:
            pos.j += 1
        elif action_type is ActionType.MOVE_SOUTH:
            pos.i += 1
        elif action_type is ActionType.MOVE_WEST:
            pos.j -= 1
        return pos


    def copy(self):
        return RobotAction(self.bin_number)

    def print_action(self):
        print(self)


    def to_string(self):

        if self.bin_number is ActionType.MOVE_NORTH:
            action = "NORTH"
        elif self.bin_number is ActionType.MOVE_EAST:
            action = "EAST"
        elif self.bin_number is ActionType.MOVE_SOUTH:
            action = "SOUTH"
        elif self.bin_number is ActionType.MOVE_WEST:
            action = "WEST"
        elif self.bin_number is ActionType.LOOK_NORTH:
            action = "LOOK_NORTH"
        elif self.bin_number is ActionType.LOOK_EAST:
            action = "LOOK_EAST"
        elif self.bin_number is ActionType.LOOK_SOUTH:
            action = "LOOK_SOUTH"
        elif self.bin_number is ActionType.LOOK_WEST:
            action = "LOOK_WEST"
        else:
            action = "UNDEFINED ACTION"
        return action

    def distance_to(self, other_point):
        pass

    def __repr__(self):
        return self.to_string()