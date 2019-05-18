from __future__ import print_function
from pomdpy.discrete_pomdp import DiscreteObservation

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



class RobotObservation(DiscreteObservation):
    """
    Default behavior is for the robot observation to say that the cell is free
    for invalid observation both obstacle and caution should true
    """
    def __init__(self, bin_number):
        super(RobotObservation, self).__init__(bin_number)
        self.bin_number = bin_number
        self.is_obstacle = True if bin_number is RSCellType.OBSTACLE else False
        self.is_caution = True if bin_number is RSCellType.CAUTION else False
        self.is_goal = True if bin_number is RSCellType.GOAL else False
        self.is_free = True if bin_number is RSCellType.FREE else False

    def distance_to(self, other_rock_observation):
        return abs(self.is_obstacle - other_rock_observation.is_obstacle)

    def copy(self):
        return RobotObservation(self.bin_number)

    def __eq__(self, other_rock_observation):
        return (self.bin_number == other_rock_observation.bin_number)

    def __hash__(self):
        return (False, True)[self.bin_number]

    def print_observation(self):

        if not self.is_obstacle and not self.is_caution:
            print("INVALID")
        elif self.is_obstacle:
            print("OBSTACLE")
        elif self.is_caution:
            print("CAUTION")
        else:
            print("FREE SPACE")

    def to_string(self):
        obs= "FREE_SPACE"
        if self.is_obstacle:
            obs = "OBSTACLE"
        elif self.is_caution:
            obs = "CAUTION"
        elif self.is_goal:
            obs = "GOAL"

        return obs

    def __repr__(self):
        return self.to_string()
