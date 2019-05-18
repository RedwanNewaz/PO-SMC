from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
from pomdpy.pomdp import HistoricalData
from .robot_action import ActionType,RobotAction
import logging
from .policy_visualization import DataClass

class CellData(object):
    """
    Stores data about each cell in the grid
    """

    def __init__(self):
        pass


class PositionAndCellData(HistoricalData):
    """
    A class to store the robot position associated with a given belief node, as well as
    explicitly calculated probabilities of each cell state.
    """

    def __init__(self, model, grid_position, all_cell_data, solver):
        self.model = model
        self.solver = solver
        self.grid_position = grid_position
        # List of RockData indexed by the rock number
        self.all_cell_data = all_cell_data
        # Holds reference to the function for generating legal actions
        self.legal_actions = self.generate_legal_actions




    def copy(self):
        """
        Default behavior is to return a shallow copy
        """
        return self.shallow_copy()

    def deep_copy(self):
        """
        Passes along a reference to the rock data to the new copy of RockPositionHistory
        """
        return PositionAndCellData(self.model, self.grid_position.copy(), self.all_cell_data, self.solver)

    def shallow_copy(self):
        """
        Creates a copy of this object's rock data to pass along to the new copy
        """

        return PositionAndCellData(self.model, self.grid_position.copy(), [], self.solver)

    def update(self, other_belief):
        self.all_cell_data = other_belief.data.all_cell_data



    def create_child(self, robot_action, robot_observation):
        next_data = self.deep_copy()
        next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), robot_action.bin_number)
        next_data.grid_position = next_position
        return next_data

    def generate_legal_actions(self):

        legal_actions = []
        for act in range(0, 4):
            action = RobotAction(act)
            # which action is legal for a given state
            new_pos = action.make_adjacent_position(self.grid_position.copy(), action.bin_number)
            if self.model.is_valid_pos(new_pos):
                legal_actions.append(action)

        return legal_actions

