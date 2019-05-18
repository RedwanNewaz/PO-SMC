from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import map
from builtins import hex
from builtins import range
from past.utils import old_div
import logging
import json
import numpy as np
from pomdpy.util import console, config_parser_robot
from .grid_position import GridPosition
from .robot_action import RobotAction,ActionType
from .cell_state import CellState
from .robot_position_history import CellData,PositionAndCellData
from .robot_observation import RobotObservation
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.pomdp import Model, StepResult
import operator as op
from functools import reduce
from pomdpy.config import ncr

module = "RobotModel"


# logging.basicConfig(level=logging.DEBUG)

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




class RobotModel(Model):
    def __init__(self, args,grid):
        super(RobotModel, self).__init__(args)
        # logging utility
        # self.logger = logging.getLogger('POMDPy.RobotModel')
        self.robot_config = json.load(open(config_parser_robot.robot_cfg, "r"))
        self.log_name = args['log_name']
        self.use_smc = args['use_smc']

        self.start_position = GridPosition()
        # The coordinates of the goal squares.
        self.goal_positions = []
        self.env_map = []
        # Smart cell data -  obs free cells
        self.all_cell_data = []

        # Actual cell states
        self.actual_cell_states = []
        self.step_positions =[]
        self.step_rewards = []

        # The environment map in text form.
        self.map_text, dimensions = config_parser_robot.parse_map(grid.name)
        self.half_efficiency_distance = 20.0
        self.n_rows = int(dimensions[1])
        self.n_cols = int(dimensions[0])
        print('robot model initialize','.'*60)
        self.initialize()
        assert grid.obstacle == self.num_obs_cells, "obstacle size mismatch"





    # initialize the maps of the grid
    def initialize(self):
        # declare some new variables
        self.num_caution_cells = 0
        self.num_obs_cells = 0
        self.num_free_cells = 0
        self.obstacle_positions =[]
        # ground_truth = self.viz_data[DataClass.GT]
        p = GridPosition()
        for p.i in range(0, self.n_rows):
            tmp = []
            for p.j in range(0, self.n_cols):
                c = self.map_text[p.i][p.j]

                # initialized to empty
                cell_type = RSCellType.FREE

                if c is 'C':
                    cell_type = RSCellType.CAUTION
                    self.num_caution_cells += 1
                elif c is 'G':
                    cell_type = RSCellType.GOAL
                    self.goal_positions.append(p.copy())
                elif c is 'S':
                    self.start_position = p.copy()
                    cell_type = RSCellType.START
                elif c is 'X':
                    cell_type = RSCellType.OBSTACLE
                    self.num_obs_cells += 1
                    self.obstacle_positions.append(p.copy())
                else:
                    self.num_free_cells+=1
                tmp.append(cell_type)
                # ground_truth[p.i][p.j] =cell_type

            self.env_map.append(tmp)
        # Total number of distinct states

        self.num_states = ncr(self.n_rows * self.n_cols,
                              self.num_obs_cells) * self.n_rows * self.n_cols

        print ("num_state {}".format(self.num_states))

    ''' ===================================================================  '''
    '''                             Utility functions                        '''
    ''' ===================================================================  '''

    # returns the RSCellType at the specified position
    def get_cell_type(self, pos):
        return self.env_map[pos.i][pos.j]

    def get_sensor_correctness_probability(self, distance):
        assert self.half_efficiency_distance is not 0, self.logger.warning("Tried to divide by 0! Naughty naughty!")
        return (1 + np.power(2.0, old_div(-distance, self.half_efficiency_distance))) * 0.5

    ''' ===================================================================  '''
    '''                             Sampling                                 '''
    ''' ===================================================================  '''

    def sample_an_init_state(self):
        #TODO modify the intial state
        # init_states = [1<<self.num_obs_cells for _ in range(self.num_obs_cells)]
        # return CellState(self.start_position, init_states)
        return CellState(self.start_position, self.sample_cells())

    def sample_state_uninformed(self):
        while True:
            pos = self.sample_position()
            if self.get_cell_type(pos) is not RSCellType.OBSTACLE:
                return CellState(pos, self.sample_cells())

    def sample_state_informed(self, belief):
        return belief.sample_particle()

    def sample_position(self):
        i = np.random.random_integers(0, self.n_rows - 1)
        j = np.random.random_integers(0, self.n_cols - 1)
        return GridPosition(j, i)

    def sample_cells(self):
        # the intuition behind is that we need evaluate those cells which are not safe or need to avoid
        return self.decode_cells(np.random.random_integers(0, self.num_states))
        # return self.decode_cells(0)

    def decode_cells(self, value):
        avoid_states = []
        for i in range(0, self.num_obs_cells):
            avoid_states.append(value & (1 << i))
        return avoid_states

    def encode_cells(self, cell_states):
        value = 0
        for i in range(0, self.num_obs_cells):
            if cell_states[i]:
                value += (1 << i)
        return value

    ''' ===================================================================  '''
    '''                 Implementation of abstract Model class               '''
    ''' ===================================================================  '''

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def is_terminal(self, rock_state):
        return self.get_cell_type(rock_state.position) is RSCellType.GOAL

    def is_valid(self, state):
        if isinstance(state, CellState):
            return self.is_valid_pos(state.position)
        if isinstance(state, GridPosition):
            return self.is_valid_pos(state)
        else:
            return False



    def is_valid_pos(self, pos):
        return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols
        # return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols and \
        #        self.get_cell_type(pos) is not RSCellType.OBSTACLE

    def get_legal_actions(self, state):
        legal_actions = []

        for action in range(0, 4 ):
            # STEP which move actions are legal for a given state
            new_pos = state.position.copy()
            _ , is_legal=self.make_next_position(new_pos,RobotAction(action))
            if is_legal:
                legal_actions.append(RobotAction(action))
        #TODO add look actions - all are them are leagal
        for action in range(4,8):
            # STEP all look actions are leagal
            legal_actions.append(RobotAction(action))

        return legal_actions

    def get_max_undiscounted_return(self):
        pass


    def reset_for_simulation(self):
        #STEP check constraint satisfaction
        pass


    def reset_for_epoch(self):
        self.actual_cell_states = self.sample_cells()
        console(2, module, "Actual cell states = " + str(self.actual_cell_states))

    def update(self, step_result):
        self.step_positions.append(step_result.next_state.position)
        self.step_rewards.append(step_result.reward)




    def get_all_states(self):
        """
        :return: Forgo returning all states to save memory, return the number of states as 2nd arg
        """
        return None, self.num_states

    def get_all_observations(self):
        """
        :return: Return a dictionary of all observations and the number of observations
        """
        return {
            "free": 0,
            "obstacle": 1,
            "caution": 2,
        }, 3

    def get_all_actions(self):
        """
        :return: Return a list of all actions along with the length
        """
        #TODO increase the number of actions from 4 to 8
        all_actions = []
        num_actions = 8
        for code in range(0, num_actions):
            all_actions.append(RobotAction(code))
        return all_actions

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, solver):
        self.all_cell_data= [CellData() for _ in range(self.num_obs_cells)]
        return PositionAndCellData(self, self.start_position.copy(), self.all_cell_data, solver)



    def make_next_position(self, pos, action):
        if type(action) is int:
            action = RobotAction(action)
        if not isinstance(action,RobotAction):
            action = RobotAction(action.bin_number)
            # print("error")
        new_pos = action.make_adjacent_position(pos.copy(), action.bin_number)
        # any action that does not cause outside of the grid or bump into the obstacles is legal
        #STEP all look operations are legal
        is_legal = self.is_valid_pos(new_pos) if action.bin_number<4 else True
        if is_legal:
            pos = new_pos.copy()

        logging.debug("{} illegal moving action {} is performed for transition {}-> {}".
                format(int(is_legal), action, pos.to_string(),new_pos.to_string()))

        return pos, is_legal

    def make_next_state(self, state, action):
        next_position, is_legal = self.make_next_position(state.position, action)

        if not is_legal:
            # returns a copy of the current state
            return state.copy(), False

        next_cell_states = list(state.cell_states)

        return CellState(next_position, next_cell_states), True

    def make_observation(self, action, next_state):

        # if checking a cell...
        def min_dist_obstacle(s):
            dist = (o.manhattan_distance(s.position) for o in self.obstacle_positions)
            return min(dist)
        # dist = next_state.position.manhattan_distance(self.obstacle_positions[action.bin_number])
        dist = min_dist_obstacle(next_state)

        # STEP NOISY OBSERVATION
        # bernoulli distribution is a binomial distribution with n = 1
        # if half efficiency distance is 20, and distance to rock is 20, correct has a 50/50
        # chance of being True. If distance is 0, correct has a 100% chance of being True.
        correct = np.random.binomial(1.0, self.get_sensor_correctness_probability(dist))

        cell_state = self.get_cell_type(next_state.position)
        if not correct:
            # Return the incorrect state if the sensors malfunctioned
            # default behavior is
            cell_state = RSCellType.FREE

        if self.is_terminal(next_state):
            # the robot has no uncertainty at goal location and it can localize it perfectly
            cell_state = RSCellType.GOAL
# STEP update cell state for look actions
        next_state.cell_states[action.bin_number%self.num_obs_cells] = cell_state


        obs = RobotObservation(cell_state)
        logging.debug("observe - {}".format(obs))
        return obs

    def belief_update(self, old_belief, action, observation):
        print("belief updating ")
        pass

    # def is_obstacle(self,s):
    #     i, j = s.position.i, s.position.j
    #     return self.env_map[i][j] == RSCellType.OBSTACLE
    def is_obstacle(self,p):
        for o in self.obstacle_positions:
            if(o==p):
                return True
        return False

    def make_reward(self, state, action, next_state, is_legal):
        # STEP the robot to avoid certain regions as much as possible by assigning
        #  the reward of -10 for states and a reward of -1 for each action.


        def min_dist_goal(s):
            dist = (g.manhattan_distance(s.position)for g in self.goal_positions)
            return min(dist)
        def is_caution(s):
            i, j = s.position.i, s.position.j
            return self.env_map[i][j] == RSCellType.CAUTION

# FIXME play with reward
        GO_TO_COST = min_dist_goal(next_state)*0+1
        # GO_TO_COST = 0
        TERMINAL_REWARD = +100
        PENALTY_COST = -100
        CAUTION_COST = -10
        if not is_legal or self.is_obstacle(next_state.position):
            return PENALTY_COST
        elif self.is_terminal(next_state):
            return TERMINAL_REWARD
        elif is_caution(next_state):
            return CAUTION_COST
        else:
            return -GO_TO_COST

    def generate_reward(self, state, action):
        next_state, is_legal = self.make_next_state(state, action)
        return self.make_reward(state, action, next_state, is_legal)

    def generate_step(self, state, action):
        # type: (object, object) -> object
        if type(action) is int:
            action = RobotAction(action)
        elif action is None:
            logging.error("Tried to generate a step with a null action")
            return None

        result = StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state, is_legal)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, is_legal

    # def generate_particles_uninformed(self, previous_belief, action, obs, n_particles):
    #     old_pos = previous_belief.get_states()[0].position
    #
    #     particles = []
    #     while particles.__len__() < n_particles:
    #         old_state = CellState(old_pos, self.sample_rocks())
    #         result, is_legal = self.generate_step(old_state, action)
    #         if obs == result.observation:
    #             particles.append(result.next_state)
    #     return particles

    @staticmethod
    def disp_cell(rs_cell_type):
        # if rs_cell_type >= RSCellType.OBSTACLE:
        #     print(hex(rs_cell_type - RSCellType.OBSTACLE), end=' ')
        #     return

        if rs_cell_type is RSCellType.FREE:
            print('.\t', end=' ')
        elif rs_cell_type is RSCellType.GOAL:
            print('G\t', end=' ')
        elif rs_cell_type is RSCellType.START:
            print('S\t', end=' ')
        elif rs_cell_type is RSCellType.CAUTION:
            print('C\t', end=' ')
        elif rs_cell_type is RSCellType.OBSTACLE:
            print('X\t', end=' ')
        else:
            print('ERROR-', end=' ')
            print(rs_cell_type, end=' ')

    def draw_env(self):
        for row in self.env_map:
            list(map(self.disp_cell, row))
            print('\n')




