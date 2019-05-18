import z3 as smt
import numpy as np
from collections import defaultdict

class ModelChecker:

    def __init__(self, n_rows,n_cols,thres=0.92):
        self.n_rows = n_rows
        self.thres = thres
        # a cell status could be obs, free, caution
        self.BELIEF_OBS_MAP = None
        self.invalid_cells = defaultdict(list)
    @staticmethod
    def check_safe_reachability(traj,obs,goal):
        s = smt.Solver()
        # convert 2d traj to 1d array
        #convert 2d obstacle to 1d array
        # convert 2d goal to 1d array
        # x_i != o_j
        # x_n == g


        for i, x in enumerate(traj):
            x_i = smt.Real("x_%d" % i)
            s.add(x_i == x)
            for j, o in enumerate(obs):
                o_j = smt.Real("o_%d"%j)
                s.add(o_j == o)
                s.add(x_i != o_j)
        #STEP - is last node in the goal location?
        x_n = smt.Real("x_%d" % i)
        for g in goal:
            s.add(x_n==g)
        result = s.check()
        return result == smt.sat



    def check_safe_goal_constraint(self, traj):
        # check the step positions satisfy the constraints
        s = smt.Solver()
        for i, x in enumerate(traj):
            y, k = x.i, x.j
            z = self.BELIEF_OBS_MAP[y][k]
            z3_var_name = "state_{}".format(i)
            z3_var = smt.Real(z3_var_name)
            s.add(z3_var == z)
            s.add(z3_var < (1-self.thres)) # probability of collision should be less than 1-delta
        LAST_ELEMENT = traj[-1]
        # GOAL_LOCATIONS = [[1,12],[2,12]]
        y_smt_var = smt.Int('y')
        s.add(y_smt_var >= self.n_rows-2)
        s.add(y_smt_var == LAST_ELEMENT.i)
        result = s.check()
        # print(traj)
        print('LAST_ELEMENT', LAST_ELEMENT, 'MODEL', result)
        return result == smt.sat

    # def smc_obs_check(self,s):
    #     i, j = s.position.i, s.position.j
    #     if self.BELIEF_OBS_MAP[i][j] > self.thres:
    #         print("obstacles at next state", s.position, self.BELIEF_OBS_MAP[i][j])
    #         self.invalid_cells[s.position] = self.BELIEF_OBS_MAP[i][j]

    def __call__(self, *args, **kwargs):
        # belief and position
        self.BELIEF_OBS_MAP=np.copy(args[0])
        # self.smc_obs_check(args[1])

    # def next(self):
    #     return np.copy(self.invalid_cells)


    def __repr__(self):
        np.set_printoptions(precision=3)
        return "{}".format(self.BELIEF_OBS_MAP)
