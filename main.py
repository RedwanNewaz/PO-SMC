#!/usr/bin/env python
from __future__ import print_function
from pomdpy.solvers import POMCPV
from pomdpy.log import init_logger
from domain.navigation import GridPosition
from domain.navigation import RobotModel
from domain.navigation import ModelChecker
import numpy as np
import random
from ruamel_yaml import YAML
#########################################################################################################
import time
from pomdpy.pomdp.history import Histories, HistoryEntry
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import os




class AgentSMC:
    def __init__(self,model,solver):
        self.model = model
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset
        self.histories = Histories()

def state_visualization(self, robot):

    print("#"*int(2*self.model.n_cols+3) )
    for y in range(self.model.n_rows):
        for x in range(self.model.n_cols):
            c = self.model.map_text[y][x]
            if(robot.i==y and robot.j ==x):
                print("R",end="\t")
            elif(self.model.is_obstacle(GridPosition(y,x))):
                print("0", end="\t")
            else:
                print(".",end="\t" )
        print("")
    print("#" * int(2*self.model.n_cols + 3))
    print("\n")

def shuffle_obstacles(self,i):
    num_obstacles = len(self.model.obstacle_positions)
    self.model.obstacle_positions = []
    row = self.model.n_rows-1
    col = self.model.n_cols
    total_cells = np.arange(1,row*col).tolist()
    next_locs =random.sample(total_cells,num_obstacles)
    OLOGFILE = defaultdict(list)
    pos = lambda x: [x % col, x // col]
    for _loc in next_locs:
        x,y = pos(_loc)
        OLOGFILE['X_%d' % i].append(x)
        OLOGFILE['Y_%d' % i].append(y)
        p = GridPosition(y, x)
        self.model.obstacle_positions.append(p)
    dtf = pd.DataFrame(OLOGFILE)
    dtf.to_csv(EXP_LOG+'OBSTACLE_%d'%i+'.csv')


def dicounted_return(self):
    eps = self.model.epsilon_start
    self.model.reset_for_epoch()
    start_new_timer =True
    for i in tqdm(range(self.model.n_epochs)):
        solver = self.solver_factory(self)
        state = solver.belief_tree_index.sample_particle()
        reward = 0
        safety_property = 0
        discounted_reward = 0
        discount = 1.0
        if(start_new_timer):
            begin_time = time.time()
            start_new_timer = False
        # shuffle_obstacles(self, i)
        LOGFILE = defaultdict(list)
        LOGFILE['X_%d'%i].append(state.position.j)
        LOGFILE['Y_%d' % i].append(state.position.i)
        traj = []
        for _ in range(self.model.max_steps):
            start_time = time.time()
            state_visualization(self, state.position)
            action = solver.select_eps_greedy_action(eps, start_time)
            step_result, is_legal = self.model.generate_step(state, action)

            # print(f"reward {step_result.reward}, is_legal {is_legal}, obs {step_result.observation} ")
            #STEP update properties and reward
            safety_property += 1 if step_result.observation.is_obstacle or  not is_legal else 0
            reward += step_result.reward
            discounted_reward += discount * step_result.reward
            discount *= self.model.discount
            state = step_result.next_state
            LOGFILE['X_%d' % i].append(state.position.i)
            LOGFILE['Y_%d' % i].append(state.position.j)
            traj.append(state.position)

            # STEP model update
            if not step_result.is_terminal or not is_legal or not self.model.is_obstacle(state.position):
                solver.update(step_result)

            # STEP Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                              step_result.action, step_result.observation, step_result.next_state)


            # STEP termination
            if step_result.is_terminal or not is_legal:
                print('Terminated after episode step ' + str(i + 1))
                break
        # STEP- find out the illegal actions given the current state?
        elapsed_time = time.time() - begin_time
        print(f" time {elapsed_time:3.3} state {state.position} reward {reward}, prob {safety_property/self.model.max_steps:.3} count {safety_property}")
        self.model.reset_for_epoch()
        #STEP perform model checking
        y_size = self.model.n_rows
        map_index = lambda pos:pos.i*y_size + pos.j
        start_new_timer =ModelChecker.check_safe_reachability(map(map_index,traj), map(map_index,self.model.obstacle_positions),map(map_index,self.model.goal_positions))
        # STEP log writing
        dtf = pd.DataFrame(LOGFILE)
        f = open(EXP_LOG+'TRAJ_%d'%i+'.csv', 'w')
        f.write('# \t seed %d\n'%args['seed'])
        f.write('# \t elapsed time {}\n'.format(elapsed_time))
        f.write('# \t reward {}\n'.format(reward))
        f.write('# \t discounted reward {}\n'.format(discounted_reward))
        f.write('# \t valid policy {}\n'.format(start_new_timer))
        dtf.to_csv(f)
        f.close()


if __name__ == '__main__':

    yaml = YAML()
    cwd = os.getcwd()
    with open('/home/exp_param.yml','r') as param:
        args=yaml.load(param)


    init_logger()
    np.random.seed(args['seed'])
    random.seed(a=args['seed'])
    solver = POMCPV
    env = RobotModel(args)
    agent = AgentSMC(env, solver)
    EXP_LOG = "/home/experiments/yue_ral/ral-4-4-3/"
    if(not os.path.exists(EXP_LOG)):
        os.makedirs(EXP_LOG)




    dicounted_return(agent)









