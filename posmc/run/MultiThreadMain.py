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
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt



q = Queue()
finish = False

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

def dicounted_return(self,thread_id):
    eps = self.model.epsilon_start
    self.model.reset_for_epoch()
    start_new_timer =True
    for i in range(self.model.n_epochs):
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
        # LOGFILE = defaultdict(list)
        # LOGFILE['X_%d'%i].append(state.position.j)
        # LOGFILE['Y_%d' % i].append(state.position.i)
        traj = []
        for _ in range(self.model.max_steps):
            start_time = time.time()
            # state_visualization(self, state.position)
            action = solver.select_eps_greedy_action(eps, start_time)
            step_result, is_legal = self.model.generate_step(state, action)

            # print(f"reward {step_result.reward}, is_legal {is_legal}, obs {step_result.observation} ")
            #STEP update properties and reward
            safety_property += 1 if step_result.observation.is_obstacle or  not is_legal else 0
            reward += step_result.reward
            discounted_reward += discount * step_result.reward
            discount *= self.model.discount
            state = step_result.next_state
            # LOGFILE['X_%d' % i].append(state.position.i)
            # LOGFILE['Y_%d' % i].append(state.position.j)
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
        start_new_timer =ModelChecker.check_safe_reachability(map(map_index,traj),
                                                              map(map_index,self.model.obstacle_positions),map(map_index,self.model.goal_positions))
        if(start_new_timer):
            q.put({thread_id:reward})
        if(finish):
            break
    q.put(thread_id,None)

class myThread (Thread):
   def __init__(self, threadID):
      Thread.__init__(self)
      self.threadID = threadID
      solver = POMCPV
      yaml = YAML()
      with open('exp_param.yml', 'r') as param:
          args = yaml.load(param)
      args['ucb_coefficient'] = threadID
      np.random.seed(args['seed'])
      random.seed(a=args['seed'])

      env = RobotModel(args)
      self.agent = AgentSMC(env, solver)
   def run(self):
      print("starting thread ",self.threadID)
      dicounted_return(self.agent,self.threadID)

def plot(reward,t):
    print(f"updating plot at {elapsed_time:.3f}")
    # plt.hold(True)
    for key in reward:
        assert (key<6)
        # if(reward[key]<=-100):
        #     continue
        csv_history[key].append(reward[key])
    csv_history["time"].append(t)
    #     plt.plot(t,reward[key],colors[key])
    # plt.axis([0,1800,-110,10])
    # plt.pause(0.1)

if __name__ == '__main__':
    threads = []
    for i in range(6):
        threads.append(myThread(i))

    for i in range(6):
        threads[i].start()


    print("main thread configure ")
    colors = ["xr","xg","xb","or","og","ob"]


    rewards ={i:-50 for i in range(6)}
    start_time = time.time()
    count = 0
    elapsed_time = 0
    csv_history = defaultdict(list)
    while(elapsed_time <1200):
        if(not q.empty()):
            data = q.get()
            print("data recived ", data)
            for key in data:
                if(not data[key]): #STEP - break conditions
                    count +=1
                # elif(data[key]>rewards[key]):
                #     rewards[key]=data[key]
                #     dt = time.time() - start_time
                #     plot(rewards, dt)
                #     elapsed_time = time.time() - start_time
                else:
                    rewards[key] = data[key]
            elapsed_time = time.time() - start_time
            csv_history["time"].append(elapsed_time)
            print(f"updating plot at {elapsed_time:.3f}")
            for key in rewards:
                csv_history[key].append(rewards[key])

        else:
            time.sleep(0.5)
            elapsed_time = time.time() - start_time
            # print(f"elapsed time {elapsed_time:.3f}")

    print("saving log ...")
    dtf=pd.DataFrame(csv_history)
    dtf.to_csv("MultiThreads2.csv", index_label="time")
    finish = True
    time.sleep(2)
    print("Threads are joining...")
    for i in range(6):
        threads[i].join()
    print("program terminate...")













