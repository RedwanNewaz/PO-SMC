from pomdpy.solvers import POMCPV
from pomdpy.log import init_logger
from domain.navigation import GridPosition
from domain.navigation.robot_model_v2 import RobotModel
from domain.navigation import ModelChecker
import numpy as np
import random
from ruamel_yaml import YAML
import time
from pomdpy.pomdp.history import Histories, HistoryEntry
#########################################################################################################
from pomdpy.config import MapStat
from collections import namedtuple
import pandas as pd
from collections import defaultdict
from threading import Thread
from queue import Queue
from tqdm import tqdm
import argparse





def dicounted_return(self,thread_id,target):
    eps = self.model.epsilon_start
    self.model.reset_for_epoch()
    start_new_timer =True
    epochs = self.model.n_epochs
    for i in range(epochs):
        solver = self.solver_factory(self)
        state = solver.belief_tree_index.sample_particle()
        reward = 0
        safety_property = 0
        discounted_reward = 0
        discount = 1.0
        if(start_new_timer):
            begin_time = time.time()
            start_new_timer = False
        traj = []
        for _ in range(self.model.max_steps):
            start_time = time.time()
            action = solver.select_eps_greedy_action(eps, start_time)
            step_result, is_legal = self.model.generate_step(state, action)
            #STEP update properties and reward
            safety_property += 1 if step_result.observation.is_obstacle or  not is_legal else 0
            reward += step_result.reward
            discounted_reward += discount * step_result.reward
            discount *= self.model.discount
            state = step_result.next_state
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
        # print(f" time {elapsed_time:3.3} state {state.position} reward {reward}, prob {safety_property/self.model.max_steps:.3} count {safety_property}")
        self.model.reset_for_epoch()
        #STEP perform model checking
        sat = False
        try:
            y_size = self.model.n_rows
            map_index = lambda pos: pos.i * y_size + pos.j
            sat = ModelChecker.check_safe_reachability(map(map_index, traj),
                                                      map(map_index, self.model.obstacle_positions),
                                                      map(map_index, self.model.goal_positions))
        except:
            print("exception for sat solver")
        finally:
            if((reward<=target and sat) or elapsed_time>TIMEOUT):
                # print(f"Target Found with {elapsed_time:.3f} sec")
                return {"time":elapsed_time,"sat":sat}




class AgentSMC:
    def __init__(self,model,solver):
        self.model = model
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset
        self.histories = Histories()

class myThread (Thread):
   def __init__(self, threadID,grid):
      Thread.__init__(self)
      self.threadID = threadID
      self.target = grid.reward
      self.threadName = "thread-" + grid.name
      solver = POMCPV
      yaml = YAML()
      with open('exp_param.yml', 'r') as param:
          args = yaml.load(param)
      res = MapStat(grid, confidence_interval=2.0)
      # args['n_start_states'] = res["particles"]
      args['min_particle_count'] = res["particles"]
      res = MapStat(grid, confidence_interval=1.0)
      args['max_particle_count'] = res["particles"]
      args['n_start_states'] = res["particles"]

      print("POPULATION SIZE %d SAMPLE SIZE: %d" % (res["num_states"], res["particles"]))
      # np.random.seed(args['seed'])
      # random.seed(a=args['seed'])

      env = RobotModel(args, grid)
      self.agent = AgentSMC(env, solver)
   def run(self):
      print("starting thread ",self.threadID)
      EXP_LOG = defaultdict(list)
      for t in tqdm(range(50)):
            res = dicounted_return(self.agent,self.threadID, self.target)
            EXP_LOG["elapsed_time"].append(res["time"])
            EXP_LOG["sat"].append(res["sat"])
            print(f"Thread:{self.threadID} | Target Found with {res['time']:.3f} sec")

      print("#"*10)
      dtf = pd.DataFrame(EXP_LOG)
      print("saving log ", self.threadName)
      logname = self.threadName.split(".")
      dtf.to_csv("experiments/AdaptiveSampling/%s_max_particles.csv"%logname[0])


if __name__ == '__main__':

    Grid = namedtuple("Grid","name obstacle size reward")
    Grid1 = Grid(name='ral-3-12-1.txt', obstacle=1, size=[4, 13], reward=-6)
    Grid2 = Grid(name='ral-3-12-2.txt', obstacle=2, size=[4, 13], reward=-14)
    Grid3 = Grid(name='ral-3-12-3.txt', obstacle=3, size=[4, 13], reward=-31)
    Grid6 = Grid(name='map-14-4-6.txt', obstacle=6, size=[4, 14], reward=-47)
    Grid7 = Grid(name='map-14-4-7.txt', obstacle=7, size=[4, 14], reward=-52)

    GRIDS = {1:Grid1,2:Grid2,3:Grid3,6:Grid6,7:Grid7}
    parser = argparse.ArgumentParser(description='Num of obstacles')
    parser.add_argument('--obs', type=int, help = 'Num of obstacles 1-3,6,7')
    parser.add_argument('--timeout', type=int, default=1000, help='Timeout for epochs')
    arg = parser.parse_args()
    TIMEOUT = arg.timeout
    _grid = GRIDS.get(arg.obs,-1)
    assert _grid != -1, 'Num of obstacles 1-3,6,7'
    process =myThread(_grid.obstacle, grid=_grid)
    process.run()


    # threads = (myThread(i,grid=G) for i, G in enumerate(GRIDS))
    # for t in threads:
    #     t.start()
    # for t in threads:
    #     t.join()




