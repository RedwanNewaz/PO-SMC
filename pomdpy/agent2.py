from __future__ import print_function, division
import time
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider
from experiments.scripts.pickle_wrapper import save_pkl
import pandas as pd
import numpy as np
from collections import defaultdict


module = "agent"


class Agent:
    """
    Train and store experimental results for different types of runs

    """


    # fetcher.start()
    # plt.show()

    def __init__(self, model, solver):
        """
        Initialize the POMDPY agent
        :param model:
        :param solver:
        :return:
        """
        self.logger = logging.getLogger('POMDPy.Solver')
        self.model = model
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver_factory = solver.reset  # Factory method for generating instances of the solver
        self.policy = defaultdict(list)
        self.policy ['reward']=-np.inf


        # plt.show()

    def discounted_return(self):

        self.multi_epoch()

        print('\n')
        console(2, module, 'epochs: ' + str(self.model.n_epochs))
        console(2, module, 'ave undiscounted return/step: ' + str(self.experiment_results.undiscounted_return.mean) +
                ' +- ' + str(self.experiment_results.undiscounted_return.std_err()))
        console(2, module, 'ave discounted return/step: ' + str(self.experiment_results.discounted_return.mean) +
                ' +- ' + str(self.experiment_results.discounted_return.std_err()))
        # console(2, module, 'ave time/epoch: ' + str(self.experiment_results.time.mean))

        self.logger.info('env: ' + self.model.env + '\t' +
                         'epochs: ' + str(self.model.n_epochs) + '\t' +
                         'ave undiscounted return: ' + str(self.experiment_results.undiscounted_return.mean) + ' +- ' +
                         str(self.experiment_results.undiscounted_return.std_err()) + '\t' +
                         'ave discounted return: ' + str(self.experiment_results.discounted_return.mean) +
                         ' +- ' + str(self.experiment_results.discounted_return.std_err()) +
                         '\t' + 'ave time/epoch: ' + str(self.experiment_results.time.mean))

        return self.policy['optimal_traj']
    def multi_epoch(self):
        eps = self.model.epsilon_start

        self.model.reset_for_epoch()
        summary = defaultdict(list)
        elapsed_time = 0
        for i in range(self.model.n_epochs):
            # Reset the epoch stats
            self.results = Results()

            eps = self.run_pomcp(i + 1, eps)
            self.model.reset_for_epoch()
            summary['discounted_return'].append(self.experiment_results.discounted_return.mean)
            summary['undiscounted_return'].append(self.experiment_results.undiscounted_return.mean)
            elapsed_time+=self.experiment_results.time.running_total
            summary['elapsed_time'].append(elapsed_time)
            summary['num_obs'].append(len(self.model.invalid_cells))
            summary['is_sat'].append(self.model.is_sat)
            # store optimal policy
            if self.model.is_sat and (self.experiment_results.discounted_return.mean>self.policy ['reward']):
                self.policy['optimal_traj']=self.policy['traj']
                self.policy['reward'] = self.experiment_results.discounted_return.mean


            if self.experiment_results.time.running_total > self.model.timeout:
                console(2, module, 'Timed out after ' + str(i) + ' epochs in ' +
                        "{}".format(self.experiment_results.time.running_total) + ' seconds')
                break

        # default dict to panda dataframe conversion
        print("saving log ",'.'*50)
        dtf = pd.DataFrame(summary)
        dtf.to_csv('results/{}.csv'.format(self.model.log_name))




    def run_pomcp(self, epoch, eps):
        epoch_start = time.time()

        # Create a new solver
        solver = self.solver_factory(self)

        # Monte-Carlo start state
        state = solver.belief_tree_index.sample_particle()

        reward = 0
        discounted_reward = 0
        discount = 1.0
        TRAJECTORY = []
        for i in range(self.model.max_steps):

            start_time = time.time()

            # action will be of type Discrete Action

            action = solver.select_eps_greedy_action(eps, start_time)

            # update epsilon
            if eps > self.model.epsilon_minimum:
                eps *= self.model.epsilon_decay

            step_result, is_legal = self.model.generate_step(state, action)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward

            discount *= self.model.discount
            state = step_result.next_state
            # print('next state chosen ',state.position)
            TRAJECTORY.append(state.position)
            # show the step result
            self.display_step_result(i, step_result)

            if not step_result.is_terminal or not is_legal:
                solver.update(step_result)

            # Extend the history sequence
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, step_result.reward,
                                              step_result.action, step_result.observation, step_result.next_state)

            if step_result.is_terminal or not is_legal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

        self.results.time.add(time.time() - epoch_start)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        # visualize trajecory
        # self.model.viz_thread.trajectory(TRAJECTORY)
        # print_divider('large')
        solver.history.show()
        self.results.show(epoch)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')
        print("obstacle dict", self.model.invalid_cells)
        print("num states", self.model.num_states)
        print_divider('medium')

        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)
        # ================================================================================================================================
        # update policy information
        self.policy['traj'] = TRAJECTORY

        return eps

    def run_value_iteration(self, solver, epoch):
        run_start_time = time.time()

        reward = 0
        discounted_reward = 0
        discount = 1.0

        solver.value_iteration(self.model.get_transition_matrix(),
                               self.model.get_observation_matrix(),
                               self.model.get_reward_matrix(),
                               self.model.planning_horizon)

        b = self.model.get_initial_belief_state()

        for i in range(self.model.max_steps):

            # TODO: record average V(b) per epoch
            action, v_b = solver.select_action(b, solver.gamma)

            step_result = self.model.generate_step(action)

            if not step_result.is_terminal:
                b = self.model.belief_update(b, action, step_result.observation)

            reward += step_result.reward
            discounted_reward += discount * step_result.reward
            discount *= self.model.discount

            # show the step result
            self.display_step_result(i, step_result)

            if step_result.is_terminal:
                console(3, module, 'Terminated after episode step ' + str(i + 1))
                break

            # TODO: add belief state History sequence

        self.results.time.add(time.time() - run_start_time)
        self.results.update_reward_results(reward, discounted_reward)

        # Pretty Print results
        self.results.show(epoch)
        console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        print_divider('medium')


        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

    @staticmethod
    def display_step_result(step_num, step_result):
        """
        Pretty prints step result information
        :param step_num:
        :param step_result:
        :return:
        """

        console(3, module, 'Step Number = ' + str(step_num))
        console(3, module, 'Step Result.Action = ' + step_result.action.to_string())
        console(3, module, 'Step Result.Observation = ' + step_result.observation.to_string())
        console(3, module, 'Step Result.Next_State = ' + step_result.next_state.to_string())
        console(3, module, 'Step Result.Reward = ' + str(step_result.reward))



class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, epoch):
        print_divider('large')
        print('\tEpoch #' + str(epoch) + ' RESULTS')
        print_divider('large')
        console(2, module, 'discounted return statistics')
        print_divider('medium')
        self.discounted_return.show()
        print_divider('medium')
        console(2, module, 'undiscounted return statistics')
        print_divider('medium')
        self.undiscounted_return.show()
        # print_divider('medium')
        # console(2, module, 'Time')
        # print_divider('medium')
        # self.time.show()
        # print_divider('medium')
