import time

import numpy as np
import tensorflow as tf


class Stats:
    def __init__(self, log_loc):
        self.writer = tf.summary.create_file_writer(logdir=log_loc)

        # Store results of every episode
        self.n_actions_list = []
        self.reward_list = []

        # Initialize variables for recording results
        self.reset()

        # Initialize timer 
        self.curr_save_time = time.time()

    def log_action(self, reward):
        # Add to total actions & rewards every step
        self.ep_n_actions += 1
        self.ep_reward    += reward

    def log_ep(self):
        # Append n_actions and rewards to respective lists after every ep
        self.n_actions_list.append(self.ep_n_actions)
        self.reward_list.append(self.ep_reward)
        self.reset()

    def save(self, ep, ep_interval, epsilon, step):
        # Compute and save results to file every "step" 
        avg_actions = np.round(np.mean(self.n_actions_list[-step:]), 5)
        avg_reward  = np.round(np.mean(self.reward_list[-step:]), 5)
        prev_save_time = self.curr_save_time
        self.curr_save_time = time.time()
        delta_time = np.round(self.curr_save_time - prev_save_time, 5)
        epsilon = np.round(epsilon, 5)

        display_results(ep, ep_interval, delta_time, avg_actions, avg_reward, epsilon)
        
        # Save results to tensorboard
        with self.writer.as_default():
            tf.summary.scalar(name='avg. actions', data=avg_actions, step=step)
            tf.summary.scalar(name='avg. reward', data=avg_reward, step=step)
            # Make graphs for avg. actions and avg. rewards
            self.writer.flush()

    def reset(self):
        self.ep_reward    = 0.0
        self.ep_n_actions = 0.0

def display_results(ep, ep_interval, delta_time, avg_actions, avg_reward, epsilon):
        print(
            f'\n~~~~~~~~~~ EPISODE DETAILS ~~~~~~~~~~\n'
            f'Episode Number:          {ep}\n'
            f'{ep_interval} Episode Duration (s): {delta_time}\n'
            f'Current Epsilon Value:   {epsilon}\n'
            f'Average Actions/Episode: {avg_actions}\n'
            f'Average Reward/Episode:  {avg_reward}\n'
        )