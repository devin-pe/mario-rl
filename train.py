import argparse
import datetime
from os import path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import config as cfg
from custom_wrappers import FrameSkip
from record import Stats
from train_mario import TrainMario


# parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train Mario')
    parser.add_argument('--mode', dest='mode',
                        help='Learning mode - dqn or ddqn',
                        default='', type=str)

    args = parser.parse_args()
    return args

# make Mario learn the first level
def train():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='rgb_array')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # apply wrappers to env
    env = ResizeObservation(env, shape=84)  # wrapper for resizing image observations
    env = GrayScaleObservation(env)         # wrapper that converts a color observation to grayscale
    env = FrameSkip(env, skip=4)            # wrapper for skipping frames
    env = FrameStack(env, num_stack=4)      # observation wrapper that stacks the observations in a rolling manner

    wrapped_env_dims = env.observation_space.shape  # final wrapped env = (4, 84, 84)
    print(f'Wrapped environment dimensions are {wrapped_env_dims}')

    args = parse_args()

    dqn_or_ddqn = args.mode
    assert dqn_or_ddqn == 'dqn' or dqn_or_ddqn == 'ddqn', 'Invalid training mode selected'

    log_folder = path.join('training_results', dqn_or_ddqn + '-logs-' + datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    results = Stats(log_loc=log_folder)

    mario = TrainMario(state_space=wrapped_env_dims, action_space=env.action_space.n, log_loc=log_folder, learning_mode=dqn_or_ddqn)

    for episode in range(cfg.training_episodes):
        print(f'Starting episode {episode}')
        # upon reset, state is a pair on of ('gym.wrappers.frame_stack.LazyFrames', {})
        # we don't care about the {}, so we get rid of it right away
        state = env.reset()[0]
        done = False

        # loop until end of game
        while not done:
            action = mario.choose_action(state)                            # choose an action to perform
            new_state, reward, done, _, _ = env.step(action)               # perform the chosen action
            results.log_action(reward)                                     # log the reward and step values
            mario.store_in_memory(state, action, reward, new_state, done)  # store the results of the chosen action to memory
            mario.learn()                                                  # learn from the previously taken actions (update model based on sampled memories)
            state = new_state                                              # update current state

        results.log_ep()

        if episode % cfg.record_stats == 0:
            results.save(ep=episode, ep_interval=cfg.record_stats, epsilon=mario.epsilon, step=mario.total_steps_taken)


if __name__ == '__main__':
    train()