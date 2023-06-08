import argparse
from os import path

import gym_super_mario_bros
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from custom_wrappers import FrameSkip
from test_mario import TestMario


# parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Test Mario')
    parser.add_argument('--log_folder', dest='log_folder',
                        help='Log folder within training_results to test',
                        default='', type=str)
    parser.add_argument('--weights', dest='weight_file',
                        default='', type=str)

    args = parser.parse_args()
    return args

# test Mario's ability to complete the first level
def test():

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode='human')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # apply wrappers to env
    env = ResizeObservation(env, shape=84)  # wrapper for resizing image observations
    env = GrayScaleObservation(env)         # wrapper that converts a color observation to grayscale
    env = FrameSkip(env, skip=4)            # wrapper for skipping frames
    env = FrameStack(env, num_stack=4)      # observation wrapper that stacks the observations in a rolling manner

    wrapped_env_dims = env.observation_space.shape  # final wrapped env = (4, 84, 84)
    print(f'Wrapped environment dimensions are {wrapped_env_dims}')

    args = parse_args()
    log_folder  = path.join('training_results', args.log_folder)

    mario = TestMario(state_space=wrapped_env_dims, action_space=env.action_space.n, log_loc=log_folder, weights=args.weight_file)

    print(f'Starting runs with trained Mario')
    run_num = 0

    while True:
        state = env.reset()[0]
        done = False
        steps = 0
        total_reward = 0.0

        # loop until end of game
        while not done:
            action = mario.choose_action(state)              # choose an action to perform
            new_state, reward, done, _, _ = env.step(action) # perform the chosen action
            state = new_state                                # update current state
            steps += 1
            total_reward += reward

        print(f'TEST RUN {run_num} STATISTICS: Steps = {steps}, Reward = {total_reward}')
        run_num += 1


if __name__ == '__main__':
    test()