# File for storing custom wrappers
import gym


# Only return every x frames since individual frames aren't useful for determining important factors like velocities and directions.
# Inspired by the AtariPreprocessing gym wrapper frame skip.
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward, done, trunc = 0.0, False, False
        for _ in range(self.skip):
            new_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return new_state, total_reward, done, trunc, info