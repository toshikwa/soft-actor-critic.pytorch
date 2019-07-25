import numpy as np


class Observer():

    def __init__(self, env):
        self._env = env
        if np.inf in np.r_[
                self.observation_space.high, self.observation_space.low]:
            self.state_scale =\
                2. / (self.observation_space.high-self.observation_space.low)
            self.state_bias =\
                - (self.observation_space.high+self.observation_space.low)\
                / (self.observation_space.high-self.observation_space.low)
        else:
            self.state_scale = 1.0
            self.state_bias = 0.0

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def _max_episode_steps(self):
        return self._env._max_episode_steps

    def reset(self):
        return self.preprocess(self._env.reset())

    def render(self):
        self._env.render()

    def step(self, action):
        n_state, reward, done, info = self._env.step(action)
        return self.preprocess(n_state), reward, done, info

    def preprocess(self, state):
        return state * self.state_scale + self.state_bias
