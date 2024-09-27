import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pokemon_rl.constants import ACTIONS
import os
import numpy as np
import random
from enum import Enum

# ACTIONS = [
#     "NONE",
#     "RIGHT_PRESS",
#     "LEFT_PRESS",
#     "RIGHT RELEASE",
#     "LEFT_RELEASE",
#     "LEFT_TILT",
#     "RIGHT_TILT",
#     "UP_TILT",
#     "LEFT_UP_TILT",
#     "RIGHT_UP_TILT"
# ]

class Actions(Enum):
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    LEFT_FLIPPER_TAP = 5
    RIGHT_FLIPPER_TAP = 6
    LEFT_TILT = 7
    RIGHT_TILT = 8
    UP_TILT = 9
    LEFT_UP_TILT = 10
    RIGHT_UP_TILT = 11


class PinballPyBoyEnv(gym.Env):

    def __init__(self, rom_file: str, debug=False, do_render=False, frame_skip: int = 4):
        super().__init__()
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.pyboy = PyBoy(os.path.join(cur_dir, rom_file))
        self.do_render = do_render
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug
        self.frame_skip = frame_skip

        if not self.debug:
            self.pyboy.set_emulation_speed(0)
  
        
        self.action_space = spaces.Discrete(12)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(16*20, ), dtype=np.float32)
        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        self.perform_action(action)

        # Consider disabling renderer when not needed to improve speed:
        reward = 0
        for i in range(4):
            self.pyboy.tick(1, self.do_render)
            self._calculate_fitness()
            reward += (self._fitness - self._previous_fitness) / 1000


        done = self.pyboy.game_wrapper.game_over
 
        # Get only the first frame as it is black and white
        observation = self._get_obs()
        num_skips = 0
        while (observation == self._prev_obs).sum() == len(observation):
            self.perform_action(4)
            self.pyboy.tick(1, self.do_render)
            self.perform_action(2)
            self.pyboy.tick(1, self.do_render)
            self.perform_action(4)
            self.perform_action(9)
            self.pyboy.tick(1, self.do_render)

            observation = self._get_obs()
            num_skips += 1
            if num_skips % 1000 == 0:
                print("NUM SKIPS:", num_skips)
        self._prev_obs = observation
        info = {}
        truncated = False
        
        return observation, reward, done, truncated, info

    def perform_action(self, action):
        # Move the agent
        if action == Actions.IDLE.value:
            pass
        elif action == Actions.LEFT_FLIPPER_PRESS.value:
            self.pyboy.button_press("left")
        elif action == Actions.RIGHT_FLIPPER_PRESS.value:
            self.pyboy.button_press("a")
        elif action == Actions.LEFT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("left")
        elif action == Actions.RIGHT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("a")
        elif action == Actions.LEFT_FLIPPER_TAP.value:
            self.pyboy.button("left")
        elif action == Actions.RIGHT_FLIPPER_TAP.value:
            self.pyboy.button("a")
        elif action == Actions.LEFT_TILT.value:
            self.pyboy.button("down")
        elif action == Actions.RIGHT_TILT.value:
            self.pyboy.button("b")
        elif action == Actions.UP_TILT.value:
            self.pyboy.button("select")
        elif action == Actions.LEFT_UP_TILT.value:
            self.pyboy.button("select")
            self.pyboy.button("down")
        elif action == Actions.RIGHT_UP_TILT.value:
            self.pyboy.button("select")
            self.pyboy.button("b")

    def _get_obs(self):
        return np.array(self.pyboy.game_area().flatten() / 255.0, dtype=np.float32)

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self.pyboy.game_wrapper.score

    def reset(self, **kwargs):

        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        # Pick a random level
        self.pyboy.tick(random.randint(1, 100), self.do_render)
        self.perform_action(8)
        self.pyboy.tick(1, self.do_render)

        self._prev_obs = observation = self._get_obs()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
