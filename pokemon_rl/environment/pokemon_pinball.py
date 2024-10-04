import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pokemon_rl.constants import ACTIONS
import os
import numpy as np
import random
from enum import Enum
from skimage.transform import downscale_local_mean
import cv2

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
    LEFT_TILT = 5
    RIGHT_TILT = 6
    UP_TILT = 7
    LEFT_UP_TILT = 8
    RIGHT_UP_TILT = 9

CUSTOM_DATA_SIZE = 14
GAME_AREA_SIZE = 16 * 20
NUM_STACKS = 2
STATE_SIZE = CUSTOM_DATA_SIZE + GAME_AREA_SIZE

class PinballPyBoyEnv(gym.Env):

    def __init__(self, rom_file: str, debug=False, do_render=False, frame_skip: int = 2):
        super().__init__()
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.pyboy = PyBoy(os.path.join(cur_dir, rom_file), window="null" if not do_render else "SDL2")
        self.do_render = do_render
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug
        self.frame_skip = frame_skip
        self.flippers_active = [0.0, 0.0]
        self.state = np.zeros((NUM_STACKS, STATE_SIZE), dtype=np.float32)

        if not self.debug:
            self.pyboy.set_emulation_speed(0)
  
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(NUM_STACKS * STATE_SIZE,), dtype=np.float32)
        self.pyboy.game_wrapper.start_game()


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        self.perform_action(action)

        # Consider disabling renderer when not needed to improve speed:
        reward = 0
        for i in range(self.frame_skip):
            self.pyboy.tick(1, self.do_render)
            reward += self._calculate_fitness()

        done = self.done

        # Get only the first frame as it is black and white
        observation = self._get_obs()
        num_skips = 0
        while (observation == self._prev_obs).sum() == len(observation) and not done:
            self.perform_action(4)
            self.pyboy.tick(1, self.do_render)
            self.perform_action(2)
            self.pyboy.tick(1, self.do_render)
            self.perform_action(4)
            self.perform_action(7)
            self.pyboy.tick(1, self.do_render)

            observation = self._get_obs()
            num_skips += 1
            if num_skips % 1000 == 0:
                print("NUM SKIPS:", num_skips)
            done = self.done
            reward += self._calculate_fitness()

        self.state = np.roll(self.state, 1, axis=0)
        self.state[0] = observation

        self._prev_obs = observation
        info = {}
        truncated = False


        return self.state.flatten(), reward, done, truncated, info

    @property
    def done(self):
        return self.pyboy.game_wrapper.game_over or self.pyboy.game_wrapper.balls_left <= 1

    def perform_action(self, action):
        # Move the agent
        if action == Actions.IDLE.value:
            pass
        elif action == Actions.LEFT_FLIPPER_PRESS.value:
            self.pyboy.button_press("left")
            self.flippers_active[0] = 255.0
        elif action == Actions.RIGHT_FLIPPER_PRESS.value:
            self.pyboy.button_press("a")
            self.flippers_active[1] = 255.0
        elif action == Actions.LEFT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("left")
            self.flippers_active[0] = 0.0
        elif action == Actions.RIGHT_FLIPPER_RELEASE.value:
            self.pyboy.button_release("a")
            self.flippers_active[1] = 0.0
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
        state = cv2.cvtColor(self.pyboy.screen.ndarray, cv2.COLOR_RGB2GRAY)
        state = downscale_local_mean(state, (2, 2))
        print(state.shape)
        # print(self.pyboy.game_area())

        game_area = np.array(self.pyboy.game_area().flatten() / 255.0, dtype=np.float32)
        custom_data = np.array(
            [
                self.pyboy.game_wrapper.ball_x,
                self.pyboy.game_wrapper.ball_y,
                self.pyboy.game_wrapper.ball_x_velocity,
                self.pyboy.game_wrapper.ball_y_velocity,
                self.pyboy.game_wrapper.current_map,
                self.pyboy.game_wrapper.current_stage,
                self.pyboy.game_wrapper.ball_saver_seconds_left,
                self.pyboy.game_wrapper.pokemon_caught_in_session,
                self.pyboy.game_wrapper.pokemon_seen_in_session,
                self.pyboy.game_wrapper.roulette_slots_opened,
                self.pyboy.game_wrapper.roulette_slots_entered,
                self.pyboy.game_wrapper.multiplier,
                *self.flippers_active
            ],
            dtype=np.float32
        ) / 255
        # print([
        #         self.pyboy.game_wrapper.ball_x,
        #         self.pyboy.game_wrapper.ball_y,
        #         self.pyboy.game_wrapper.ball_x_velocity,
        #         self.pyboy.game_wrapper.ball_y_velocity,
        #         self.pyboy.game_wrapper.ball_saver_seconds_left
        # ])
        # print(self.pyboy.game_wrapper)
        # print(self.pyboy.memory[0xd4b3], self.pyboy.memory[0xd4b7])
        # print(self.pyboy.memory[0xd4b5], self.pyboy.memory[0xd4b9])
        # observation = self.pyboy.screen.ndarray[:, :, 0]
        # print("self.pyboy.screen.ndarray", self.pyboy.screen.ndarray.shape)
        return np.concatenate([game_area, custom_data])

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness
        self._fitness = self.pyboy.game_wrapper.score
        score_gained = self._fitness - self._previous_fitness 
        return score_gained / 1000

    def reset(self, **kwargs):
        self.flippers_active = [0.0, 0.0]
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        # Pick a random level
        self.pyboy.tick(random.randint(1, 100), self.do_render)
        self.perform_action(6)
        self.pyboy.tick(1, self.do_render)

        self._prev_obs = observation = self._get_obs()
        info = {}

        self.state[:] = observation

        return self.state.flatten(), info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
