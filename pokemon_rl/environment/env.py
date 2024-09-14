# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pokemon_rl.constants import ACTIONS, FRAME_SHAPE

class GenericPyBoyEnv(gym.Env):

    def __init__(self, pyboy, debug=False, do_render=True):
        super().__init__()
        self.pyboy = pyboy
        self.do_render = do_render
        self._fitness = 0
        self._previous_fitness = 0
        self.visited_map_locs = set()
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=1.0, shape=FRAME_SHAPE, dtype=np.float32)

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # Move the agent
        if action == 0:
            pass
        else:
            self.pyboy.button(ACTIONS[action])

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        self.pyboy.tick(1, self.do_render)

        # done = self.pyboy.game_wrapper.game_over
        done = False
        print(self.get_game_coords())
        self._calculate_fitness()
        reward = self._fitness-self._previous_fitness

        # Get only the first frame as it is black and white
        observation = self.pyboy.screen.ndarray[:, :, 0]
        info = {}
        truncated = False
        
        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        cur_map_coords = self.get_game_coords()

        if cur_map_coords not in self.visited_map_locs:
            self.visited_map_locs.append(cur_map_coords)
            self._fitness = 1
        else:
            self._fitness = 0


    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        observation = self.pyboy.screen.ndarray[:, :, 0]
        info = {}
        return observation, info

    def get_game_coords(self):
        return (
            self.pyboy.memory[0xD35E], # Current Map Number
            self.pyboy.memory[0xD362], # Current Player X-Position
            self.pyboy.memory[0xD361]  # Current Player Y-Position
        )

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
