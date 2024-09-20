import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces

RESHAPE_SIZE = (200, 300)
class LunarImageWrapper(gym.Wrapper):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env):
        super().__init__(env)
        self.num_steps = 4
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.num_steps, *RESHAPE_SIZE), dtype=np.uint8)
        # Keep a stack of the previous 4 observations
        self.prev_obs = np.zeros((self.num_steps, RESHAPE_SIZE[0], RESHAPE_SIZE[1]), dtype=np.float32)

    def _get_image_observation(self):
        # Render the CartPole environment
        cartpole_image = self.render()
        # print("cartpole_image.shape", cartpole_image.shape)

        # Resize the image to 84x84 pixels
        resized_image = cv2.resize(cartpole_image, (RESHAPE_SIZE[1], RESHAPE_SIZE[0]))
        # make it grayscale
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        resized_image = (np.expand_dims(resized_image, axis=0) / 255.0).astype(np.float32)

        return resized_image

    def reset(self, **kwargs):
        self.env.reset(**kwargs)  # Ignore the 'seed' argument
        obs = self._get_image_observation()
        self.prev_obs[:] = obs
        return self.prev_obs, {}

    def step(self, action):
        _, reward, done, terminated, info = self.env.step(action)
        
        # Update the current observation
        self.prev_obs = np.roll(self.prev_obs, 1, axis=0)
        self.prev_obs[0] = self._get_image_observation() 
        
        return self.prev_obs, reward, done, terminated, info