import time
from pyboy import PyBoy
from pokemon_rl.environment.env import GenericPyBoyEnv
from pokemon_rl.constants import ACTIONS
import random
import os
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean, resize
actions = ['','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select']


cur_dir = os.path.dirname(os.path.realpath(__file__))
print("cur_dir", cur_dir)

pyboy = PyBoy(os.path.join(cur_dir, '..\ROMs/Pokemon Blue.gb'))

with open(os.path.join(cur_dir, "..\saves\pokemon_blue_save_after_delivery.state"), "rb") as f:
    pyboy.load_state(f)

start_time = time.time()
# while pyboy.tick():
#     # if int(time.time()) % 120 == 0:
#     #     with open("pokemon_blue_save_after_delivery.state", "wb") as f:
#     #         pyboy.save_state(f)
#     #         print("SAVING STATE")
#     pass
env = GenericPyBoyEnv(pyboy)
for i in range(100000000):
    
    observation, reward, done, truncated, info = env.step(0)
    # obs_downsampled_resize = resize(observation, (72, 80))
    # obs_downsampled_mean = downscale_local_mean(observation, (2, 2))

    # print(obs_downsampled_resize.shape)
    # if i % 100 == 0:
    #     fig, ax = plt.subplots(1, 2)

    #     ax[0].imshow(obs_downsampled_resize, interpolation='nearest')
    #     ax[1].imshow(obs_downsampled_mean, interpolation='nearest')
        
    #     ax[0].set_xlabel("resize")
    #     ax[1].set_xlabel("downsample local mean")

    #     plt.show()
pyboy.stop()