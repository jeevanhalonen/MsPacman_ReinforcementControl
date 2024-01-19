import matplotlib.pyplot as plt
from preprocessing import make_env
ENV_NAME = 'MsPacman-v0'

environment = make_env(ENV_NAME)
environment.reset()
observation = []
for _ in range(100):
    obs, info = environment.step(environment.action_space.sample())
    observation.append(obs)

fig, axes = plt.subplots(figsize=(12, 4), ncols=4)
for i in range(4):
    axes[i].imshow(observation[i])
    axes[i].set_axis('off')
    axes[i].set_title(f'Frame #{i}')

environment.close()
