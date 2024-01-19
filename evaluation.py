from preprocessing import make_env
from deep_q_net import DQN
import torch
import collections
import time
import numpy as np
from train import ENV_NAME
"""
Evaluation of the trained model
"""
FPS = 5
env = make_env(ENV_NAME)
net = DQN(env.observation_space.shape, env.action_space.n)
state = torch.load(ENV_NAME + ".dat", map_location=lambda stg, _: stg)
net.load_state_dict(state)
observation = env.reset()
total_reward = 0.0
c = collections.Counter()
frames = [env.render(mode='rgb_array')]
done = False
while not done:
    start_ts = time.time()
    frames.append(env.render(mode='rgb_array'))
    observation_v = torch.tensor(np.array([observation], copy=False))
    q_vals = net(observation_v).data.numpy()[0]
    action = np.argmax(q_vals)
    c[action] += 1

    observation, reward, done, _ = env.step(action)
    total_reward += reward
    if done:
        break
    delta = 1 / FPS - (time.time() - start_ts)
    if delta > 0:
        time.sleep(delta)
env.close()
