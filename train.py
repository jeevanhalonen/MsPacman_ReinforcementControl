import torch
import torch.optim as optim
from preprocessing import make_env
from deep_q_net import DQN
from experience_buffer import ExperienceBuffer
from agent import Agent, calc_loss
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
To train the model
"""
# hyperparameters
GAMMA = 0.99
BATCH_SIZE = 128
REPLAY_SIZE = 10_000
REPLAY_START_SIZE = 10_000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1_000
EPSILON_DECAY_LAST_FRAME = 1000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
MAX_ITERATIONS = 2_000

ENV_NAME = 'MsPacman-v0'
env = make_env(ENV_NAME)

net = DQN(env.observation_space.shape, env.action_space.n).to(device)
tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
epsilon = EPSILON_START

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
total_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_m_reward = None
history = []

while frame_idx < MAX_ITERATIONS:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
    reward = agent.play_step(net, epsilon)
    if reward is not None:
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        m_reward = np.mean(total_rewards[-100:])
        print(f"{frame_idx:,}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
              f"eps {epsilon:.2f}, speed {speed:.2f} f/s")
        history.append((frame_idx, m_reward, epsilon, speed))
        if best_m_reward is None or best_m_reward < m_reward:
            torch.save(net.state_dict(), ENV_NAME + ".dat")
            if best_m_reward is not None:
                print(f"{frame_idx}: best reward updated "
                      f"{best_m_reward:.3f} -> {m_reward:.3f} (eps:{epsilon:.2},"
                      f" speed: {speed:.2f} f/s)")
            best_m_reward = m_reward

    if len(buffer) < REPLAY_START_SIZE:
        continue

    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.load_state_dict(net.state_dict())

    optimizer.zero_grad()
    batch = buffer.sample(BATCH_SIZE)
    loss_t = calc_loss(batch, net, tgt_net, GAMMA)
    loss_t.backward()
    optimizer.step()
