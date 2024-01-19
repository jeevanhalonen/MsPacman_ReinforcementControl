import torch
import torch.nn as nn
import numpy as np
import collections

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Experience = collections.namedtuple('Experience',
                                    field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Agent:
    """
    The Agent class encapsulates the logic of the agent
    """
    def __init__(self, env, exp_buffer):
        self.state = None
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        state = self.env.reset()
        self.state = state
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        obs = self.env.step(action)
        new_state = obs[0]
        reward = obs[1]
        done = obs[2]

        self.total_reward += reward

        exp = Experience(self.state, action, reward, done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if done.any():
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, gamma):
    states, actions, rewards, dones, next_states = batch
    states_v = torch.tensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards_v
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)

    return loss
