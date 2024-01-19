import gym
import gym.spaces
import cv2
import numpy as np
import collections


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame
    """
    def __init__(self, env, skip):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=skip)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    The ProcessFrame84 wrapper resizes the frame to a smaller image.
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        # convert to a single color
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        # reduce the image resolution
        resized_screen = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        x_t = np.reshape(resized_screen, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    """
     To extend the observations from images to a mini-video.
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.buffer = None
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation[0]
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    """
    To convert the image to the PyTorch format and scale the values.
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        if old_shape is None:
            raise ValueError("The observation_space of the environment is None.")
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    envi = gym.make(env_name)
    envi = MaxAndSkipEnv(envi, skip=4)
    envi = ProcessFrame84(envi)
    envi = ImageToPyTorch(envi)
    envi = BufferWrapper(envi, n_steps=4)
    envi = ScaledFloatFrame(envi)
    return envi
