import numpy as np
import torch

from civtensor.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from civtensor.envs.freeciv_tensor_env.freeciv_tensor_env import FreecivTensorEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def make_train_env(seed, n_threads, env_args):
    """Make env for training."""

    def get_env_fn(rank):
        def init_env():
            env = FreecivTensorEnv(env_args)
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""

    def get_env_fn(rank):
        def init_env():
            env = FreecivTensorEnv(env_args)
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])
