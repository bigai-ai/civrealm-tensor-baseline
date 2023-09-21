import os
import random
import time
import gymnasium

import numpy as np
import torch

from civtensor.envs.freeciv_tensor_env.freeciv_tensor_env import TensorBaselineEnv
from freeciv_gym.freeciv.utils.port_list import DEV_PORT_LIST
from freeciv_gym.freeciv.utils.port_list import EVAL_PORT_LIST

# from civtensor.envs.env_wrappers import ShareSubprocVecEnv, DummyVecEnv
# from civtensor.envs.freeciv_tensor_env.freeciv_tensor_env import FreecivTensorEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def make_train_env(env, seed, n_threads):
    return TensorBaselineEnv(n_threads, DEV_PORT_LIST[0])


# def make_train_env(env, seed, n_threads, env_args) -> gymnasium.Env:
#     """Make env for training."""

#     print(f"making environments with {n_threads} and env_args: {env_args}")
#     env_args = env_args if env_args else {}
#     # TODO: distribute ports to ranks
#     from freeciv_gym.freeciv.utils.port_list import DEV_PORT_LIST

#     # TODO: Currently env_args are not useful
#     def get_env_fn(rank):
#         # TODO: put this somewhere better
#         def init_env():
#             env = FreecivTensorEnv(client_port=random.choice(DEV_PORT_LIST))
#             env.seed(seed + rank * 1000)
#             return env

#         return init_env

#     if n_threads == 1:
#         print(f"got {n_threads} thread")
#         return DummyVecEnv(get_env_fn(0))
#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


# def make_eval_env(env_name, seed, n_threads, env_args):
#     """Make env for evaluation."""

#     def get_env_fn(rank):
#         def init_env():
#             env = FreecivTensorEnv(env_args)
#             env.seed(seed * 50000 + rank * 10000)
#             return env

#         return init_env

#     if n_threads == 1:
#         return ShareDummyVecEnv([get_env_fn(0)])
#     else:
#         return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])
