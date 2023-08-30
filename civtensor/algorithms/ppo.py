import numpy as np
import torch

from civtensor.utils.models_tools import update_linear_schedule
from civtensor.models.agent import Agent


class PPO:
    def __init__(self, args, state_spaces, action_spaces, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces

        self.agent = Agent(args, state_spaces, action_spaces, device=device)
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions_values(self, data):
        pass

    def train(self):
        pass

    def prep_training(self):
        """Prepare for training."""
        self.agent.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.agent.eval()
