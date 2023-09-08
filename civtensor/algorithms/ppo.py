import numpy as np
import torch

from civtensor.utils.models_tools import (
    update_linear_schedule,
    get_grad_norm,
    huber_loss,
    mse_loss,
)
from civtensor.models.agent import Agent
from civtensor.utils.envs_tools import check


class PPO:
    def __init__(
        self, args, observation_spaces, action_spaces, device=torch.device("cpu")
    ):
        self.args = args
        self.device = device

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.num_mini_batch = args["num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.value_loss_coef = args["value_loss_coef"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

        self.agent = Agent(args, observation_spaces, action_spaces, device=device)
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

    def cal_value_loss(
        self,
        value_preds_batch,
        old_value_preds_batch,
        return_batch,
        value_normalizer=None,
    ):
        """Calculate value function loss.
        Args:
            value_preds_batch: (torch.Tensor) value function predictions.
            old_value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (ValueNorm) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = old_value_preds_batch + (
            value_preds_batch - old_value_preds_batch
        ).clamp(-self.clip_param, self.clip_param)
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = (
                value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = (
                value_normalizer.normalize(return_batch) - value_preds_batch
            )
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - value_preds_batch

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def update(self, sample, value_normalizer=None):
        (
            rules_batch,
            player_batch,
            others_player_batch,
            unit_batch,
            city_batch,
            others_unit_batch,
            others_city_batch,
            map_batch,
            others_player_masks_batch,
            unit_masks_batch,
            city_masks_batch,
            others_unit_masks_batch,
            others_city_masks_batch,
            rnn_hidden_states_batch,
            old_value_preds_batch,
            return_batch,
            adv_targ,
            actor_type_batch,
            old_actor_type_log_probs_batch,
            actor_type_masks_batch,
            city_id_batch,
            old_city_id_log_probs_batch,
            city_id_masks_batch,
            city_action_type_batch,
            old_city_action_type_log_probs_batch,
            city_action_type_masks_batch,
            unit_id_batch,
            old_unit_id_log_probs_batch,
            unit_id_masks_batch,
            unit_action_type_batch,
            old_unit_action_type_log_probs_batch,
            unit_action_type_masks_batch,
            gov_action_type_batch,
            old_gov_action_type_log_probs_batch,
            gov_action_type_masks_batch,
            masks_batch,
            bad_masks_batch,
        ) = sample

        rules_batch = check(rules_batch).to(self.device)
        player_batch = check(player_batch).to(self.device)
        others_player_batch = check(others_player_batch).to(self.device)
        unit_batch = check(unit_batch).to(self.device)
        city_batch = check(city_batch).to(self.device)
        others_unit_batch = check(others_unit_batch).to(self.device)
        others_city_batch = check(others_city_batch).to(self.device)
        map_batch = check(map_batch).to(self.device)
        others_player_masks_batch = check(others_player_masks_batch).to(self.device)
        unit_masks_batch = check(unit_masks_batch).to(self.device)
        city_masks_batch = check(city_masks_batch).to(self.device)
        others_unit_masks_batch = check(others_unit_masks_batch).to(self.device)
        others_city_masks_batch = check(others_city_masks_batch).to(self.device)
        rnn_hidden_states_batch = check(rnn_hidden_states_batch).to(self.device)
        old_value_preds_batch = check(old_value_preds_batch).to(self.device)
        return_batch = check(return_batch).to(self.device)
        adv_targ = check(adv_targ).to(self.device)
        actor_type_batch = check(actor_type_batch).to(self.device)
        old_actor_type_log_probs_batch = check(old_actor_type_log_probs_batch).to(
            self.device
        )
        actor_type_masks_batch = check(actor_type_masks_batch).to(self.device)
        city_id_batch = check(city_id_batch).to(self.device)
        old_city_id_log_probs_batch = check(old_city_id_log_probs_batch).to(self.device)
        city_id_masks_batch = check(city_id_masks_batch).to(self.device)
        city_action_type_batch = check(city_action_type_batch).to(self.device)
        old_city_action_type_log_probs_batch = check(
            old_city_action_type_log_probs_batch
        ).to(self.device)
        city_action_type_masks_batch = check(city_action_type_masks_batch).to(
            self.device
        )
        unit_id_batch = check(unit_id_batch).to(self.device)
        old_unit_id_log_probs_batch = check(old_unit_id_log_probs_batch).to(self.device)
        unit_id_masks_batch = check(unit_id_masks_batch).to(self.device)
        unit_action_type_batch = check(unit_action_type_batch).to(self.device)
        old_unit_action_type_log_probs_batch = check(
            old_unit_action_type_log_probs_batch
        ).to(self.device)
        unit_action_type_masks_batch = check(unit_action_type_masks_batch).to(
            self.device
        )
        gov_action_type_batch = check(gov_action_type_batch).to(self.device)
        old_gov_action_type_log_probs_batch = check(
            old_gov_action_type_log_probs_batch
        ).to(self.device)
        gov_action_type_masks_batch = check(gov_action_type_masks_batch).to(self.device)
        masks_batch = check(masks_batch).to(self.device)
        bad_masks_batch = check(bad_masks_batch).to(self.device)

        (
            actor_type_log_probs_batch,
            actor_type_dist_entropy,
            city_id_log_probs_batch,
            city_action_type_log_probs_batch,
            city_action_type_dist_entropy,
            unit_id_log_probs_batch,
            unit_action_type_log_probs_batch,
            unit_action_type_dist_entropy,
            gov_action_type_log_probs_batch,
            gov_action_type_dist_entropy,
            value_preds_batch,
        ) = self.agent.evaluate_actions(
            rules_batch,
            player_batch,
            others_player_batch,
            unit_batch,
            city_batch,
            others_unit_batch,
            others_city_batch,
            map_batch,
            others_player_masks_batch,
            unit_masks_batch,
            city_masks_batch,
            others_unit_masks_batch,
            others_city_masks_batch,
            rnn_hidden_states_batch,
            actor_type_batch,
            actor_type_masks_batch,
            city_id_batch,
            city_id_masks_batch,
            city_action_type_batch,
            city_action_type_masks_batch,
            unit_id_batch,
            unit_id_masks_batch,
            unit_action_type_batch,
            unit_action_type_masks_batch,
            gov_action_type_batch,
            gov_action_type_masks_batch,
            masks_batch,
        )

        action_log_probs_batch = (
            (actor_type_batch == 0)
            * (
                actor_type_log_probs_batch
                + city_id_log_probs_batch
                + city_action_type_log_probs_batch
            )
            + (actor_type_batch == 1)
            * (
                actor_type_log_probs_batch
                + unit_id_log_probs_batch
                + unit_action_type_log_probs_batch
            )
            + (actor_type_batch == 2)
            * (actor_type_log_probs_batch + gov_action_type_log_probs_batch)
            + (actor_type_batch == 3) * (actor_type_log_probs_batch)
        )

        old_action_log_probs_batch = (
            (actor_type_batch == 0)
            * (
                old_actor_type_log_probs_batch
                + old_city_id_log_probs_batch
                + old_city_action_type_log_probs_batch
            )
            + (actor_type_batch == 1)
            * (
                old_actor_type_log_probs_batch
                + old_unit_id_log_probs_batch
                + old_unit_action_type_log_probs_batch
            )
            + (actor_type_batch == 2)
            * (old_actor_type_log_probs_batch + old_gov_action_type_log_probs_batch)
            + (actor_type_batch == 3) * (old_actor_type_log_probs_batch)
        )

        dist_entropy = (
            (actor_type_batch == 0)
            * (actor_type_dist_entropy + city_action_type_dist_entropy)
            + (actor_type_batch == 1)
            * (actor_type_dist_entropy + unit_action_type_dist_entropy)
            + (actor_type_batch == 2)
            * (actor_type_dist_entropy + gov_action_type_dist_entropy)
            + (actor_type_batch == 3) * (actor_type_dist_entropy)
        ).mean()

        ratio = torch.exp(action_log_probs_batch - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        )
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = self.cal_value_loss(
            value_preds_batch,
            old_value_preds_batch,
            return_batch,
            value_normalizer=value_normalizer,
        )

        self.optimizer.zero_grad()
        (
            value_loss * self.value_loss_coef
            + policy_loss
            - dist_entropy * self.entropy_coef
        ).backward()
        if self.use_max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = get_grad_norm(self.agent.parameters())
        self.optimizer.step()

        return policy_loss, value_loss, dist_entropy, grad_norm, ratio

    def train(self, buffer, advantages, value_normalizer=None):
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["value_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["grad_norm"] = 0
        train_info["ratio"] = 0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                policy_loss, value_loss, dist_entropy, grad_norm, ratio = self.update(
                    sample, value_normalizer=value_normalizer
                )
                train_info["policy_loss"] += policy_loss.item()
                train_info["value_loss"] += value_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["grad_norm"] += grad_norm
                train_info["ratio"] += ratio.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.agent.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.agent.eval()
