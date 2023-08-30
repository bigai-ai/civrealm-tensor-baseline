import numpy as np
import torch

from civtensor.utils.models_tools import update_linear_schedule
from civtensor.models.agent import Agent
from civtensor.utils.envs_tools import check


class PPO:
    def __init__(self, args, state_spaces, action_spaces, device=torch.device("cpu")):
        self.args = args
        self.device = device

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.num_mini_batch = args["num_mini_batch"]
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

    def update(self, sample):
        (
            rules_batch,
            player_batch,
            other_players_batch,
            units_batch,
            cities_batch,
            other_units_batch,
            other_cities_batch,
            map_batch,
            other_players_masks_batch,
            units_masks_batch,
            cities_masks_batch,
            other_units_masks_batch,
            other_cities_masks_batch,
            rnn_hidden_states_batch,
            value_preds_batch,
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
        other_players_batch = check(other_players_batch).to(self.device)
        units_batch = check(units_batch).to(self.device)
        cities_batch = check(cities_batch).to(self.device)
        other_units_batch = check(other_units_batch).to(self.device)
        other_cities_batch = check(other_cities_batch).to(self.device)
        map_batch = check(map_batch).to(self.device)
        other_players_masks_batch = check(other_players_masks_batch).to(self.device)
        units_masks_batch = check(units_masks_batch).to(self.device)
        cities_masks_batch = check(cities_masks_batch).to(self.device)
        other_units_masks_batch = check(other_units_masks_batch).to(self.device)
        other_cities_masks_batch = check(other_cities_masks_batch).to(self.device)
        rnn_hidden_states_batch = check(rnn_hidden_states_batch).to(self.device)
        value_preds_batch = check(value_preds_batch).to(self.device)
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
            city_id_dist_entropy,
            city_action_type_log_probs_batch,
            city_action_type_dist_entropy,
            unit_id_log_probs_batch,
            unit_id_dist_entropy,
            unit_action_type_log_probs_batch,
            unit_action_type_dist_entropy,
            gov_action_type_log_probs_batch,
            gov_action_type_dist_entropy,
        ) = self.agent.evaluate_actions(
            rules_batch,
            player_batch,
            other_players_batch,
            units_batch,
            cities_batch,
            other_units_batch,
            other_cities_batch,
            map_batch,
            other_players_masks_batch,
            units_masks_batch,
            cities_masks_batch,
            other_units_masks_batch,
            other_cities_masks_batch,
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

    def train(self, buffer, advantages):
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
                    sample
                )
                train_info["policy_loss"] += policy_loss
                train_info["value_loss"] += value_loss
                train_info["dist_entropy"] += dist_entropy
                train_info["grad_norm"] += grad_norm
                train_info["ratio"] += ratio

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
