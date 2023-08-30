import numpy as np
import torch

from civtensor.utils.trans_tools import _flatten


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class Buffer:
    def __init__(self, args, state_spaces, action_spaces):
        # init parameters
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.lstm_hidden_dim = args["lstm_hidden_dim"]
        self.n_lstm_layers = args["n_lstm_layers"]
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        # obtain input dimensions. TODO: be consistent with env
        self.rules_dim = self.state_spaces["rules"].shape[0]
        self.player_dim = self.state_spaces["player"].shape[0]
        self.other_players_dim = self.state_spaces["other_players"].shape[
            1
        ]  # or Sequence?
        self.units_dim = self.state_spaces["units"].shape[1]  # or Sequence?
        self.cities_dim = self.state_spaces["cities"].shape[1]  # or Sequence?
        self.other_units_dim = self.state_spaces["other_units"].shape[1]  # or Sequence?
        self.other_cities_dim = self.state_spaces["other_cities"].shape[
            1
        ]  # or Sequence?
        self.map_dim = self.state_spaces["map"].shape
        self.xsize, self.ysize, self.map_channels = self.map_dim
        self.n_max_other_players = self.state_spaces["other_players"].shape[0]
        self.n_max_units = self.state_spaces["units"].shape[0]
        self.n_max_cities = self.state_spaces["cities"].shape[0]
        self.n_max_other_units = self.state_spaces["other_units"].shape[0]
        self.n_max_other_cities = self.state_spaces["other_cities"].shape[0]

        # obtain output dimensions. TODO: be consistent with env
        self.actor_type_dim = self.action_spaces["actor_type"].n
        self.city_action_type_dim = self.action_spaces["city_action_type"].n
        self.unit_action_type_dim = self.action_spaces["unit_action_type"].n
        self.gov_action_type_dim = self.action_spaces["gov_action_type"].n

        # init buffers
        self.rules_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.rules_dim),
            dtype=np.float32,
        )
        self.player_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.player_dim),
            dtype=np.float32,
        )
        self.other_players_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_other_players,
                self.other_players_dim,
            ),
            dtype=np.float32,
        )
        self.units_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_units,
                self.units_dim,
            ),
            dtype=np.float32,
        )
        self.cities_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_cities,
                self.cities_dim,
            ),
            dtype=np.float32,
        )
        self.other_units_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_other_units,
                self.other_units_dim,
            ),
            dtype=np.float32,
        )
        self.other_cities_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_other_cities,
                self.other_cities_dim,
            ),
            dtype=np.float32,
        )
        self.map_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *self.map_dim),
            dtype=np.float32,
        )

        self.other_players_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_other_players,
                1,
            ),
            dtype=np.int32,
        )
        self.units_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_units, 1),
            dtype=np.int32,
        )
        self.cities_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_cities, 1),
            dtype=np.int32,
        )
        self.other_units_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_other_units,
                1,
            ),
            dtype=np.int32,
        )
        self.other_cities_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_other_cities,
                1,
            ),
            dtype=np.int32,
        )

        self.lstm_hidden_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_lstm_layers,
                self.lstm_hidden_dim,
            ),
            dtype=np.float32,
        )

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )

        self.actor_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.actor_type_dim),
            dtype=np.float32,
        )
        self.actor_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.actor_type_masks = np.ones(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )

        self.city_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int32
        )
        self.city_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.city_id_masks = np.ones(
            (self.episode_length, self.n_rollout_threads, self.n_max_cities, 1),
            dtype=np.int32,
        )

        self.city_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int32,
        )  # TODO: check data type
        self.city_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.city_action_type_masks = np.ones(
            (self.episode_length, self.n_rollout_threads, self.city_action_type_dim),
            dtype=np.int32,
        )

        self.unit_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int32
        )
        self.unit_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.unit_id_masks = np.ones(
            (self.episode_length, self.n_rollout_threads, self.n_max_units, 1),
            dtype=np.int32,
        )

        self.unit_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int32,
        )
        self.unit_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.unit_action_type_masks = np.ones(
            (self.episode_length, self.n_rollout_threads, self.unit_action_type_dim),
            dtype=np.int32,
        )

        self.gov_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int32,
        )
        self.gov_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.gov_action_type_masks = np.ones(
            (self.episode_length, self.n_rollout_threads, self.gov_action_type_dim),
            dtype=np.int32,
        )

        self.masks = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int32
        )
        self.bad_masks = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int32
        )

        self.step = 0

    def insert(self, data):
        """Insert data into buffer."""
        (
            rules,
            player,
            other_players,
            units,
            cities,
            other_units,
            other_cities,
            map,
            other_players_mask,
            units_mask,
            cities_mask,
            other_units_mask,
            other_cities_mask,
            lstm_hidden_state,
            actor_type,
            actor_type_log_prob,
            actor_type_mask,
            city_id,
            city_id_log_prob,
            city_id_mask,
            city_action_type,
            city_action_type_log_prob,
            city_action_type_mask,
            unit_id,
            unit_id_log_prob,
            unit_id_mask,
            unit_action_type,
            unit_action_type_log_prob,
            unit_action_type_mask,
            gov_action_type,
            gov_action_type_log_prob,
            gov_action_type_mask,
            mask,
            bad_mask,
            reward,
            value_pred,
        ) = data

        self.rules_input[self.step + 1] = rules.copy()
        self.player_input[self.step + 1] = player.copy()
        self.other_players_input[self.step + 1] = other_players.copy()
        self.units_input[self.step + 1] = units.copy()
        self.cities_input[self.step + 1] = cities.copy()
        self.other_units_input[self.step + 1] = other_units.copy()
        self.other_cities_input[self.step + 1] = other_cities.copy()
        self.map_input[self.step + 1] = map.copy()
        self.other_players_masks[self.step + 1] = other_players_mask.copy()
        self.units_masks[self.step + 1] = units_mask.copy()
        self.cities_masks[self.step + 1] = cities_mask.copy()
        self.other_units_masks[self.step + 1] = other_units_mask.copy()
        self.other_cities_masks[self.step + 1] = other_cities_mask.copy()
        self.lstm_hidden_states[self.step + 1] = lstm_hidden_state.copy()
        self.actor_type_output[self.step] = actor_type.copy()
        self.actor_type_log_probs[self.step] = actor_type_log_prob.copy()
        self.actor_type_masks[self.step + 1] = actor_type_mask.copy()
        self.city_id_output[self.step] = city_id.copy()
        self.city_id_log_probs[self.step] = city_id_log_prob.copy()
        self.city_id_masks[self.step + 1] = city_id_mask.copy()
        self.city_action_type_output[self.step] = city_action_type.copy()
        self.city_action_type_log_probs[self.step] = city_action_type_log_prob.copy()
        self.city_action_type_masks[self.step + 1] = city_action_type_mask.copy()
        self.unit_id_output[self.step] = unit_id.copy()
        self.unit_id_log_probs[self.step] = unit_id_log_prob.copy()
        self.unit_id_masks[self.step + 1] = unit_id_mask.copy()
        self.unit_action_type_output[self.step] = unit_action_type.copy()
        self.unit_action_type_log_probs[self.step] = unit_action_type_log_prob.copy()
        self.unit_action_type_masks[self.step + 1] = unit_action_type_mask.copy()
        self.gov_action_type_output[self.step] = gov_action_type.copy()
        self.gov_action_type_log_probs[self.step] = gov_action_type_log_prob.copy()
        self.gov_action_type_masks[self.step + 1] = gov_action_type_mask.copy()
        self.masks[self.step + 1] = mask.copy()
        self.bad_masks[self.step + 1] = bad_mask.copy()
        self.rewards[self.step] = reward.copy()
        self.value_preds[self.step] = value_pred.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.rules_input[0] = self.rules_input[-1].copy()
        self.player_input[0] = self.player_input[-1].copy()
        self.other_players_input[0] = self.other_players_input[-1].copy()
        self.units_input[0] = self.units_input[-1].copy()
        self.cities_input[0] = self.cities_input[-1].copy()
        self.other_units_input[0] = self.other_units_input[-1].copy()
        self.other_cities_input[0] = self.other_cities_input[-1].copy()
        self.map_input[0] = self.map_input[-1].copy()
        self.other_players_masks[0] = self.other_players_masks[-1].copy()
        self.units_masks[0] = self.units_masks[-1].copy()
        self.cities_masks[0] = self.cities_masks[-1].copy()
        self.other_units_masks[0] = self.other_units_masks[-1].copy()
        self.other_cities_masks[0] = self.other_cities_masks[-1].copy()
        self.lstm_hidden_states[0] = self.lstm_hidden_states[-1].copy()
        self.actor_type_masks[0] = self.actor_type_masks[-1].copy()
        self.city_id_masks[0] = self.city_id_masks[-1].copy()
        self.city_action_type_masks[0] = self.city_action_type_masks[-1].copy()
        self.unit_id_masks[0] = self.unit_id_masks[-1].copy()
        self.unit_action_type_masks[0] = self.unit_action_type_masks[-1].copy()
        self.gov_action_type_masks[0] = self.gov_action_type_masks[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(
        self, next_value, use_gae, gamma, gae_lambda, use_proper_time_limits=True
    ):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    ) * self.bad_masks[step + 1] + (
                        1 - self.bad_masks[step + 1]
                    ) * self.value_preds[
                        step
                    ]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                    )
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                    )

    def recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.n_rollout_threads
        assert n_rollout_threads >= num_mini_batch, (
            f"The number of processes ({n_rollout_threads}) "
            f"has to be greater than or equal to the number of "
            f"mini batches ({num_mini_batch})."
        )
        num_envs_per_batch = n_rollout_threads // num_mini_batch

        # shuffle indices
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        # prepare data for each mini batch
        for batch_id in range(num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            rules_batch = _flatten(self.rules_input[:-1, ids])
            player_batch = _flatten(self.player_input[:-1, ids])
            other_players_batch = _flatten(self.other_players_input[:-1, ids])
            units_batch = _flatten(self.units_input[:-1, ids])
            cities_batch = _flatten(self.cities_input[:-1, ids])
            other_units_batch = _flatten(self.other_units_input[:-1, ids])
            other_cities_batch = _flatten(self.other_cities_input[:-1, ids])
            map_batch = _flatten(self.map_input[:-1, ids])
            other_players_masks_batch = _flatten(self.other_players_masks[:-1, ids])
            units_masks_batch = _flatten(self.units_masks[:-1, ids])
            cities_masks_batch = _flatten(self.cities_masks[:-1, ids])
            other_units_masks_batch = _flatten(self.other_units_masks[:-1, ids])
            other_cities_masks_batch = _flatten(self.other_cities_masks[:-1, ids])
            lstm_hidden_states_batch = self.lstm_hidden_states[0:1, ids]
            value_preds_batch = _flatten(self.value_preds[:-1, ids])
            return_batch = _flatten(self.returns[:-1, ids])
            adv_targ = _flatten(advantages[:-1, ids])
            actor_type_batch = _flatten(self.actor_type_output[:, ids])
            actor_type_log_probs_batch = _flatten(self.actor_type_log_probs[:, ids])
            actor_type_masks_batch = _flatten(self.actor_type_masks[:-1, ids])
            city_id_batch = _flatten(self.city_id_output[:, ids])
            city_id_log_probs_batch = _flatten(self.city_id_log_probs[:, ids])
            city_id_masks_batch = _flatten(self.city_id_masks[:-1, ids])
            city_action_type_batch = _flatten(self.city_action_type_output[:, ids])
            city_action_type_log_probs_batch = _flatten(
                self.city_action_type_log_probs[:, ids]
            )
            city_action_type_masks_batch = _flatten(
                self.city_action_type_masks[:-1, ids]
            )
            unit_id_batch = _flatten(self.unit_id_output[:, ids])
            unit_id_log_probs_batch = _flatten(self.unit_id_log_probs[:, ids])
            unit_id_masks_batch = _flatten(self.unit_id_masks[:-1, ids])
            unit_action_type_batch = _flatten(self.unit_action_type_output[:, ids])
            unit_action_type_log_probs_batch = _flatten(
                self.unit_action_type_log_probs[:, ids]
            )
            unit_action_type_masks_batch = _flatten(
                self.unit_action_type_masks[:-1, ids]
            )
            gov_action_type_batch = _flatten(self.gov_action_type_output[:, ids])
            gov_action_type_log_probs_batch = _flatten(
                self.gov_action_type_log_probs[:, ids]
            )
            gov_action_type_masks_batch = _flatten(self.gov_action_type_masks[:-1, ids])
            masks_batch = _flatten(self.masks[:-1, ids])
            bad_masks_batch = _flatten(self.bad_masks[:-1, ids])

            lstm_hidden_states_batch = lstm_hidden_states_batch.squeeze(0)

            yield (
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
                lstm_hidden_states_batch,
                value_preds_batch,
                return_batch,
                adv_targ,
                actor_type_batch,
                actor_type_log_probs_batch,
                actor_type_masks_batch,
                city_id_batch,
                city_id_log_probs_batch,
                city_id_masks_batch,
                city_action_type_batch,
                city_action_type_log_probs_batch,
                city_action_type_masks_batch,
                unit_id_batch,
                unit_id_log_probs_batch,
                unit_id_masks_batch,
                unit_action_type_batch,
                unit_action_type_log_probs_batch,
                unit_action_type_masks_batch,
                gov_action_type_batch,
                gov_action_type_log_probs_batch,
                gov_action_type_masks_batch,
                masks_batch,
                bad_masks_batch,
            )
