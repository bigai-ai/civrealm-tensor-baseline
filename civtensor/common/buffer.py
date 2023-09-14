import numpy as np
import torch

from civtensor.utils.trans_tools import _flatten


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class Buffer:
    def __init__(self, args, observation_spaces, action_spaces):
        # init parameters
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.rnn_hidden_dim = args["rnn_hidden_dim"]
        self.n_rnn_layers = args["n_rnn_layers"]
        self.gamma = args["gamma"]
        self.gae_lambda = args["gae_lambda"]
        self.use_gae = args["use_gae"]
        self.use_proper_time_limits = args["use_proper_time_limits"]

        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        # obtain input dimensions. TODO: be consistent with env
        self.rules_dim = self.observation_spaces["rules"].shape[0]
        self.player_dim = self.observation_spaces["player"].shape[0]
        self.others_player_dim = self.observation_spaces["others_player"].shape[
            1
        ]  # or Sequence?
        self.unit_dim = self.observation_spaces["unit"].shape[1]  # or Sequence?
        self.city_dim = self.observation_spaces["city"].shape[1]  # or Sequence?
        self.others_unit_dim = self.observation_spaces["others_unit"].shape[
            1
        ]  # or Sequence?
        self.others_city_dim = self.observation_spaces["others_city"].shape[
            1
        ]  # or Sequence?
        self.map_dim = self.observation_spaces["map"].shape
        self.xsize, self.ysize, self.map_channels = self.map_dim
        self.n_max_others_player = self.observation_spaces["others_player"].shape[0]
        self.n_max_unit = self.observation_spaces["unit"].shape[0]
        self.n_max_city = self.observation_spaces["city"].shape[0]
        self.n_max_others_unit = self.observation_spaces["others_unit"].shape[0]
        self.n_max_others_city = self.observation_spaces["others_city"].shape[0]

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
        self.others_player_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_player,
                self.others_player_dim,
            ),
            dtype=np.float32,
        )
        self.unit_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_unit,
                self.unit_dim,
            ),
            dtype=np.float32,
        )
        self.city_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_city,
                self.city_dim,
            ),
            dtype=np.float32,
        )
        self.others_unit_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_unit,
                self.others_unit_dim,
            ),
            dtype=np.float32,
        )
        self.others_city_input = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_city,
                self.others_city_dim,
            ),
            dtype=np.float32,
        )
        self.map_input = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *self.map_dim),
            dtype=np.float32,
        )

        self.others_player_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_player,
                1,
            ),
            dtype=np.int64,
        )
        self.unit_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_unit, 1),
            dtype=np.int64,
        )
        self.city_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_city, 1),
            dtype=np.int64,
        )
        self.others_unit_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_unit,
                1,
            ),
            dtype=np.int64,
        )
        self.others_city_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_others_city,
                1,
            ),
            dtype=np.int64,
        )

        self.rnn_hidden_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_rnn_layers,
                self.rnn_hidden_dim,
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
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.actor_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.actor_type_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.actor_type_dim),
            dtype=np.int64,
        )

        self.city_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.city_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.city_id_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_city, 1),
            dtype=np.int64,
        )

        self.city_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )  # TODO: check data type
        self.city_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.city_action_type_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_city,
                self.city_action_type_dim,
            ),
            dtype=np.int64,
        )

        self.unit_id_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.unit_id_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.unit_id_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.n_max_unit, 1),
            dtype=np.int64,
        )

        self.unit_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.unit_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.unit_action_type_masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.n_max_unit,
                self.unit_action_type_dim,
            ),
            dtype=np.int64,
        )

        self.gov_action_type_output = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1),
            dtype=np.int64,
        )
        self.gov_action_type_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.gov_action_type_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, self.gov_action_type_dim),
            dtype=np.int64,
        )

        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int64
        )
        self.bad_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.int64
        )

        self.step = 0

        self.valid_pos = np.full((self.n_rollout_threads,), -1, dtype=np.int64)
        self.num_to_distr_rew = np.zeros((self.n_rollout_threads,), dtype=np.int64)

    def get_mean_rewards(self):
        return np.mean(self.rewards)

    def insert(self, data):
        """Insert data into buffer."""
        (
            rules,
            player,
            others_player,
            unit,
            city,
            others_unit,
            others_city,
            map,
            others_player_mask,
            unit_mask,
            city_mask,
            others_unit_mask,
            others_city_mask,
            rnn_hidden_state,
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

        self.num_to_distr_rew += 1

        self.rules_input[self.step + 1] = rules.copy()
        self.player_input[self.step + 1] = player.copy()
        self.others_player_input[self.step + 1] = others_player.copy()
        self.unit_input[self.step + 1] = unit.copy()
        self.city_input[self.step + 1] = city.copy()
        self.others_unit_input[self.step + 1] = others_unit.copy()
        self.others_city_input[self.step + 1] = others_city.copy()
        self.map_input[self.step + 1] = map.copy()
        self.others_player_masks[self.step + 1] = others_player_mask.copy()
        self.unit_masks[self.step + 1] = unit_mask.copy()
        self.city_masks[self.step + 1] = city_mask.copy()
        self.others_unit_masks[self.step + 1] = others_unit_mask.copy()
        self.others_city_masks[self.step + 1] = others_city_mask.copy()
        self.rnn_hidden_states[self.step + 1] = rnn_hidden_state.copy()
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
        self.value_preds[self.step] = value_pred.copy()

        for env_id in range(self.n_rollout_threads):
            if actor_type[env_id][0] == 3:  # choose turn done
                self.valid_pos[env_id] = self.step
                rew_to_distr = reward[env_id][0] / self.num_to_distr_rew[env_id]
                distr_start_pos = max(self.step - self.num_to_distr_rew[env_id] + 1, 0)
                self.rewards[distr_start_pos : self.step + 1, env_id] = rew_to_distr
                self.num_to_distr_rew[env_id] = 0
            else:
                self.rewards[self.step][env_id][0] = 0

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """After an update, copy the data at the last step to the first position of the buffer."""
        self.rules_input[0] = self.rules_input[-1].copy()
        self.player_input[0] = self.player_input[-1].copy()
        self.others_player_input[0] = self.others_player_input[-1].copy()
        self.unit_input[0] = self.unit_input[-1].copy()
        self.city_input[0] = self.city_input[-1].copy()
        self.others_unit_input[0] = self.others_unit_input[-1].copy()
        self.others_city_input[0] = self.others_city_input[-1].copy()
        self.map_input[0] = self.map_input[-1].copy()
        self.others_player_masks[0] = self.others_player_masks[-1].copy()
        self.unit_masks[0] = self.unit_masks[-1].copy()
        self.city_masks[0] = self.city_masks[-1].copy()
        self.others_unit_masks[0] = self.others_unit_masks[-1].copy()
        self.others_city_masks[0] = self.others_city_masks[-1].copy()
        self.rnn_hidden_states[0] = self.rnn_hidden_states[-1].copy()
        self.actor_type_masks[0] = self.actor_type_masks[-1].copy()
        self.city_id_masks[0] = self.city_id_masks[-1].copy()
        self.city_action_type_masks[0] = self.city_action_type_masks[-1].copy()
        self.unit_id_masks[0] = self.unit_id_masks[-1].copy()
        self.unit_action_type_masks[0] = self.unit_action_type_masks[-1].copy()
        self.gov_action_type_masks[0] = self.gov_action_type_masks[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

        self.valid_pos = np.full((self.n_rollout_threads,), -1, dtype=np.int64)

    def compute_returns(self, next_value, value_normalizer=None):
        """Compute returns either as discounted sum of rewards, or using GAE.
        Args:
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
        """
        self.valid_last_pos = self.valid_pos.min()
        if (
            self.use_proper_time_limits
        ):  # consider the difference between truncation and termination
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.valid_last_pos + 1)):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        gae = self.bad_masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(self.valid_last_pos + 1)):
                    if value_normalizer is not None:  # use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        self.returns[step] = (
                            self.returns[step + 1] * self.gamma * self.masks[step + 1]
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] + (
                            1 - self.bad_masks[step + 1]
                        ) * self.value_preds[
                            step
                        ]
        else:  # do not consider the difference between truncation and termination, i.e. all done episodes are terminated
            if self.use_gae:  # use GAE
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.valid_last_pos + 1)):
                    if value_normalizer is not None:  # use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * value_normalizer.denormalize(self.value_preds[step + 1])
                            * self.masks[step + 1]
                            - value_normalizer.denormalize(self.value_preds[step])
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + value_normalizer.denormalize(
                            self.value_preds[step]
                        )
                    else:  # do not use ValueNorm
                        delta = (
                            self.rewards[step]
                            + self.gamma
                            * self.value_preds[step + 1]
                            * self.masks[step + 1]
                            - self.value_preds[step]
                        )
                        gae = (
                            delta
                            + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        )
                        self.returns[step] = gae + self.value_preds[step]
            else:  # do not use GAE
                self.returns[-1] = next_value
                for step in reversed(range(self.valid_last_pos + 1)):
                    self.returns[step] = (
                        self.returns[step + 1] * self.gamma * self.masks[step + 1]
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

        T, N = self.valid_last_pos + 1, num_envs_per_batch

        # prepare data for each mini batch
        for batch_id in range(num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            rules_batch = _flatten(T, N, self.rules_input[:T, ids])
            player_batch = _flatten(T, N, self.player_input[:T, ids])
            others_player_batch = _flatten(T, N, self.others_player_input[:T, ids])
            unit_batch = _flatten(T, N, self.unit_input[:T, ids])
            city_batch = _flatten(T, N, self.city_input[:T, ids])
            others_unit_batch = _flatten(T, N, self.others_unit_input[:T, ids])
            others_city_batch = _flatten(T, N, self.others_city_input[:T, ids])
            map_batch = _flatten(T, N, self.map_input[:T, ids])
            others_player_masks_batch = _flatten(
                T, N, self.others_player_masks[:T, ids]
            )
            unit_masks_batch = _flatten(T, N, self.unit_masks[:T, ids])
            city_masks_batch = _flatten(T, N, self.city_masks[:T, ids])
            others_unit_masks_batch = _flatten(T, N, self.others_unit_masks[:T, ids])
            others_city_masks_batch = _flatten(T, N, self.others_city_masks[:T, ids])
            rnn_hidden_states_batch = self.rnn_hidden_states[0:1, ids]
            old_value_preds_batch = _flatten(T, N, self.value_preds[:T, ids])
            return_batch = _flatten(T, N, self.returns[:T, ids])
            adv_targ = _flatten(T, N, advantages[:T, ids])
            actor_type_batch = _flatten(T, N, self.actor_type_output[:T, ids])
            old_actor_type_log_probs_batch = _flatten(
                T, N, self.actor_type_log_probs[:T, ids]
            )
            actor_type_masks_batch = _flatten(T, N, self.actor_type_masks[:T, ids])
            city_id_batch = _flatten(T, N, self.city_id_output[:T, ids])
            old_city_id_log_probs_batch = _flatten(
                T, N, self.city_id_log_probs[:T, ids]
            )
            city_id_masks_batch = _flatten(T, N, self.city_id_masks[:T, ids])
            city_action_type_batch = _flatten(
                T, N, self.city_action_type_output[:T, ids]
            )
            old_city_action_type_log_probs_batch = _flatten(
                T, N, self.city_action_type_log_probs[:T, ids]
            )
            city_action_type_masks_batch = _flatten(
                T, N, self.city_action_type_masks[:T, ids]
            )
            unit_id_batch = _flatten(T, N, self.unit_id_output[:T, ids])
            old_unit_id_log_probs_batch = _flatten(
                T, N, self.unit_id_log_probs[:T, ids]
            )
            unit_id_masks_batch = _flatten(T, N, self.unit_id_masks[:T, ids])
            unit_action_type_batch = _flatten(
                T, N, self.unit_action_type_output[:T, ids]
            )
            old_unit_action_type_log_probs_batch = _flatten(
                T, N, self.unit_action_type_log_probs[:T, ids]
            )
            unit_action_type_masks_batch = _flatten(
                T, N, self.unit_action_type_masks[:T, ids]
            )
            gov_action_type_batch = _flatten(T, N, self.gov_action_type_output[:T, ids])
            old_gov_action_type_log_probs_batch = _flatten(
                T, N, self.gov_action_type_log_probs[:T, ids]
            )
            gov_action_type_masks_batch = _flatten(
                T, N, self.gov_action_type_masks[:T, ids]
            )
            masks_batch = _flatten(T, N, self.masks[:T, ids])
            bad_masks_batch = _flatten(T, N, self.bad_masks[:T, ids])

            rnn_hidden_states_batch = rnn_hidden_states_batch.squeeze(0)

            yield (
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
            )
