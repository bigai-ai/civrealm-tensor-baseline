import setproctitle

import numpy as np
import torch

from civtensor.algorithms.ppo import PPO
from civtensor.common.buffer import Buffer
from civtensor.common.valuenorm import ValueNorm
from civtensor.envs.freeciv_tensor_env.freeciv_tensor_logger import FreecivTensorLogger
from civtensor.utils.envs_tools import set_seed, make_train_env, make_eval_env
from civtensor.utils.models_tools import init_device
from civtensor.utils.configs_tools import init_dir, save_config
from civtensor.utils.trans_tools import _t2n


class Runner:
    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.rnn_hidden_dim = algo_args["model"]["rnn_hidden_dim"]
        self.n_rnn_layers = algo_args["model"]["n_rnn_layers"]

        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            algo_args["seed"]["seed"],
            logger_path=algo_args["logger"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        self.envs = make_train_env(
            args["env"],
            algo_args["seed"]["seed"],
            algo_args["train"]["n_rollout_threads"],
            env_args,
        )
        self.eval_envs = (
            make_eval_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_eval_rollout_threads"],
                env_args,
            )
            if algo_args["eval"]["use_eval"]
            else None
        )

        print("state_spaces: ", self.envs.state_spaces)
        print("action_spaces: ", self.envs.action_spaces)

        self.algo = PPO(
            {**algo_args["model"], **algo_args["algo"]},
            self.envs.state_spaces,
            self.envs.action_spaces,
            device=self.device,
        )

        self.buffer = Buffer(
            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
            self.envs.state_spaces,
            self.envs.action_spaces,
        )

        if self.algo_args["train"]["use_valuenorm"] is True:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self.logger = FreecivTensorLogger(
            args, algo_args, env_args, self.writter, self.run_dir
        )
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def run(self):
        print("start training")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                self.algo.lr_decay(episode, episodes)

            self.logger.episode_init(episode)

            self.prep_rollout()
            for step in range(self.algo_args["train"]["episode_length"]):
                with torch.no_grad():
                    (
                        actor_type,
                        actor_type_log_prob,
                        city_id,
                        city_id_log_prob,
                        city_action_type,
                        city_action_type_log_prob,
                        unit_id,
                        unit_id_log_prob,
                        unit_action_type,
                        unit_action_type_log_prob,
                        gov_action_type,
                        gov_action_type_log_prob,
                        value_pred,
                        rnn_hidden_state,
                    ) = self.algo.agent(
                        self.buffer.rules_input[step],
                        self.buffer.player_input[step],
                        self.buffer.other_players_input[step],
                        self.buffer.units_input[step],
                        self.buffer.cities_input[step],
                        self.buffer.other_units_input[step],
                        self.buffer.other_cities_input[step],
                        self.buffer.civmap_input[step],
                        self.buffer.other_players_masks[step],
                        self.buffer.units_masks[step],
                        self.buffer.cities_masks[step],
                        self.buffer.other_units_masks[step],
                        self.buffer.other_cities_masks[step],
                        self.buffer.actor_type_masks[step],
                        self.buffer.city_id_masks[step],
                        self.buffer.city_action_type_masks[step],
                        self.buffer.unit_id_masks[step],
                        self.buffer.unit_action_type_masks[step],
                        self.buffer.gov_action_type_masks[step],
                        self.buffer.rnn_hidden_states[
                            step
                        ],  # use previous rnn hidden state
                        self.buffer.masks[step],
                        deterministic=False,
                    )

                actor_type = _t2n(actor_type)
                actor_type_log_prob = _t2n(actor_type_log_prob)
                city_id = _t2n(city_id)
                city_id_log_prob = _t2n(city_id_log_prob)
                city_action_type = _t2n(city_action_type)
                city_action_type_log_prob = _t2n(city_action_type_log_prob)
                unit_id = _t2n(unit_id)
                unit_id_log_prob = _t2n(unit_id_log_prob)
                unit_action_type = _t2n(unit_action_type)
                unit_action_type_log_prob = _t2n(unit_action_type_log_prob)
                gov_action_type = _t2n(gov_action_type)
                gov_action_type_log_prob = _t2n(gov_action_type_log_prob)
                value_pred = _t2n(value_pred)
                rnn_hidden_state = _t2n(rnn_hidden_state)

                (
                    rules,
                    player,
                    other_players,
                    units,
                    cities,
                    other_units,
                    other_cities,
                    civmap,
                    other_players_mask,
                    units_mask,
                    cities_mask,
                    other_units_mask,
                    other_cities_mask,
                    actor_type_mask,
                    city_id_mask,
                    city_action_type_mask,
                    unit_id_mask,
                    unit_action_type_mask,
                    gov_action_type_mask,
                    reward,
                    term,
                    trunc,
                ) = self.envs.step(
                    actor_type,
                    city_id,
                    city_action_type,
                    unit_id,
                    unit_action_type,
                    gov_action_type,
                )  # no info at the moment

                # mask: 1 if not done, 0 if done
                done = np.logical_or(term, trunc)  # (n_rollout_threads, 1)
                mask = np.logical_not(done)  # (n_rollout_threads, 1)

                # bad_mask use 0 to denote truncation and 1 to denote termination or not done
                bad_mask = np.logical_not(trunc)

                # reset certain rnn hidden state
                done_env = done.squeeze(1)
                rnn_hidden_state[done_env == True] = np.zeros(
                    (
                        (done_env == True).sum(),
                        self.n_rnn_layers,
                        self.rnn_hidden_dim,
                    )
                )

                data = (
                    rules,
                    player,
                    other_players,
                    units,
                    cities,
                    other_units,
                    other_cities,
                    civmap,
                    other_players_mask,
                    units_mask,
                    cities_mask,
                    other_units_mask,
                    other_cities_mask,
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
                )

                self.buffer.insert(data)

                self.logger.per_step(data)

            self.compute()
            self.prep_training()

            train_info = self.train()

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    train_info,
                    self.buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def warmup(self):
        (
            rules,
            player,
            other_players,
            units,
            cities,
            other_units,
            other_cities,
            civmap,
            other_players_mask,
            units_mask,
            cities_mask,
            other_units_mask,
            other_cities_mask,
            actor_type_mask,
            city_id_mask,
            city_action_type_mask,
            unit_id_mask,
            unit_action_type_mask,
            gov_action_type_mask,
        ) = self.envs.reset()
        self.buffer.rules_input[0] = rules.copy()
        self.buffer.player_input[0] = player.copy()
        self.buffer.other_players_input[0] = other_players.copy()
        self.buffer.units_input[0] = units.copy()
        self.buffer.cities_input[0] = cities.copy()
        self.buffer.other_units_input[0] = other_units.copy()
        self.buffer.other_cities_input[0] = other_cities.copy()
        self.buffer.civmap_input[0] = civmap.copy()
        self.buffer.other_players_masks[0] = other_players_mask.copy()
        self.buffer.units_masks[0] = units_mask.copy()
        self.buffer.cities_masks[0] = cities_mask.copy()
        self.buffer.other_units_masks[0] = other_units_mask.copy()
        self.buffer.other_cities_masks[0] = other_cities_mask.copy()
        self.buffer.actor_type_masks[0] = actor_type_mask.copy()
        self.buffer.city_id_masks[0] = city_id_mask.copy()
        self.buffer.city_action_type_masks[0] = city_action_type_mask.copy()
        self.buffer.unit_id_masks[0] = unit_id_mask.copy()
        self.buffer.unit_action_type_masks[0] = unit_action_type_mask.copy()
        self.buffer.gov_action_type_masks[0] = gov_action_type_mask.copy()

    @torch.no_grad()
    def compute(self):
        (
            actor_type,
            actor_type_log_prob,
            city_id,
            city_id_log_prob,
            city_action_type,
            city_action_type_log_prob,
            unit_id,
            unit_id_log_prob,
            unit_action_type,
            unit_action_type_log_prob,
            gov_action_type,
            gov_action_type_log_prob,
            value_pred,
            rnn_hidden_state,
        ) = self.algo.agent(
            self.buffer.rules_input[-1],
            self.buffer.player_input[-1],
            self.buffer.other_players_input[-1],
            self.buffer.units_input[-1],
            self.buffer.cities_input[-1],
            self.buffer.other_units_input[-1],
            self.buffer.other_cities_input[-1],
            self.buffer.civmap_input[-1],
            self.buffer.other_players_masks[-1],
            self.buffer.units_masks[-1],
            self.buffer.cities_masks[-1],
            self.buffer.other_units_masks[-1],
            self.buffer.other_cities_masks[-1],
            self.buffer.actor_type_masks[-1],
            self.buffer.city_id_masks[-1],
            self.buffer.city_action_type_masks[-1],
            self.buffer.unit_id_masks[-1],
            self.buffer.unit_action_type_masks[-1],
            self.buffer.gov_action_type_masks[-1],
            self.buffer.rnn_hidden_states[-1],  # use previous rnn hidden state
            self.buffer.masks[-1],
            deterministic=False,
        )
        self.buffer.compute_returns(value_pred, self.value_normalizer)

    def after_update(self):
        self.buffer.after_update()

    def train(self):
        advantages = self.buffer.returns[:-1] - self.value_normalizer.denormalize(
            self.buffer.value_preds[:-1]
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        train_info = self.algo.train(self.buffer, advantages, self.value_normalizer)
        return train_info

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        (
            rules,
            player,
            other_players,
            units,
            cities,
            other_units,
            other_cities,
            civmap,
            other_players_mask,
            units_mask,
            cities_mask,
            other_units_mask,
            other_cities_mask,
            actor_type_mask,
            city_id_mask,
            city_action_type_mask,
            unit_id_mask,
            unit_action_type_mask,
            gov_action_type_mask,
        ) = self.eval_envs.reset()

        eval_rnn_hidden_state = np.zeros(
            (
                self.algo_args["train"]["n_eval_rollout_threads"],
                self.n_rnn_layers,
                self.rnn_hidden_dim,
            ),
            dtype=np.float32,
        )

        eval_mask = np.ones(
            (self.algo_args["train"]["n_eval_rollout_threads"], 1), dtype=np.float32
        )

        while True:
            (
                actor_type,
                actor_type_log_prob,
                city_id,
                city_id_log_prob,
                city_action_type,
                city_action_type_log_prob,
                unit_id,
                unit_id_log_prob,
                unit_action_type,
                unit_action_type_log_prob,
                gov_action_type,
                gov_action_type_log_prob,
                value_pred,
                eval_rnn_hidden_state,
            ) = self.algo.agent(
                rules,
                player,
                other_players,
                units,
                cities,
                other_units,
                other_cities,
                civmap,
                other_players_mask,
                units_mask,
                cities_mask,
                other_units_mask,
                other_cities_mask,
                actor_type_mask,
                city_id_mask,
                city_action_type_mask,
                unit_id_mask,
                unit_action_type_mask,
                gov_action_type_mask,
                eval_rnn_hidden_state,
                eval_mask,  # TODO check whether logic related to mask is correct
                deterministic=True,
            )

            actor_type = _t2n(actor_type)
            actor_type_log_prob = _t2n(actor_type_log_prob)
            city_id = _t2n(city_id)
            city_id_log_prob = _t2n(city_id_log_prob)
            city_action_type = _t2n(city_action_type)
            city_action_type_log_prob = _t2n(city_action_type_log_prob)
            unit_id = _t2n(unit_id)
            unit_id_log_prob = _t2n(unit_id_log_prob)
            unit_action_type = _t2n(unit_action_type)
            unit_action_type_log_prob = _t2n(unit_action_type_log_prob)
            gov_action_type = _t2n(gov_action_type)
            gov_action_type_log_prob = _t2n(gov_action_type_log_prob)
            value_pred = _t2n(value_pred)
            eval_rnn_hidden_state = _t2n(eval_rnn_hidden_state)

            (
                rules,
                player,
                other_players,
                units,
                cities,
                other_units,
                other_cities,
                civmap,
                other_players_mask,
                units_mask,
                cities_mask,
                other_units_mask,
                other_cities_mask,
                actor_type_mask,
                city_id_mask,
                city_action_type_mask,
                unit_id_mask,
                unit_action_type_mask,
                gov_action_type_mask,
                reward,
                term,
                trunc,
            ) = self.eval_envs.step(
                actor_type,
                city_id,
                city_action_type,
                unit_id,
                unit_action_type,
                gov_action_type,
            )  # no info at the moment

            # mask: 1 if not done, 0 if done
            done = np.logical_or(term, trunc)  # (n_rollout_threads, 1)
            mask = np.logical_not(done)  # (n_rollout_threads, 1)

            # bad_mask use 0 to denote truncation and 1 to denote termination or not done
            bad_mask = np.logical_not(trunc)

            # reset certain rnn hidden state
            done_env = done.squeeze(1)
            eval_rnn_hidden_state[done_env == True] = np.zeros(
                (
                    (done_env == True).sum(),
                    self.n_rnn_layers,
                    self.rnn_hidden_dim,
                )
            )

            eval_data = (
                rules,
                player,
                other_players,
                units,
                cities,
                other_units,
                other_cities,
                civmap,
                other_players_mask,
                units_mask,
                cities_mask,
                other_units_mask,
                other_cities_mask,
                eval_rnn_hidden_state,
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
            )

            self.logger.eval_per_step(eval_data)

            for eval_i in range(self.algo_args["train"]["n_eval_rollout_threads"]):
                if done[eval_i][0]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["train"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)
                break

    def prep_training(self):
        """Prepare for training."""
        self.algo.prep_training()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.algo.prep_rollout()

    def save(self):
        torch.save(
            self.algo.agent.state_dict(),
            str(self.save_dir) + "/agent.pt",
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def restore(self):
        agent_state_dict = torch.load(
            str(self.algo_args["train"]["model_dir"]) + "/agent.pt"
        )
        self.algo.agent.load_state_dict(agent_state_dict)
        if self.value_normalizer is not None:
            value_normalizer_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"]) + "/value_normalizer" + ".pt"
            )
            self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()
        self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
        self.writter.close()
        self.logger.close()
