import setproctitle

import torch

from civtensor.algorithms.ppo import PPO
from civtensor.common.buffer import Buffer
from civtensor.common.valuenorm import ValueNorm
from civtensor.envs.freeciv_tensor_env.freeciv_tensor_logger import FreecivTensorLogger
from civtensor.utils.envs_tools import set_seed, make_train_env, make_eval_env
from civtensor.utils.models_tools import init_device
from civtensor.utils.configs_tools import init_dir, save_config


class Runner:
    def __init__(self, args, algo_args, env_args):
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.lstm_hidden_dim = algo_args["model"]["lstm_hidden_dim"]
        self.n_lstm_layers = algo_args["model"]["n_lstm_layers"]

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
        self.envs = (
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
                ) = self.collect(step)

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
                    actor_type_mask,
                    city_id_mask,
                    city_action_type_mask,
                    unit_id_mask,
                    unit_action_type_mask,
                    gov_action_type_mask,
                    reward,
                    term,
                    trunc,
                ) = self.envs.reset()  # no info at the moment

    def warmup(self):
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
        self.buffer.map_input[0] = map.copy()
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

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

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
        raise NotImplementedError
