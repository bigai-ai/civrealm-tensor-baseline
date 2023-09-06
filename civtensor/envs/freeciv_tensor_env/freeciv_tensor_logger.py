import time
import os
import numpy as np


class FreecivTensorLogger:
    def __init__(self, args, algo_args, env_args, writter, run_dir):
        """Initialize the logger."""
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = env_args["task_name"]
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"
        )

    def init(self, episodes):
        """Initialize the logger."""
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            self.algo_args["train"]["n_rollout_threads"]
        )
        self.done_episodes_rewards = []

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""
        (
            rules,
            player,
            other_players,
            unit,
            city,
            others_unit,
            others_city,
            map,
            other_players_mask,
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
        done = np.logical_not(mask)
        reward_env = reward.flatten()
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if done[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t])
                self.train_episode_rewards[t] = 0

    def episode_log(self, train_info, buffer):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        train_info["average_step_rewards"] = buffer.get_mean_rewards()
        self.log_train(train_info)

        print("Average step reward is {}.".format(train_info["average_step_rewards"]))

        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = []

    def eval_init(self):
        """Initialize the logger for evaluation."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            rules,
            player,
            other_players,
            unit,
            city,
            others_unit,
            others_city,
            map,
            other_players_mask,
            unit_mask,
            city_mask,
            others_unit_mask,
            others_city_mask,
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
        ) = eval_data
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(reward[eval_i])

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": self.eval_episode_rewards,
            "eval_max_episode_rewards": [np.max(self.eval_episode_rewards)],
        }
        self.log_env(eval_env_infos)
        eval_avg_rew = np.mean(self.eval_episode_rewards)
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()

    def log_train(self, train_info):
        """Log training information."""
        # log critic
        for k, v in train_info.items():
            self.writter.add_scalars(k, {k: v}, self.total_num_steps)

    def log_env(self, env_infos):
        """Log environment information."""
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, self.total_num_steps)

    def close(self):
        """Close the logger."""
        self.log_file.close()
