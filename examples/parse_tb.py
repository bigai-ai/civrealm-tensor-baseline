import os
import re
from functools import cached_property

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from tbparse import SummaryReader

TbDefaultPattern = ".*/freeciv_tensor_env/(?P<task>.*)/ppo/.*/seed\\-(?P<seed>\\d*)\\-(?P<time>[\\d\\-]*)"


class TbSummary:
    tag_replacement = {
        "average_step_rewards": "step reward",
        "train_episode_rewards": "episode reward",
        "train_episode_scores": "episode score",
        "train_episode_success": "success rate",
        "train_episode_cities": "cities",
        "train_episode_economics": "economics",
        "train_episode_gold": "gold",
        "train_episode_land_area": "land area",
        "train_episode_military_units": "military units",
        "train_episode_population": "population",
        "train_episode_production": "production",
        "train_episode_research_speed": "research speed",
        "train_episode_researched_techs": "researched techs",
        "train_episode_rewards": "rewards",
        "train_episode_scores": "scores",
        "train_episode_settled_area": "settled area",
        "train_episode_units_built": "units built",
        "train_episode_units_killed": "units killed",
        "train_episode_units_lost": "units lost",
        "train_episode_units_used": "units used",
        "train_episode_wonders": "wonders",
    }
    level_ordr = ["easy", "normal", "hard"]

    def __init__(self, logdir, pattern=TbDefaultPattern):
        self.df = pd.DataFrame()
        self.dir_parser = re.compile(pattern)
        self.logdir = os.path.expanduser(logdir)

        dfs = []

        for root, directories, _ in os.walk(self.logdir):
            if "logs" in directories:
                matches = self.dir_parser.match(root)
                if matches == None:
                    continue
                (task, seed, time) = matches.groups()
                if task == "classic":
                    continue
                is_fullgame = task.split(" ")[0] in ["fullgame", "classic"]
                run_df = SummaryReader(os.path.join(root, "logs")).scalars
                run_df["task"] = task
                run_df["seed"] = seed
                run_df["time"] = time
                run_df["fullgame"] = is_fullgame
                run_df.replace({"tag": self.tag_replacement}, inplace=True)
                dfs.append(run_df)
        self.df = pd.concat(dfs)

    def parse(self, logdir="", replace=False):
        logdir = self.logdir if logdir == "" else os.path.expanduser(logdir)
        dfs = []

        for root, directories, _ in os.walk(logdir):
            if "logs" in directories:
                matches = self.dir_parser.match(root)
                if matches == None:
                    continue
                (task, seed, time) = matches.groups()
                is_fullgame = task.split(" ")[0] in ["fullgame", "classic"]
                run_df = SummaryReader(os.path.join(root, "logs")).scalars
                run_df["task"] = task
                run_df["seed"] = seed
                run_df["time"] = time
                run_df["fullgame"] = is_fullgame
                run_df.replace({"tag": self.tag_replacement}, inplace=True)
                dfs.append(run_df)
        if replace:
            self.df = pd.concat(dfs)
        else:
            self.df = pd.concat([self.df] + dfs)

    @cached_property
    def tasks(self):
        result = self.df[["task", "seed"]].drop_duplicates()["task"].value_counts()
        print("Total items: ", result.sum())
        return result

    @cached_property
    def minitasks(self):
        df = self.df[self.df["fullgame"] == False].drop(columns=["fullgame"])

        def get_task_type(task):
            return task.split(" ")[0]

        def get_task_level(task):
            return task.split(" ")[1]

        df["type"] = df["task"].apply(get_task_type)
        df["level"] = df["task"].apply(get_task_level)

        # TODO: revert this after naval level is corrected

        idx = (df["type"] == "battle_naval_modern") | (df["type"] == "battle_naval")
        df.loc[idx, "level"] = df[idx]["level"].replace(
            {"easy": "hard", "hard": "easy"}
        )

        return df

    @property
    def fullgames(self):
        df = self.df[self.df["fullgame"] == True].drop(columns=["fullgame"])

        # HACK: Mannly merge discontinued fullgame runs seed=[07985, 01543]
        # df.loc[df["seed"] == "01543", "step"] += 109000
        # df.loc[df["seed"] == "01543", ["seed", "time"]] = [
        #     "07985",
        #     df[df["seed"] == "07985"]["time"].unique()[0],
        # ]
        df.loc[df.seed == "07814","step"] -= 650000
        return df[(df.seed == "07814") & (df.step>0)]

    @staticmethod
    def plot_metrics(
        data,
        x_label,
        y_label,
        title="",
        hue_label=None,
        style_label=None,
        hue_order=None,
        style_order=None,
        font_scale=1.7,
        tick_style=None,
        other_settings={},
    ):
        sns.set_theme()
        sns.set_context("paper")
        sns.set_style("darkgrid")
        sns.set(font_scale=font_scale)
        if title == "":
            title = f"Plot of {y_label} against turns"
        sns.lineplot(
            data=data,
            x=x_label,
            y=y_label,
            hue=hue_label,
            style=style_label,
            hue_order=hue_order,
            style_order=style_order,
        ).set(title=title, **other_settings)

        ax = plt.gca()

        assert tick_style in [None, "K", "M"]

        def tick_transform(x, pos):
            if tick_style == "K":
                return f"{int(x/1000)}K"
            elif tick_style == "M":
                return "{:,.1f}M".format(x / 1_000_000)
            else:
                return x

        ax.xaxis.set_major_formatter(FuncFormatter(tick_transform))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels)

    def plot_minitask(self, task_type, y_label, x_label="step"):
        if "step" in self.minitasks.columns:
            minitasks = self.minitasks.rename({"step": x_label})
        elif x_label in self.minitasks.columns:
            minitasks = self.minitasks
        else:
            raise ValueError(f"Cannot find column name {x_label} in minitasks")

        minitasks = minitasks[
            (minitasks["type"] == task_type) & (minitasks["tag"] == y_label)
        ][["seed", "value","level", x_label]]
        self.plot_metrics(
            data=minitasks,
            x_label=x_label,
            y_label="value",
            title=None,
            hue_label="level",
            hue_order=self.level_ordr,
            tick_style="K",
            other_settings={"ylabel": y_label.title(), "xlabel": x_label.title()},
        )

    def plot_fullgame(self, y_label, x_label="step"):
        if "step" in self.fullgames.columns:
            fullgames = self.fullgames.rename({"step": x_label})
        elif x_label in self.fullgames.columns:
            fullgames = self.fullgames
        else:
            raise ValueError(f"Cannot find column name {x_label} in fullgames")

        # fullgames = fullgames[fullgames["tag"] == y_label]
        fullgames = fullgames[fullgames["tag"] == y_label][[x_label, "seed", "value"]]
        # breakpoint()
        for seed in fullgames.seed.unique():
            index = fullgames.seed == seed
            if fullgames[index].shape[0] < 20:
                continue
            filtered = fullgames[index].rolling(window=50, on="step", closed="both")
            fullgames.loc[index, "mean"] = filtered.mean()["value"]
            fullgames.loc[index, "std"] = filtered.std()["value"]
            fullgames = fullgames.dropna()

        self.plot_metrics(
            data=fullgames,
            x_label=x_label,
            y_label="mean",
            title=None,
            tick_style="M",
            other_settings={"ylabel": y_label.title(), "xlabel": x_label.title()},
        )
        ax = plt.gca()
        ax.fill_between(
            fullgames[x_label],
            fullgames["mean"] - fullgames["std"],
            fullgames["mean"] + fullgames["std"],
            alpha=0.2,
        )

    def plot_minitask_result(self, subplot=False, save=True):
        os.makedirs("./minitask", exist_ok=True)
        fig_dir = "./minitask"
        if subplot:
            raise NotImplemented()
        else:
            for task_type in self.minitasks["type"].unique():
                for metric in self.minitasks["tag"].unique():
                    plt.figure()
                    self.plot_minitask(task_type, y_label=metric)
                    if save:
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(fig_dir, f"{task_type}_{metric}.jpg"),
                            dpi=500,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        # plt.savefig(os.path.join(fig_dir, f"{task_type}_{metric}.pgf"))
                        plt.close()

    def plot_fullgame_result(self, subplot=False, save=True):
        os.makedirs("./fullgame", exist_ok=True)
        fig_dir = "./fullgame"
        if subplot:
            raise NotImplemented()
        else:
            for metric in self.fullgames["tag"].unique():
                plt.figure()
                self.plot_fullgame(y_label=metric)
                if save:
                    plt.tight_layout()
                    plt.legend([],[], frameon=False)
                    plt.savefig(
                        os.path.join(fig_dir, f"fullgame_{metric}.jpg"),
                        dpi=500,
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    # plt.savefig(os.path.join(fig_dir, f"fullgame_{metric}.pgf"))
                    plt.close()


if __name__ == "__main__":
    summary = TbSummary('~/nfs')
    summary.plot_minitask_result()
    summary.plot_fullgame_result()
