import os
import re

import pandas as pd
from tbparse import SummaryReader

TbDefaultPattern = ".*/freeciv_tensor_env/(?P<task>.*)/ppo/.*/seed\\-(?P<seed>\\d*)\\-(?P<time>[\\d\\-]*)"


class TbSummary:
    def __init__(self, logdir, pattern=TbDefaultPattern):
        self.df = pd.DataFrame()
        self.dir_parser = re.compile(pattern)
        dfs = []
        for root, directories, files in os.walk(logdir):
            if "logs" in directories:
                matches = self.dir_parser.match(root)
                if matches == None:
                    continue
                (task, seed, time) = matches.groups()
                is_fullgame = task.split(" ")[0] == "fullgame"
                run_df = SummaryReader(os.path.join(root, "logs")).scalars
                run_df["task"] = task
                run_df["seed"] = seed
                run_df["time"] = time
                run_df["fullgame"] = is_fullgame
                dfs.append(run_df)
        self.df = pd.concat(dfs)

    def tasks(self):
        return self.df['task'].unique()
