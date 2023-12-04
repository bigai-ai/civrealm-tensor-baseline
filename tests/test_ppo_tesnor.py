import pytest
import os
import yaml
from civtensor.runners.runner import Runner
from civrealm.envs.freeciv_minitask_env import MinitaskType, MinitaskDifficulty

MinitaskSet = [
    f"{task} {level}"
    for task in MinitaskType.list()
    for level in MinitaskDifficulty.list()
]


@pytest.fixture(params=MinitaskSet + ["fullgame"])
def config(request):
    dir = os.path.dirname(__file__)
    with open(os.path.join(dir, "test_config.yaml"), encoding="utf-8") as file:
        all_config = yaml.safe_load(file)

    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]
    env_args["task_name"] = request.param

    args = {"algo": "ppo", "env": "freeciv_tensor_env", "exp_name": "installtest"}
    return args, algo_args, env_args


@pytest.fixture
def runner(config):
    runner = Runner(*config)
    yield runner
    runner.close()


def test_tensor_env(runner):
    runner.run()
