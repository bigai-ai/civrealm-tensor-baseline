import pytest
from civtensor.runners.runner import Runner
from civtensor.utils.configs_tools import get_defaults_yaml_args, update_args


@pytest.fixture
def runner():
    args = {'algo':'ppo', 'env': "freeciv_tensor_env", "exp_name":"installtest"}
    algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    algo_args["train"]["episode_length"]= 200
    runner = Runner(args, algo_args, env_args)
    yield runner
    runner.close()

def test_tensor_env(runner):
    runner.run()
