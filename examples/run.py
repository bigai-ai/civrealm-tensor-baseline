"""Train an algorithm."""
import argparse
import yaml
from civtensor.utils.configs_tools import get_defaults_yaml_args, update_args
import os
import ray
import time


def main():
    """Main function."""
    args = {'algo':'ppo',"env":"freeciv_tensor_env","exp_name":"iclr"}

    config_args = []
    for config_file in os.listdir("run_configs"):
        print(f"Loading {config_file}")
        with open("./run_configs/"+config_file, encoding="utf-8") as file:
            all_config = yaml.safe_load(file)
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
        config_args.append(({'args':args,"algo_args":algo_args,"env_args":env_args},all_config['run_times']))
        print(f"Task name {env_args['task_name']} loaded!")

    # start training
    from civtensor.runners.runner import Runner

    for config_arg in config_args:
        (runner_args, run_times) = config_arg
        for i in range(run_times):
            try:
                print(f"Runing for task {runner_args['env_args']['task_name']} for the {i}-th time")
                runner = Runner(**runner_args)
                runner.run()
                print(f"Runner for task {runner_args['env_args']['task_name']} the {i}-th time finished!")
            except Exception as e:
                print(f"Runner for task {runner_args['env_args']['task_name']} aborted!")
                print(e)
            try:
                runner.close()
            except Exception as e:
                print(e)
            time.sleep(15)
            


if __name__ == "__main__":
    main()
