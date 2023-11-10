"""Train an algorithm."""
import argparse
import os
import subprocess
import time

import yaml

from civtensor.utils.configs_tools import get_defaults_yaml_args, update_args

runner_map = {
    "cs": [
        "run_battle_ancient_era_easy.yaml",
        "run_battle_ancient_era_hard.yaml",
        "run_battle_ancient_era_normal.yaml",
        "run_battle_attack_city_easy.yaml",
        "run_battle_attack_city_hard.yaml",
        "run_battle_attack_city_normal.yaml",
        "run_battle_defend_city_easy.yaml",
        "run_battle_defend_city_hard.yaml",
        "run_battle_defend_city_normal.yaml",
        "run_battle_industry_era_easy.yaml",
        "run_battle_industry_era_hard.yaml",
        "run_battle_industry_era_normal.yaml",
        "run_battle_info_era_easy.yaml",
        "run_battle_info_era_hard.yaml",
    ],
    "qsy": [
        "run_battle_info_era_normal.yaml",
        "run_battle_medieval_easy.yaml",
        "run_battle_medieval_hard.yaml",
        "run_battle_medieval_normal.yaml",
        "run_battle_modern_era_easy.yaml",
        "run_battle_modern_era_hard.yaml",
        "run_battle_modern_era_normal.yaml",
        "run_battle_naval_easy.yaml",
        "run_battle_naval_hard.yaml",
        "run_battle_naval_modern_easy.yaml",
        "run_battle_naval_modern_hard.yaml",
        "run_battle_naval_modern_normal.yaml",
        "run_battle_naval_normal.yaml",
        "run_development_build_city_easy.yaml",
    ],
    "lyx": [
        "run_development_build_city_hard.yaml",
        "run_development_build_city_normal.yaml",
        "run_development_build_infra_easy.yaml",
        "run_development_build_infra_hard.yaml",
        "run_development_build_infra_normal.yaml",
        "run_development_citytile_wonder_easy.yaml",
        "run_development_citytile_wonder_hard.yaml",
        "run_development_citytile_wonder_normal.yaml",
        "run_development_transport_easy.yaml",
        "run_development_transport_hard.yaml",
        "run_development_transport_normal.yaml",
        "run_diplomacy_trade_tech_easy.yaml",
        "run_diplomacy_trade_tech_hard.yaml",
        "run_diplomacy_trade_tech_normal.yaml",
    ],
}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--runner",
        type=str,
        help="runner, must be in [cs, qsy, lys]",
    )
    parser.add_argument(
        "--webdir",
        type=str,
        default=os.path.join("..", "..", "freeciv-web"),
        help="directory of freeciv-web repo",
    )
    known_args = parser.parse_known_args()[0]
    webdir = known_args.webdir
    runner = known_args.runner

    assert os.path.exists(
        webdir
    ), f"Cannot find directory of freeciv-web repo, {webdir}"

    def web_docker_cmd(cmd):
        subprocess.call(f"cd {webdir} && " + cmd, shell=True, executable="/bin/bash")

    args = {"algo": "ppo", "env": "freeciv_tensor_env", "exp_name": "iclr"}

    config_args = []
    for config_file in os.listdir("run_configs"):
        if config_file not in runner_map[runner]:
            print(f"runner for {runner_map[runner]}")
            print('config_file not for me:', config_file)
            continue
        print(f"Loading {config_file}")
        with open("./run_configs/" + config_file, encoding="utf-8") as file:
            all_config = yaml.safe_load(file)
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
        config_args.append(
            (
                {"args": args, "algo_args": algo_args, "env_args": env_args},
                all_config["run_times"],
            )
        )
        print(f"Task name {env_args['task_name']} loaded!")

    # start training
    from civtensor.runners.runner import Runner

    for config_arg in config_args:
        (runner_args, run_times) = config_arg
        for i in range(run_times):
            try:
                print(
                    f"\n==== Runing for task {runner_args['env_args']['task_name']} for the {i}-th time ===="
                )
                runner = Runner(**runner_args)
                runner.run()
                print(
                    f"\n==== Runner for task {runner_args['env_args']['task_name']} the {i}-th time finished! ===="
                )
            except Exception as e:
                print(
                    f"\n==== Runner for task {runner_args['env_args']['task_name']} aborted! ===="
                )
                print(e)
            try:
                runner.close()
                print(
                    f"\n==== Runner for task {runner_args['env_args']['task_name']} closed! ===="
                )
            except Exception as e:
                print(e)
            print(f"!!! Freeciv-web Docker in {webdir} closing down... !!! ")
            web_docker_cmd("docker compose down")
            print(f"!!! Freeciv-web Docker in {webdir} restarting... !!! ")
            web_docker_cmd("docker compose up -d")
            time.sleep(15)


if __name__ == "__main__":
    main()
