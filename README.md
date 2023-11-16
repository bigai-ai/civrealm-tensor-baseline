# Freeciv Tensor Baseline

## Quick Start
First install
```
cd freeciv-tensor-baseline
pip install -e .
```

then run the training script
```sh
cd examples
./train.sh # or python train.py
```
## Fullgame
Edit `freeciv-tensor-baseline/civtensor/configs/envs_cfgs/freeciv_tensor_env.yaml`

Change `task_name` to `fullgame`

```yaml
task_name: fullgame
```

## Minigame

### Run all minitasks

freeciv-tensor-baseline/examples/run.py will run all minitasks specified in `examples/run_configs` including $`12\text{ types}\times 3\text{ levels} = 36 \text{ runs}`$
```sh
cd exmaples
python run.py --webdir $civrealm-dir
```
where `$civrealm-dir` should specify the path of your local **civrelam** repo.

### Expreiment results are in
```sh
freeciv-tensor-baseline/examples/results 
```


### Single Run
Edit `freeciv-tensor-baseline/civtensor/configs/envs_cfgs/freeciv_tensor_env.yaml`

Change `task_name` to `random_minitask`  or any task name in the list of MinitaskType.list()

    # copied from freeciv_minitask_env.py
    MT_DEVELOPMENT_BUILD_CITY = "development_build_city"
    MT_DEVELOPMENT_CITYTILE_WONDER = "development_citytile_wonder"
    MT_BATTLE_ANCIENT = "battle_ancient_era"
    MT_BATTLE_INDUSTRY = "battle_industry_era"
    MT_BATTLE_INFO = "battle_info_era"
    MT_BATTLE_MEDIEVAL = "battle_medieval"
    MT_BATTLE_MODERN = "battle_modern_era"
    MT_BATTLE_NAVAL_MODERN = "battle_naval_modern"
    MT_BATTLE_NAVAL = "battle_naval"
    MT_BATTLE_ATTACK_CITY = "battle_attack_city"
    MT_BATTLE_DEFEND_CITY = "battle_defend_city"
    MT_DIPLOMACY_TRADE_TECH = "diplomacy_trade_tech"

```yaml
task_name: development_build_city # or random_minitask to randomly sample a minitask
```
You may optionally specify difficulty level and ids for minitask.
```yaml
task_name: development_build_city easy 100 # type level id 
# unspecified fields would be randomly sampled 
```


# Customize
Training parameters could be adjusted in `freeciv-tensor-baseline/civtensor/configs/algos_cfgs/ppo.yaml`

For training minitask tensor baseline, we used the following default setting:

```yaml
seed:
  # whether to use the specified seed
  seed_specify: False
  # seed
  seed: None
train:
  # number of parallel environments for training data collection
  n_rollout_threads: 8
  # number of total training steps
  num_env_steps: 20000
  # number of steps per environment per training data collection
  episode_length: 125
  # logging interval
  log_interval: 1
```
