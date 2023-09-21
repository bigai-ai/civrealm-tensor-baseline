# Freeciv Tensor Baseline

## Quick Start
First install
```
git clone ssh://git@gitlab.mybigai.ac.cn:2222/civilization/freeciv-tensor-baseline.git
cd freeciv-tensor-baseline
pip install -e .
```

then run the training script
```sh
cd examples
./train.sh # or python train.py
```
## Fullgame
Edit `civtensor/configs/envs_cfgs/freeciv_tensor_env.yaml`

Change `task_name` to `fullgame`

```yaml
task_name: fullgame
```

## Minigame

Edit `civtensor/configs/envs_cfgs/freeciv_tensor_env.yaml`

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
Training parameters could be adjusted in `civtensor/configs/algos_cfgs/ppo.yaml`

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


# Notes

- Remember to set `max_turns` to `100` in freeciv-gym for early phase only training.
- On 3090Ti GPU, I think `n_rollout_threads: 5` and `episode_length: 200` will reach the maximum of its CUDA memory.




## Interface

### Spaces

```python
state_spaces = {
    "rules": Box(low=-inf, high=inf, shape=(rules_dim,), dtype=np.float32),
    "player": Box(low=-inf, high=inf, shape=(player_dim,), dtype=np.float32),
    "other_players": Box(low=-inf, high=inf, shape=(n_max_other_players, other_players_dim), dtype=np.float32),
    "units": Box(low=-inf, high=inf, shape=(n_max_units, units_dim), dtype=np.float32),
    "cities": Box(low=-inf, high=inf, shape=(n_max_cities, cities_dim), dtype=np.float32),
    "other_units": Box(low=-inf, high=inf, shape=(n_max_other_units, other_units_dim), dtype=np.float32),
    "other_cities": Box(low=-inf, high=inf, shape=(n_max_other_cities, other_cities_dim), dtype=np.float32),
    "civmap": Box(low=-inf, high=inf, shape=(x_size, y_size, civmap_channels), dtype=np.float32),
    "other_players_mask": MultiBinary((n_max_other_players, 1)), 
    "units_mask": MultiBinary((n_max_units, 1)), 
    "cities_mask": MultiBinary((n_max_cities, 1)), 
    "other_units_mask": MultiBinary((n_max_other_units, 1)), 
    "other_cities_mask": MultiBinary((n_max_other_cities, 1)), 
    "actor_type_mask": MultiBinary((actor_type_dim,)), 
    "city_id_mask": MultiBinary((n_max_cities, 1)), 
    "city_action_type_mask": MultiBinary((n_max_cities, city_action_type_dim)), 
    "unit_id_mask": MultiBinary((n_max_units, 1)), 
    "unit_action_type_mask": MultiBinary((n_max_units, unit_action_type_dim)), 
    "gov_action_type_mask": MultiBinary((gov_action_type_dim,)), 
}

action_spaces = {
    "actor_type": Discrete(actor_type_dim), # actor_type_dim = 4; 0 for city, 1 for unit, 2 for gov, 3 for turn done
    "city_id": Discrete(n_max_cities),
    "city_action_type": Discrete(city_action_type_dim),
    "unit_id": Discrete(n_max_units),
    "unit_action_type": Discrete(unit_action_type_dim),
    "gov_action_type": Discrete(gov_action_type_dim),
}
```


### Interaction

```python
(
    rules,  # (n_parallel_envs, rules_dim)
    player,  # (n_parallel_envs, player_dim)
    other_players,  # (n_parallel_envs, n_max_other_players, other_players_dim)
    units,  # (n_parallel_envs, n_max_units, units_dim)
    cities,  # (n_parallel_envs, n_max_cities, cities_dim)
    other_units,  # (n_parallel_envs, n_max_other_units, other_units_dim)
    other_cities,  # (n_parallel_envs, n_max_other_cities, other_cities_dim)
    civmap,  # (n_parallel_envs, x_size, y_size, civmap_channels) TODO check input order
    other_players_mask,  # (n_parallel_envs, n_max_other_players, 1) Note: masks are 0 for padding, 1 for non-padding
    units_mask,  # (n_parallel_envs, n_max_units, 1)
    cities_mask,  # (n_parallel_envs, n_max_cities, 1)
    other_units_mask,  # (n_parallel_envs, n_max_other_units, 1)
    other_cities_mask,  # (n_parallel_envs, n_max_other_cities, 1)
    actor_type_mask,  # (n_parallel_envs, actor_type_dim)
    city_id_mask,  # (n_parallel_envs, n_max_cities, 1)
    city_action_type_mask,  # (n_parallel_envs, n_max_cities, city_action_type_dim)
    unit_id_mask,  # (n_parallel_envs, n_max_units, 1)
    unit_action_type_mask,  # (n_parallel_envs, n_max_units, unit_action_type_dim)
    gov_action_type_mask,  # (n_parallel_envs, gov_action_type_dim)
    reward,  # (n_parallel_envs, 1)
    term,  # (n_parallel_envs, 1)
    trunc,  # (n_parallel_envs, 1)
) = self.envs.step(
    actor_type,  # (n_parallel_envs, 1)
    city_id,  # (n_parallel_envs, 1)
    city_action_type,  # (n_parallel_envs, 1)
    unit_id,  # (n_parallel_envs, 1)
    unit_action_type,  # (n_parallel_envs, 1)
    gov_action_type,  # (n_parallel_envs, 1)
)
```

### Tensor Env

observation space keys = ['rules', 'map', 'player', 'city', 'unit', 'others_player', 'others_unit', 'others_city', 'unit_mask', 'city_mask', 'others_unit_mask', 'others_city_mask', 'others_player_mask', 'actor_type_mask', 'city_id_mask', 'city_action_type_mask', 'unit_id_mask', 'unit_action_type_mask', 'gov_action_type_mask']

# Acknowledgement

This tensor baseline is developed based on https://github.com/PKU-MARL/HARL, please feel free to give it a star!
