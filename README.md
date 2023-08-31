# Freeciv Tensor Baseline



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
    "other_players_mask": Box(low=-inf, high=inf, shape=(n_max_other_players, 1), dtype=np.int32), 
    "units_mask": Box(low=-inf, high=inf, shape=(n_max_units, 1), dtype=np.int32), 
    "cities_mask": Box(low=-inf, high=inf, shape=(n_max_cities, 1), dtype=np.int32), 
    "other_units_mask": Box(low=-inf, high=inf, shape=(n_max_other_units, 1), dtype=np.int32), 
    "other_cities_mask": Box(low=-inf, high=inf, shape=(n_max_other_cities, 1), dtype=np.int32), 
    "actor_type_mask": Box(low=-inf, high=inf, shape=(actor_type_dim,), dtype=np.int32), 
    "city_id_mask": Box(low=-inf, high=inf, shape=(n_max_cities, 1), dtype=np.int32), 
    "city_action_type_mask": Box(low=-inf, high=inf, shape=(n_max_cities, city_action_type_dim), dtype=np.int32), 
    "unit_id_mask": Box(low=-inf, high=inf, shape=(n_max_units, 1), dtype=np.int32), 
    "unit_action_type_mask": Box(low=-inf, high=inf, shape=(n_max_units, unit_action_type_dim), dtype=np.int32), 
    "gov_action_type_mask": Box(low=-inf, high=inf, shape=(gov_action_type_dim,), dtype=np.int32), 
}

action_spaces = {
    "actor_type": Discrete(actor_type_dim),
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