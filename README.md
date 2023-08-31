# Freeciv Tensor Baseline



## Interface

### Spaces

```python
state_spaces = {
    "rules": Box(low=-inf, high=inf, shape=(rules_dim, ), dtype=np.float32),
    "player": Box(low=-inf, high=inf, shape=(player_dim, ), dtype=np.float32),
    "other_players": Box(low=-inf, high=inf, shape=(n_max_other_players, other_players_dim), dtype=np.float32),
    "units": Box(low=-inf, high=inf, shape=(n_max_units, units_dim), dtype=np.float32),
    "cities": Box(low=-inf, high=inf, shape=(n_max_cities, cities_dim), dtype=np.float32),
    "other_units": Box(low=-inf, high=inf, shape=(n_max_other_units, other_units_dim), dtype=np.float32),
    "other_cities": Box(low=-inf, high=inf, shape=(n_max_other_cities, other_cities_dim), dtype=np.float32),
    "map": Box(low=-inf, high=inf, shape=(x_size, y_size, map_channels), dtype=np.float32),
    "other_players_mask": Box(low=-inf, high=inf, shape=(n_max_other_players, 1), dtype=np.int32), 
    "units_mask": Box(low=-inf, high=inf, shape=(n_max_units, 1), dtype=np.int32), 
    "cities_mask": Box(low=-inf, high=inf, shape=(n_max_cities, 1), dtype=np.int32), 
    "other_units_mask": Box(low=-inf, high=inf, shape=(n_max_other_units, 1), dtype=np.int32), 
    "other_cities_mask": Box(low=-inf, high=inf, shape=(n_max_other_cities, 1), dtype=np.int32), 
    "actor_type_mask": Box(low=-inf, high=inf, shape=(actor_type_dim, ), dtype=np.int32), 
    "city_id_mask": Box(low=-inf, high=inf, shape=(n_max_cities, 1), dtype=np.int32), 
    "city_action_type_mask": Box(low=-inf, high=inf, shape=(n_max_cities, city_action_type_dim), dtype=np.int32), 
    "unit_id_mask": Box(low=-inf, high=inf, shape=(n_max_units, 1), dtype=np.int32), 
    "unit_action_type_mask": Box(low=-inf, high=inf, shape=(n_max_units, unit_action_type_dim), dtype=np.int32), 
    "gov_action_type_mask": Box(low=-inf, high=inf, shape=(gov_action_type_dim, ), dtype=np.int32), 
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
    rules,  # (batch_size, rules_dim)
    player,  # (batch_size, player_dim)
    other_players,  # (batch_size, n_max_other_players, other_players_dim)
    units,  # (batch_size, n_max_units, units_dim)
    cities,  # (batch_size, n_max_cities, cities_dim)
    other_units,  # (batch_size, n_max_other_units, other_units_dim)
    other_cities,  # (batch_size, n_max_other_cities, other_cities_dim)
    map,  # (batch_size, x_size, y_size, map_input_channels) TODO check input order
    other_players_mask,  # (batch_size, n_max_other_players, 1) Note: masks are 0 for padding, 1 for non-padding
    units_mask,  # (batch_size, n_max_units, 1)
    cities_mask,  # (batch_size, n_max_cities, 1)
    other_units_mask,  # (batch_size, n_max_other_units, 1)
    other_cities_mask,  # (batch_size, n_max_other_cities, 1)
    actor_type_mask,  # (batch_size, actor_type_dim)
    city_id_mask,  # (batch_size, n_max_cities, 1)
    city_action_type_mask,  # (batch_size, n_max_cities, city_action_type_dim)
    unit_id_mask,  # (batch_size, n_max_units, 1)
    unit_action_type_mask,  # (batch_size, n_max_units, unit_action_type_dim)
    gov_action_type_mask,  # (batch_size, gov_action_type_dim)
    reward,  # (batch_size, 1)
    term,  # (batch_size, 1)
    trunc,  # (batch_size, 1)
) = self.envs.step(
    actor_type,  # (batch_size, 1)
    city_id,  # (batch_size, 1)
    city_action_type,  # (batch_size, 1)
    unit_id,  # (batch_size, 1)
    unit_action_type,  # (batch_size, 1)
    gov_action_type,  # (batch_size, 1)
)
```