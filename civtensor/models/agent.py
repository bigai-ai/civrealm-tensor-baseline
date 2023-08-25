import torch
import torch.nn as nn
from civtensor.models.encoder.transformer_encoder import TransformerEncoder


class Agent(nn.Module):
    """
    Freeciv Tensor Baseline Agent.
    It outputs actions, action_log_probs, and values given states and available_actions.
    """

    def __init__(self, args, state_space, action_space, device=torch.device("cpu")):
        super(Agent, self).__init__()
        self.args = args
        self.state_space = state_space
        self.action_space = action_space
        self.device = device

        self.init_network()

    def init_network(self):
        # obtain input dimensions. TODO: list the numbers.
        self.rules_dim = self.state_space["rules"].shape[0]
        self.player_dim = self.state_space["player"].shape[0]
        self.other_players_dim = self.state_space["other_players"].shape[
            0
        ]  # or Sequence?
        self.units_dim = self.state_space["units"].shape[0]  # or Sequence?
        self.cities_dim = self.state_space["cities"].shape[0]  # or Sequence?
        self.other_units_dim = self.state_space["other_units"].shape[0]  # or Sequence?
        self.other_cities_dim = self.state_space["other_cities"].shape[
            0
        ]  # or Sequence?

        # obtain hidden dimensions
        self.hidden_dim = 256
        self.map_hidden_dim = 512
        self.lstm_hidden_dim = 1024
        self.n_head = 2
        self.n_layers = 2
        self.drop_prob = 0

        # initialize encoders
        self.rules_encoder = nn.Linear(self.rules_dim, self.hidden_dim)

        self.player_encoder = nn.Linear(self.player_dim, self.hidden_dim)

        self.other_players_embedding = nn.Linear(
            self.other_players_dim, self.hidden_dim
        )
        self.other_players_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.units_embedding = nn.Linear(self.units_dim, self.hidden_dim)
        self.units_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.cities_embedding = nn.Linear(self.cities_dim, self.hidden_dim)
        self.cities_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.other_units_embedding = nn.Linear(self.other_units_dim, self.hidden_dim)
        self.other_units_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.other_cities_embedding = nn.Linear(self.other_cities_dim, self.hidden_dim)
        self.other_cities_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.map_encoder = None

    def forward(self, states, masks, available_actions):
        # obtain states
        rules = states["rules"]  # (batch_size, rules_dim)
        player = states["player"]  # (batch_size, player_dim)
        other_players = states[
            "other_players"
        ]  # (batch_size, n_max_other_players, other_players_dim)
        units = states["units"]  # (batch_size, n_max_units, units_dim)
        cities = states["cities"]  # (batch_size, n_max_cities, cities_dim)
        other_units = states[
            "other_units"
        ]  # (batch_size, n_max_other_units, other_units_dim)
        other_cities = states[
            "other_cities"
        ]  # (batch_size, n_max_other_cities, other_cities_dim)

        # obtain masks
        other_players_mask = masks[
            "other_players"
        ]  # (batch_size, n_max_other_players, 1)
        units_mask = masks["units"]  # (batch_size, n_max_units, 1)
        cities_mask = masks["cities"]  # (batch_size, n_max_cities, 1)
        other_units_mask = masks["other_units"]  # (batch_size, n_max_other_units, 1)
        other_cities_mask = masks["other_cities"]  # (batch_size, n_max_other_cities, 1)

        # encoding step
        rules_encoded = self.rules_encoder(rules)  # (batch_size, hidden_dim)

        player_encoded = self.player_encoder(player)  # (batch_size, hidden_dim)

        batch_size, n_max_other_players, other_players_dim = other_players.shape
        other_players_embedding = self.other_players_embedding(
            other_players.view(-1, self.other_players_dim)
        ).view(
            batch_size, n_max_other_players, other_players_dim
        )  # (batch_size, n_max_other_players, hidden_dim)
        other_players_encoded = self.other_players_encoder(
            other_players_embedding, other_players_mask
        )  # (batch_size, n_max_other_players, hidden_dim)

        batch_size, n_max_units, units_dim = units.shape
        units_embedding = self.units_embedding(units.view(-1, self.units_dim)).view(
            batch_size, n_max_units, units_dim
        )  # (batch_size, n_max_units, hidden_dim)
        units_encoded = self.units_encoder(
            units_embedding, units_mask
        )  # (batch_size, n_max_units, hidden_dim)

        batch_size, n_max_cities, cities_dim = cities.shape
        cities_embedding = self.cities_embedding(cities.view(-1, self.cities_dim)).view(
            batch_size, n_max_cities, cities_dim
        )  # (batch_size, n_max_cities, hidden_dim)
        cities_encoded = self.cities_encoder(
            cities_embedding, cities_mask
        )  # (batch_size, n_max_cities, hidden_dim)

        batch_size, n_max_other_units, other_units_dim = other_units.shape
        other_units_embedding = self.other_units_embedding(
            other_units.view(-1, self.other_units_dim)
        ).view(
            batch_size, n_max_other_units, other_units_dim
        )  # (batch_size, n_max_other_units, hidden_dim)
        other_units_encoded = self.other_units_encoder(
            other_units_embedding, other_units_mask
        )  # (batch_size, n_max_other_units, hidden_dim)

        batch_size, n_max_other_cities, other_cities_dim = other_cities.shape
        other_cities_embedding = self.other_cities_embedding(
            other_cities.view(-1, self.other_cities_dim)
        ).view(
            batch_size, n_max_other_cities, other_cities_dim
        )  # (batch_size, n_max_other_cities, hidden_dim)
        other_cities_encoded = self.other_cities_encoder(
            other_cities_embedding, other_cities_mask
        )  # (batch_size, n_max_other_cities, hidden_dim)

        # lstm step

        # action step
