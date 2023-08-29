import torch
import torch.nn as nn
from torch.distributions import Categorical

from civtensor.models.encoder.transformer_encoder import TransformerEncoder
from civtensor.models.pointer_network.pointer_network import PointerNetwork


class Agent(nn.Module):
    """
    Freeciv Tensor Baseline Agent.
    It outputs actions, action_log_probs, and values given states and available_actions.
    """

    def __init__(self, args, state_spaces, action_spaces, device=torch.device("cpu")):
        super(Agent, self).__init__()
        self.args = args
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.device = device

        self.init_network(args)

    def init_network(self, args):
        # obtain input dimensions. TODO: be consistent with env
        self.rules_dim = self.state_spaces["rules"].shape[1]
        self.player_dim = self.state_spaces["player"].shape[1]
        self.other_players_dim = self.state_spaces["other_players"].shape[
            1
        ]  # or Sequence?
        self.units_dim = self.state_spaces["units"].shape[1]  # or Sequence?
        self.cities_dim = self.state_spaces["cities"].shape[1]  # or Sequence?
        self.other_units_dim = self.state_spaces["other_units"].shape[1]  # or Sequence?
        self.other_cities_dim = self.state_spaces["other_cities"].shape[
            1
        ]  # or Sequence?
        self.map_dim = self.state_spaces["map"].shape

        # obtain output dimensions. TODO: be consistent with env
        self.actor_type_dim = self.action_spaces["actor_type"].n
        self.city_action_type_dim = self.action_spaces["city_action_type"].n
        self.unit_action_type_dim = self.action_spaces["unit_action_type"].n
        self.gov_action_type_dim = self.action_spaces["gov_action_type"].n

        # obtain hidden dimensions
        self.hidden_dim = args["hidden_dim"]  # 256
        self.lstm_hidden_dim = args["lstm_hidden_dim"]  # 1024
        self.n_head = args["n_head"]  # 2
        self.n_layers = args["n_layers"]  # 2
        self.drop_prob = args["drop_prob"]  # 0
        self.n_lstm_layers = args["n_lstm_layers"]  # 2

        # initialize encoders
        self.rules_encoder = nn.Sequential(
            nn.Linear(self.rules_dim, self.hidden_dim), nn.ReLU()
        )

        self.player_encoder = nn.Sequential(
            nn.Linear(self.player_dim, self.hidden_dim), nn.ReLU()
        )

        self.other_players_embedding = nn.Sequential(
            nn.Linear(self.other_players_dim, self.hidden_dim), nn.ReLU()
        )

        self.other_players_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.units_embedding = nn.Sequential(
            nn.Linear(self.units_dim, self.hidden_dim), nn.ReLU()
        )

        self.units_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.cities_embedding = nn.Sequential(
            nn.Linear(self.cities_dim, self.hidden_dim), nn.ReLU()
        )

        self.cities_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.other_units_embedding = nn.Sequential(
            nn.Linear(self.other_units_dim, self.hidden_dim), nn.ReLU()
        )

        self.other_units_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.other_cities_embedding = nn.Sequential(
            nn.Linear(self.other_cities_dim, self.hidden_dim), nn.ReLU()
        )

        self.other_cities_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.map_encoder = nn.Sequential(
            nn.Conv2d(112, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 3 * 32, 256),
            nn.ReLU(),
        )

        # initialize global transformer
        self.global_transformer = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        # initialize lstm.
        self.lstm = nn.LSTM(
            8 * self.hidden_dim, self.lstm_hidden_dim, self.n_lstm_layers
        )

        # initialize value head
        self.value_linear = nn.Linear(self.lstm_hidden_dim, 1)

        # initialize actor heads
        self.actor_type_linear = nn.Linear(self.lstm_hidden_dim, self.actor_type_dim)
        self.actor_type_softmax = nn.Softmax(dim=-1)

        self.city_id_head = PointerNetwork(self.hidden_dim, self.lstm_hidden_dim)

        self.unit_id_head = PointerNetwork(self.hidden_dim, self.lstm_hidden_dim)

        self.city_action_linear = nn.Linear(
            self.lstm_hidden_dim + self.hidden_dim, self.city_action_type_dim
        )
        self.city_action_softmax = nn.Softmax(dim=-1)

        self.unit_action_linear = nn.Linear(
            self.lstm_hidden_dim + self.hidden_dim, self.unit_action_type_dim
        )
        self.unit_action_softmax = nn.Softmax(dim=-1)

        self.gov_action_linear = nn.Linear(
            self.lstm_hidden_dim, self.gov_action_type_dim
        )
        self.gov_action_softmax = nn.Softmax(dim=-1)

    def forward(self, states, lstm_hidden_state, masks):
        """
        Args:
            lstm_hidden_state: (h, c) where h and c are (n_lstm_layers, batch_size, lstm_hidden_dim)
        """
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
        map = states[
            "map"
        ]  # (batch_size, x_size, y_size, map_input_channels) TODO check input order
        map = map.permute(
            0, 3, 1, 2
        )  # (batch_size, map_input_channels, x_size, y_size)

        # obtain masks
        # Note: masks are 0 for padding, 1 for non-padding
        other_players_mask = masks[
            "other_players"
        ]  # (batch_size, n_max_other_players, 1)
        units_mask = masks["units"]  # (batch_size, n_max_units, 1)
        cities_mask = masks["cities"]  # (batch_size, n_max_cities, 1)
        other_units_mask = masks["other_units"]  # (batch_size, n_max_other_units, 1)
        other_cities_mask = masks["other_cities"]  # (batch_size, n_max_other_cities, 1)
        actor_type_mask = masks["actor_type"]  # (batch_size, actor_type_dim)
        city_id_mask = masks["city_id"]  # (batch_size, n_max_cities, 1)
        unit_id_mask = masks["unit_id"]  # (batch_size, n_max_units, 1)
        city_action_type_mask = masks[
            "city_action_type"
        ]  # (batch_size, city_action_type_dim)
        unit_action_type_mask = masks[
            "unit_action_type"
        ]  # (batch_size, unit_action_type_dim)
        gov_action_type_mask = masks[
            "gov_action_type"
        ]  # (batch_size, gov_action_type_dim)

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

        map_encoded = self.map_encoder(map)  # (batch_size, hidden_dim)

        # global transformer step
        other_players_global_encoding = torch.sum(
            other_players_encoded * other_players_mask, dim=1
        ) / torch.sum(
            other_players_mask, dim=1
        )  # (batch_size, hidden_dim)

        units_global_encoding = torch.sum(
            units_encoded * units_mask, dim=1
        ) / torch.sum(
            units_mask, dim=1
        )  # (batch_size, hidden_dim)

        cities_global_encoding = torch.sum(
            cities_encoded * cities_mask, dim=1
        ) / torch.sum(cities_mask, dim=1)

        other_units_global_encoding = torch.sum(
            other_units_encoded * other_units_mask, dim=1
        ) / torch.sum(other_units_mask, dim=1)

        other_cities_global_encoding = torch.sum(
            other_cities_encoded * other_cities_mask, dim=1
        ) / torch.sum(other_cities_mask, dim=1)

        global_encoding = torch.stack(
            [
                rules_encoded,
                player_encoded,
                other_players_global_encoding,
                units_global_encoding,
                cities_global_encoding,
                other_units_global_encoding,
                other_cities_global_encoding,
                map_encoded,
            ],
            dim=1,
        )  # (batch_size, 8, hidden_dim)

        global_encoding_processed = self.global_transformer(
            global_encoding, src_mask=None
        )  # (batch_size, 8, hidden_dim)

        # lstm step
        global_encoding_concat = global_encoding_processed.view(
            batch_size, -1
        )  # (batch_size, 8 * hidden_dim)

        # TODO: add training logic; we may need to extract this into a separate class
        # lstm_hidden_state: (h, c) where h and c are (n_lstm_layers, batch_size, lstm_hidden_dim)
        lstm_out, lstm_hidden_state = self.lstm(
            global_encoding_concat.unsqueeze(0), lstm_hidden_state
        )  # (1, batch_size, lstm_hidden_dim), ((n_lstm_layers, batch_size, lstm_hidden_dim), (n_lstm_layers, batch_size, lstm_hidden_dim))
        lstm_out = lstm_out.squeeze(0)  # (batch_size, lstm_hidden_dim)

        # output value predictions
        value_predictions = self.value_linear(lstm_out)  # (batch_size, 1)

        # action step
        # actor type head
        actor_type_logits = self.actor_type_linear(
            lstm_out
        )  # (batch_size, actor_type_dim)
        actor_type_logits = actor_type_logits.masked_fill(actor_type_mask == 0, -10000)
        actor_type = self.actor_type_softmax(
            actor_type_logits
        )  # (batch_size, actor_type_dim)
        actor_distribution = Categorical(actor_type)
        actor_type_id = actor_distribution.sample()  # (batch_size, 1)

        # city id head
        city_id, city_chosen_encoded = self.city_id_head(
            lstm_out, cities_encoded, city_id_mask
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # city action type head
        city_action_input = torch.cat([lstm_out, city_chosen_encoded], dim=-1)
        city_action_logits = self.city_action_linear(city_action_input)
        city_action_logits = city_action_logits.masked_fill(
            city_action_type_mask == 0, -10000
        )
        city_action_type = self.city_action_softmax(city_action_logits)
        city_action_distribution = Categorical(city_action_type)
        city_action_type_id = city_action_distribution.sample()  # (batch_size, 1)

        # unit id head
        unit_id, unit_chosen_encoded = self.unit_id_head(
            lstm_out, units_encoded, unit_id_mask
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # unit action type head
        unit_action_input = torch.cat([lstm_out, unit_chosen_encoded], dim=-1)
        unit_action_logits = self.unit_action_linear(unit_action_input)
        unit_action_logits = unit_action_logits.masked_fill(
            unit_action_type_mask == 0, -10000
        )
        unit_action_type = self.unit_action_softmax(unit_action_logits)
        unit_action_distribution = Categorical(unit_action_type)
        unit_action_type_id = unit_action_distribution.sample()  # (batch_size, 1)

        # gov action type head
        gov_action_logits = self.gov_action_linear(lstm_out)
        gov_action_logits = gov_action_logits.masked_fill(
            gov_action_type_mask == 0, -10000
        )
        gov_action_type = self.gov_action_softmax(gov_action_logits)
        gov_action_distribution = Categorical(gov_action_type)
        gov_action_type_id = gov_action_distribution.sample()  # (batch_size, 1)

        return (
            actor_type_id,
            city_id,
            city_action_type_id,
            unit_id,
            unit_action_type_id,
            gov_action_type_id,
            value_predictions,
            lstm_hidden_state,
        )
