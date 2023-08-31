import torch
import torch.nn as nn

from civtensor.models.base.distributions import Categorical
from civtensor.models.base.rnn import RNNLayer
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

        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.init_network(args)

        self.to(device)

    def init_network(self, args):
        # obtain input dimensions. TODO: be consistent with env
        self.rules_dim = self.state_spaces["rules"].shape[0]
        self.player_dim = self.state_spaces["player"].shape[0]
        self.other_players_dim = self.state_spaces["other_players"].shape[
            1
        ]  # or Sequence?
        self.units_dim = self.state_spaces["units"].shape[1]  # or Sequence?
        self.cities_dim = self.state_spaces["cities"].shape[1]  # or Sequence?
        self.other_units_dim = self.state_spaces["other_units"].shape[1]  # or Sequence?
        self.other_cities_dim = self.state_spaces["other_cities"].shape[
            1
        ]  # or Sequence?
        self.civmap_dim = self.state_spaces["civmap"].shape

        # obtain output dimensions. TODO: be consistent with env
        self.actor_type_dim = self.action_spaces["actor_type"].n
        self.city_action_type_dim = self.action_spaces["city_action_type"].n
        self.unit_action_type_dim = self.action_spaces["unit_action_type"].n
        self.gov_action_type_dim = self.action_spaces["gov_action_type"].n

        # obtain hidden dimensions
        self.hidden_dim = args["hidden_dim"]  # 256
        self.rnn_hidden_dim = args["rnn_hidden_dim"]  # 1024
        self.n_head = args["n_head"]  # 2
        self.n_layers = args["n_layers"]  # 2
        self.drop_prob = args["drop_prob"]  # 0
        self.n_rnn_layers = args["n_rnn_layers"]  # 2

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

        self.civmap_encoder = nn.Sequential(
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

        # initialize rnn.
        self.rnn = RNNLayer(
            8 * self.hidden_dim,
            self.rnn_hidden_dim,
            self.n_rnn_layers,
            self.initialization_method,
        )

        # initialize value head
        self.value_linear = nn.Linear(self.rnn_hidden_dim, 1)

        # initialize actor heads
        self.actor_type_head = Categorical(
            self.rnn_hidden_dim,
            self.actor_type_dim,
            self.initialization_method,
            self.gain,
        )

        self.city_id_head = PointerNetwork(self.hidden_dim, self.rnn_hidden_dim)

        self.unit_id_head = PointerNetwork(self.hidden_dim, self.rnn_hidden_dim)

        self.city_action_head = Categorical(
            self.rnn_hidden_dim + self.hidden_dim,
            self.city_action_type_dim,
            self.initialization_method,
            self.gain,
        )

        self.unit_action_head = Categorical(
            self.rnn_hidden_dim + self.hidden_dim,
            self.unit_action_type_dim,
            self.initialization_method,
            self.gain,
        )

        self.gov_action_head = Categorical(
            self.rnn_hidden_dim,
            self.gov_action_type_dim,
            self.initialization_method,
            self.gain,
        )

    def encoding_step(
        self,
        rules,
        player,
        other_players,
        units,
        cities,
        other_units,
        other_cities,
        civmap,
        other_players_mask,
        units_mask,
        cities_mask,
        other_units_mask,
        other_cities_mask,
    ):
        """
        Args:
            rules: (batch_size, rules_dim)
            player: (batch_size, player_dim)
            other_players: (batch_size, n_max_other_players, other_players_dim)
            units: (batch_size, n_max_units, units_dim)
            cities: (batch_size, n_max_cities, cities_dim)
            other_units: (batch_size, n_max_other_units, other_units_dim)
            other_cities: (batch_size, n_max_other_cities, other_cities_dim)
            civmap: (batch_size, x_size, y_size, civmap_channels) TODO check input order
            other_players_mask: (batch_size, n_max_other_players, 1) Note: masks are 0 for padding, 1 for non-padding
            units_mask: (batch_size, n_max_units, 1)
            cities_mask: (batch_size, n_max_cities, 1)
            other_units_mask: (batch_size, n_max_other_units, 1)
            other_cities_mask: (batch_size, n_max_other_cities, 1)
        """
        civmap = civmap.permute(
            0, 3, 1, 2
        )  # (batch_size, civmap_channels, x_size, y_size)

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

        civmap_encoded = self.civmap_encoder(civmap)  # (batch_size, hidden_dim)

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
                civmap_encoded,
            ],
            dim=1,
        )  # (batch_size, 8, hidden_dim)

        global_encoding_processed = self.global_transformer(
            global_encoding, src_mask=None
        )  # (batch_size, 8, hidden_dim)

        return global_encoding_processed, units_encoded, cities_encoded

    def forward(
        self,
        rules,
        player,
        other_players,
        units,
        cities,
        other_units,
        other_cities,
        civmap,
        other_players_mask,
        units_mask,
        cities_mask,
        other_units_mask,
        other_cities_mask,
        actor_type_mask,
        city_id_mask,
        city_action_type_mask,
        unit_id_mask,
        unit_action_type_mask,
        gov_action_type_mask,
        rnn_hidden_state,
        mask,  # TODO check whether logic related to mask is correct
        deterministic,
    ):
        """
        Args:
            rules: (batch_size, rules_dim)
            player: (batch_size, player_dim)
            other_players: (batch_size, n_max_other_players, other_players_dim)
            units: (batch_size, n_max_units, units_dim)
            cities: (batch_size, n_max_cities, cities_dim)
            other_units: (batch_size, n_max_other_units, other_units_dim)
            other_cities: (batch_size, n_max_other_cities, other_cities_dim)
            civmap: (batch_size, x_size, y_size, civmap_channels) TODO check input order
            other_players_mask: (batch_size, n_max_other_players, 1) Note: masks are 0 for padding, 1 for non-padding
            units_mask: (batch_size, n_max_units, 1)
            cities_mask: (batch_size, n_max_cities, 1)
            other_units_mask: (batch_size, n_max_other_units, 1)
            other_cities_mask: (batch_size, n_max_other_cities, 1)
            actor_type_mask: (batch_size, actor_type_dim)
            city_id_mask: (batch_size, n_max_cities, 1)
            city_action_type_mask: (batch_size, n_max_cities, city_action_type_dim)
            unit_id_mask: (batch_size, n_max_units, 1)
            unit_action_type_mask: (batch_size, n_max_units, unit_action_type_dim)
            gov_action_type_mask: (batch_size, gov_action_type_dim)
            rnn_hidden_state: (h, c) where h and c are (n_rnn_layers, batch_size, rnn_hidden_dim)
            mask: (batch_size, 1)
            deterministic: if True use argmax, else sample from distribution
        """

        # encoding step
        global_encoding_processed, units_encoded, cities_encoded = self.encoding_step(
            rules,
            player,
            other_players,
            units,
            cities,
            other_units,
            other_cities,
            civmap,
            other_players_mask,
            units_mask,
            cities_mask,
            other_units_mask,
            other_cities_mask,
        )

        batch_size = rules.shape[0]

        # rnn step
        global_encoding_concat = global_encoding_processed.view(
            batch_size, -1
        )  # (batch_size, 8 * hidden_dim)

        rnn_out, rnn_hidden_state = self.rnn(
            global_encoding_concat, rnn_hidden_state, mask
        )  # TODO: check is there problems with gradient due to same name used by rnn_hidden_state

        # output value predictions
        value_pred = self.value_linear(rnn_out)  # (batch_size, 1)

        # action step
        # actor type head
        actor_type_distribution = self.actor_type_head(rnn_out, actor_type_mask)
        actor_type = (
            actor_type_distribution.mode()
            if deterministic
            else actor_type_distribution.sample()
        )
        actor_type_log_prob = actor_type_distribution.log_probs(actor_type)

        # city id head
        city_id, city_id_log_prob, city_chosen_encoded = self.city_id_head(
            rnn_out, cities_encoded, city_id_mask, deterministic
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # city action type head
        city_action_input = torch.cat([rnn_out, city_chosen_encoded], dim=-1)
        chosen_city_action_type_mask = city_action_type_mask[
            torch.arange(batch_size), city_id.squeeze(), :
        ]  # TODO: check whether gradient is correct
        city_action_type_distribution = self.city_action_head(
            city_action_input, chosen_city_action_type_mask
        )
        city_action_type = (
            city_action_type_distribution.mode()
            if deterministic
            else city_action_type_distribution.sample()
        )
        city_action_type_log_prob = city_action_type_distribution.log_probs(
            city_action_type
        )

        # unit id head
        unit_id, unit_id_log_prob, unit_chosen_encoded = self.unit_id_head(
            rnn_out, units_encoded, unit_id_mask, deterministic
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # unit action type head
        unit_action_input = torch.cat([rnn_out, unit_chosen_encoded], dim=-1)
        chosen_unit_action_type_mask = unit_action_type_mask[
            torch.arange(batch_size), unit_id.squeeze(), :
        ]  # TODO: check whether gradient is correct
        unit_action_type_distribution = self.unit_action_head(
            unit_action_input, chosen_unit_action_type_mask
        )
        unit_action_type = (
            unit_action_type_distribution.mode()
            if deterministic
            else unit_action_type_distribution.sample()
        )
        unit_action_type_log_prob = unit_action_type_distribution.log_probs(
            unit_action_type
        )

        # gov action type head
        gov_action_type_distribution = self.gov_action_head(
            rnn_out, gov_action_type_mask
        )
        gov_action_type = (
            gov_action_type_distribution.mode()
            if deterministic
            else gov_action_type_distribution.sample()
        )
        gov_action_type_log_prob = gov_action_type_distribution.log_probs(
            gov_action_type
        )

        return (
            actor_type,
            actor_type_log_prob,
            city_id,
            city_id_log_prob,
            city_action_type,
            city_action_type_log_prob,
            unit_id,
            unit_id_log_prob,
            unit_action_type,
            unit_action_type_log_prob,
            gov_action_type,
            gov_action_type_log_prob,
            value_pred,
            rnn_hidden_state,
        )

    def evaluate_actions(
        self,
        rules_batch,  # (episode_length * num_envs_per_batch, rules_dim)
        player_batch,  # (episode_length * num_envs_per_batch, player_dim)
        other_players_batch,  # (episode_length * num_envs_per_batch, n_max_other_players, other_players_dim)
        units_batch,  # (episode_length * num_envs_per_batch, n_max_units, units_dim)
        cities_batch,  # (episode_length * num_envs_per_batch, n_max_cities, cities_dim)
        other_units_batch,  # (episode_length * num_envs_per_batch, n_max_other_units, other_units_dim)
        other_cities_batch,  # (episode_length * num_envs_per_batch, n_max_other_cities, other_cities_dim)
        civmap_batch,  # (episode_length * num_envs_per_batch, x_size, y_size, civmap_channels)
        other_players_masks_batch,  # (episode_length * num_envs_per_batch, n_max_other_players, 1)
        units_masks_batch,  # (episode_length * num_envs_per_batch, n_max_units, 1)
        cities_masks_batch,  # (episode_length * num_envs_per_batch, n_max_cities, 1)
        other_units_masks_batch,  # (episode_length * num_envs_per_batch, n_max_other_units, 1)
        other_cities_masks_batch,  # (episode_length * num_envs_per_batch, n_max_other_cities, 1)
        rnn_hidden_states_batch,  # (1 * num_envs_per_batch, n_rnn_layers, rnn_hidden_dim)
        actor_type_batch,  # (episode_length * num_envs_per_batch, 1)
        actor_type_masks_batch,  # (episode_length * num_envs_per_batch, actor_type_dim)
        city_id_batch,  # (episode_length * num_envs_per_batch, 1)
        city_id_masks_batch,  # (episode_length * num_envs_per_batch, n_max_cities, 1)
        city_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        city_action_type_masks_batch,  # (episode_length * num_envs_per_batch, city_action_type_dim)
        unit_id_batch,  # (episode_length * num_envs_per_batch, 1)
        unit_id_masks_batch,  # (episode_length * num_envs_per_batch, n_max_units, 1)
        unit_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        unit_action_type_masks_batch,  # (episode_length * num_envs_per_batch, unit_action_type_dim)
        gov_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        gov_action_type_masks_batch,  # (episode_length * num_envs_per_batch, gov_action_type_dim)
        masks_batch,  # (episode_length * num_envs_per_batch, 1)
    ):
        # encoding step
        global_encoding_processed, units_encoded, cities_encoded = self.encoding_step(
            rules_batch,
            player_batch,
            other_players_batch,
            units_batch,
            cities_batch,
            other_units_batch,
            other_cities_batch,
            civmap_batch,
            other_players_masks_batch,
            units_masks_batch,
            cities_masks_batch,
            other_units_masks_batch,
            other_cities_masks_batch,
        )

        batch_size = rules_batch.shape[0]

        # rnn step
        global_encoding_concat = global_encoding_processed.view(
            batch_size, -1
        )  # (batch_size, 8 * hidden_dim)

        rnn_out, rnn_hidden_states_batch = self.rnn(
            global_encoding_concat, rnn_hidden_states_batch, masks_batch
        )  # TODO: check is there problems with gradient due to same name used by rnn_hidden_state

        # output value predictions
        value_preds_batch = self.value_linear(rnn_out)  # (batch_size, 1)

        # action step
        # actor type head
        actor_type_distribution = self.actor_type_head(rnn_out, actor_type_masks_batch)
        actor_type_log_probs_batch = actor_type_distribution.log_probs(actor_type_batch)
        actor_type_dist_entropy = actor_type_distribution.entropy().mean()

        # city id head
        (
            city_id_log_probs_batch,
            city_chosen_encoded,
        ) = self.city_id_head.evaluate_actions(
            rnn_out, cities_encoded, city_id_masks_batch, city_id_batch
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # city action type head
        city_action_input = torch.cat([rnn_out, city_chosen_encoded], dim=-1)
        chosen_city_action_type_mask = city_action_type_masks_batch[
            torch.arange(batch_size), city_id_batch.squeeze(), :
        ]  # TODO: check whether gradient is correct
        city_action_type_distribution = self.city_action_head(
            city_action_input, chosen_city_action_type_mask
        )
        city_action_type_log_probs_batch = city_action_type_distribution.log_probs(
            city_action_type_batch
        )
        city_action_type_dist_entropy = city_action_type_distribution.entropy().mean()

        # unit id head
        unit_id_log_probs_batch, unit_chosen_encoded = self.unit_id_head(
            rnn_out, units_encoded, unit_id_masks_batch, unit_id_batch
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # unit action type head
        unit_action_input = torch.cat([rnn_out, unit_chosen_encoded], dim=-1)
        chosen_unit_action_type_mask = unit_action_type_masks_batch[
            torch.arange(batch_size), unit_id_batch.squeeze(), :
        ]  # TODO: check whether gradient is correct
        unit_action_type_distribution = self.unit_action_head(
            unit_action_input, chosen_unit_action_type_mask
        )
        unit_action_type_log_probs_batch = unit_action_type_distribution.log_probs(
            unit_action_type_batch
        )
        unit_action_type_dist_entropy = unit_action_type_distribution.entropy().mean()

        # gov action type head
        gov_action_type_distribution = self.gov_action_head(
            rnn_out, gov_action_type_masks_batch
        )
        gov_action_type_log_probs_batch = gov_action_type_distribution.log_probs(
            gov_action_type_batch
        )
        gov_action_type_dist_entropy = gov_action_type_distribution.entropy().mean()

        return (
            actor_type_log_probs_batch,
            actor_type_dist_entropy,
            city_id_log_probs_batch,
            city_action_type_log_probs_batch,
            city_action_type_dist_entropy,
            unit_id_log_probs_batch,
            unit_action_type_log_probs_batch,
            unit_action_type_dist_entropy,
            gov_action_type_log_probs_batch,
            gov_action_type_dist_entropy,
            value_preds_batch,
        )
