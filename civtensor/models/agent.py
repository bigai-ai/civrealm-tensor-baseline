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

    def __init__(
        self, args, observation_spaces, action_spaces, device=torch.device("cpu")
    ):
        super(Agent, self).__init__()
        self.args = args
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.device = device

        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]

        self.init_network(args)

        self.to(device)

    def init_network(self, args):
        # obtain input dimensions. TODO: be consistent with env
        self.rules_dim = self.observation_spaces["rules"].shape[0]
        self.player_dim = self.observation_spaces["player"].shape[0]
        self.others_player_dim = self.observation_spaces["others_player"].shape[
            1
        ]  # or Sequence?
        self.unit_dim = self.observation_spaces["unit"].shape[1]  # or Sequence?
        self.city_dim = self.observation_spaces["city"].shape[1]  # or Sequence?
        self.dipl_dim = self.observation_spaces["dipl"].shape[1]  # or Sequence?
        self.others_unit_dim = self.observation_spaces["others_unit"].shape[
            1
        ]  # or Sequence?
        self.others_city_dim = self.observation_spaces["others_city"].shape[
            1
        ]  # or Sequence?
        self.map_dim = self.observation_spaces["map"].shape

        # obtain output dimensions. TODO: be consistent with env
        self.actor_type_dim = self.action_spaces["actor_type"].n
        self.city_action_type_dim = self.action_spaces["city_action_type"].n
        self.unit_action_type_dim = self.action_spaces["unit_action_type"].n
        self.dipl_action_type_dim = self.action_spaces["dipl_action_type"].n
        self.gov_action_type_dim = self.action_spaces["gov_action_type"].n
        self.tech_action_type_dim = self.action_spaces["tech_action_type"].n

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

        self.others_player_embedding = nn.Sequential(
            nn.Linear(self.others_player_dim, self.hidden_dim), nn.ReLU()
        )

        self.others_player_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.unit_embedding = nn.Sequential(
            nn.Linear(self.unit_dim, self.hidden_dim), nn.ReLU()
        )

        self.unit_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.city_embedding = nn.Sequential(
            nn.Linear(self.city_dim, self.hidden_dim), nn.ReLU()
        )

        self.city_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.dipl_embedding = nn.Sequential(
            nn.Linear(self.dipl_dim, self.hidden_dim), nn.ReLU()
        )

        self.dipl_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.others_unit_embedding = nn.Sequential(
            nn.Linear(self.others_unit_dim, self.hidden_dim), nn.ReLU()
        )

        self.others_unit_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.others_city_embedding = nn.Sequential(
            nn.Linear(self.others_city_dim, self.hidden_dim), nn.ReLU()
        )

        self.others_city_encoder = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        self.map_encoder = nn.Sequential(
            nn.Conv2d(112, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
        )

        # initialize global transformer
        self.global_transformer = TransformerEncoder(
            self.hidden_dim, self.hidden_dim, self.n_head, self.n_layers, self.drop_prob
        )

        # initialize rnn.
        self.rnn = RNNLayer(
            9 * self.hidden_dim,
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

        self.dipl_id_head = PointerNetwork(self.hidden_dim, self.rnn_hidden_dim)

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

        self.dipl_action_head = Categorical(
            self.rnn_hidden_dim + self.hidden_dim,
            self.dipl_action_type_dim,
            self.initialization_method,
            self.gain,
        )

        self.gov_action_head = Categorical(
            self.rnn_hidden_dim,
            self.gov_action_type_dim,
            self.initialization_method,
            self.gain,
        )

        self.tech_action_head = Categorical(
            self.rnn_hidden_dim,
            self.tech_action_type_dim,
            self.initialization_method,
            self.gain,
        )

        self.eps = torch.tensor(1e-8).to(self.device)

    def encoding_step(
        self,
        rules,
        player,
        others_player,
        unit,
        city,
        dipl,
        others_unit,
        others_city,
        map,
        others_player_mask,
        unit_mask,
        city_mask,
        others_unit_mask,
        others_city_mask,
    ):
        """
        Args:
            rules: (batch_size, rules_dim)
            player: (batch_size, player_dim)
            others_player: (batch_size, n_max_others_player, others_player_dim)
            unit: (batch_size, n_max_unit, unit_dim)
            city: (batch_size, n_max_city, city_dim)
            others_unit: (batch_size, n_max_others_unit, others_unit_dim)
            others_city: (batch_size, n_max_others_city, others_city_dim)
            map: (batch_size, x_size, y_size, map_channels) TODO check input order
            others_player_mask: (batch_size, n_max_others_player, 1) Note: masks are 0 for padding, 1 for non-padding
            unit_mask: (batch_size, n_max_unit, 1)
            city_mask: (batch_size, n_max_city, 1)
            others_unit_mask: (batch_size, n_max_others_unit, 1)
            others_city_mask: (batch_size, n_max_others_city, 1)
        """
        map = map.permute(0, 3, 1, 2)  # (batch_size, map_channels, x_size, y_size)

        # encoding step
        rules_encoded = self.rules_encoder(rules)  # (batch_size, hidden_dim)

        player_encoded = self.player_encoder(player)  # (batch_size, hidden_dim)

        batch_size, n_max_others_player, others_player_dim = others_player.shape
        others_player_embedding = self.others_player_embedding(
            others_player.view(-1, self.others_player_dim)
        ).view(
            batch_size, n_max_others_player, self.hidden_dim
        )  # (batch_size, n_max_others_player, hidden_dim)
        others_player_encoded = self.others_player_encoder(
            others_player_embedding, others_player_mask
        )  # (batch_size, n_max_others_player, hidden_dim)

        batch_size, n_max_unit, unit_dim = unit.shape
        unit_embedding = self.unit_embedding(unit.view(-1, self.unit_dim)).view(
            batch_size, n_max_unit, self.hidden_dim
        )  # (batch_size, n_max_unit, hidden_dim)
        unit_encoded = self.unit_encoder(
            unit_embedding, unit_mask
        )  # (batch_size, n_max_unit, hidden_dim)

        batch_size, n_max_city, city_dim = city.shape
        city_embedding = self.city_embedding(city.view(-1, self.city_dim)).view(
            batch_size, n_max_city, self.hidden_dim
        )  # (batch_size, n_max_city, hidden_dim)
        city_encoded = self.city_encoder(
            city_embedding, city_mask
        )  # (batch_size, n_max_city, hidden_dim)

        batch_size, n_max_dipl, dipl_dim = dipl.shape
        dipl_embedding = self.dipl_embedding(dipl.view(-1, self.dipl_dim)).view(
            batch_size, n_max_dipl, self.hidden_dim
        )  # (batch_size, n_max_dipl, hidden_dim)
        dipl_encoded = self.dipl_encoder(
            dipl_embedding, others_player_mask
        )  # (batch_size, n_max_dipl, hidden_dim)

        batch_size, n_max_others_unit, others_unit_dim = others_unit.shape
        others_unit_embedding = self.others_unit_embedding(
            others_unit.view(-1, self.others_unit_dim)
        ).view(
            batch_size, n_max_others_unit, self.hidden_dim
        )  # (batch_size, n_max_others_unit, hidden_dim)
        others_unit_encoded = self.others_unit_encoder(
            others_unit_embedding, others_unit_mask
        )  # (batch_size, n_max_others_unit, hidden_dim)

        batch_size, n_max_others_city, others_city_dim = others_city.shape
        others_city_embedding = self.others_city_embedding(
            others_city.view(-1, self.others_city_dim)
        ).view(
            batch_size, n_max_others_city, self.hidden_dim
        )  # (batch_size, n_max_others_city, hidden_dim)
        others_city_encoded = self.others_city_encoder(
            others_city_embedding, others_city_mask
        )  # (batch_size, n_max_others_city, hidden_dim)

        map_encoded = self.map_encoder(map)  # (batch_size, hidden_dim)

        # global transformer step
        others_player_global_encoding = torch.sum(
            others_player_encoded * others_player_mask, dim=1
        ) / (torch.sum(
            others_player_mask, dim=1
        ) + self.eps)  # (batch_size, hidden_dim)

        unit_global_encoding = torch.sum(unit_encoded * unit_mask, dim=1) / (torch.sum(
            unit_mask, dim=1
        ) + self.eps)  # (batch_size, hidden_dim)

        city_global_encoding = torch.sum(city_encoded * city_mask, dim=1) / (torch.sum(
            city_mask, dim=1
        ) + self.eps)

        dipl_global_encoding = torch.sum(dipl_encoded * others_player_mask, dim=1) / (torch.sum(
            others_player_mask, dim=1
        ) + self.eps)

        others_unit_global_encoding = torch.sum(
            others_unit_encoded * others_unit_mask, dim=1
        ) / (torch.sum(others_unit_mask, dim=1) + self.eps)

        others_city_global_encoding = torch.sum(
            others_city_encoded * others_city_mask, dim=1
        ) / (torch.sum(others_city_mask, dim=1) + self.eps)

        global_encoding = torch.stack(
            [
                rules_encoded,
                player_encoded,
                others_player_global_encoding,
                unit_global_encoding,
                city_global_encoding,
                dipl_global_encoding,
                others_unit_global_encoding,
                others_city_global_encoding,
                map_encoded,
            ],
            dim=1,
        )  # (batch_size, 8, hidden_dim)

        global_encoding_processed = self.global_transformer(
            global_encoding, src_mask=None
        )  # (batch_size, 8, hidden_dim)

        return global_encoding_processed, unit_encoded, city_encoded, dipl_encoded

    def forward(
        self,
        rules,
        player,
        others_player,
        unit,
        city,
        dipl,
        others_unit,
        others_city,
        map,
        others_player_mask,
        unit_mask,
        city_mask,
        others_unit_mask,
        others_city_mask,
        actor_type_mask,
        city_id_mask,
        city_action_type_mask,
        unit_id_mask,
        unit_action_type_mask,
        dipl_id_mask,
        dipl_action_type_mask,
        gov_action_type_mask,
        tech_action_type_mask,
        rnn_hidden_state,
        mask,  # TODO check whether logic related to mask is correct
        deterministic,
    ):
        """
        Args:
            rules: (batch_size, rules_dim)
            player: (batch_size, player_dim)
            others_player: (batch_size, n_max_others_player, others_player_dim)
            unit: (batch_size, n_max_unit, unit_dim)
            city: (batch_size, n_max_city, city_dim)
            others_unit: (batch_size, n_max_others_unit, others_unit_dim)
            others_city: (batch_size, n_max_others_city, others_city_dim)
            map: (batch_size, x_size, y_size, map_channels) TODO check input order
            others_player_mask: (batch_size, n_max_others_player, 1) Note: masks are 0 for padding, 1 for non-padding
            unit_mask: (batch_size, n_max_unit, 1)
            city_mask: (batch_size, n_max_city, 1)
            others_unit_mask: (batch_size, n_max_others_unit, 1)
            others_city_mask: (batch_size, n_max_others_city, 1)
            actor_type_mask: (batch_size, actor_type_dim)
            city_id_mask: (batch_size, n_max_city, 1)
            city_action_type_mask: (batch_size, n_max_city, city_action_type_dim)
            unit_id_mask: (batch_size, n_max_unit, 1)
            unit_action_type_mask: (batch_size, n_max_unit, unit_action_type_dim)
            gov_action_type_mask: (batch_size, gov_action_type_dim)
            rnn_hidden_state: (batch_size, n_rnn_layers, rnn_hidden_dim)
            mask: (batch_size, 1)
            deterministic: if True use argmax, else sample from distribution
        """
        rules = torch.from_numpy(rules).to(self.device)
        player = torch.from_numpy(player).to(self.device)
        others_player = torch.from_numpy(others_player).to(self.device)
        unit = torch.from_numpy(unit).to(self.device)
        city = torch.from_numpy(city).to(self.device)
        dipl = torch.from_numpy(dipl).to(self.device)
        others_unit = torch.from_numpy(others_unit).to(self.device)
        others_city = torch.from_numpy(others_city).to(self.device)
        map = torch.from_numpy(map).to(self.device)
        others_player_mask = torch.from_numpy(others_player_mask).to(self.device)
        unit_mask = torch.from_numpy(unit_mask).to(self.device)
        city_mask = torch.from_numpy(city_mask).to(self.device)
        others_unit_mask = torch.from_numpy(others_unit_mask).to(self.device)
        others_city_mask = torch.from_numpy(others_city_mask).to(self.device)
        actor_type_mask = torch.from_numpy(actor_type_mask).to(self.device)
        city_id_mask = torch.from_numpy(city_id_mask).to(self.device)
        city_action_type_mask = torch.from_numpy(city_action_type_mask).to(self.device)
        unit_id_mask = torch.from_numpy(unit_id_mask).to(self.device)
        unit_action_type_mask = torch.from_numpy(unit_action_type_mask).to(self.device)
        dipl_id_mask = torch.from_numpy(dipl_id_mask).to(self.device)
        dipl_action_type_mask = torch.from_numpy(dipl_action_type_mask).to(self.device)
        gov_action_type_mask = torch.from_numpy(gov_action_type_mask).to(self.device)
        tech_action_type_mask = torch.from_numpy(tech_action_type_mask).to(self.device)
        rnn_hidden_state = torch.from_numpy(rnn_hidden_state).to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        # encoding step
        global_encoding_processed, unit_encoded, city_encoded, dipl_encoded = self.encoding_step(
            rules,
            player,
            others_player,
            unit,
            city,
            dipl,
            others_unit,
            others_city,
            map,
            others_player_mask,
            unit_mask,
            city_mask,
            others_unit_mask,
            others_city_mask,
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
            rnn_out, city_encoded, city_id_mask, deterministic
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
            rnn_out, unit_encoded, unit_id_mask, deterministic
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

        # dipl id head
        dipl_id, dipl_id_log_prob, dipl_chosen_encoded = self.dipl_id_head(
            rnn_out, dipl_encoded, dipl_id_mask, deterministic
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # dipl action type head
        dipl_action_input = torch.cat([rnn_out, dipl_chosen_encoded], dim=-1)
        chosen_dipl_action_type_mask = dipl_action_type_mask[
            torch.arange(batch_size), dipl_id.squeeze(), :
        ]  # TODO: check whether gradient is correct
        dipl_action_type_distribution = self.dipl_action_head(
            dipl_action_input, chosen_dipl_action_type_mask
        )
        dipl_action_type = (
            dipl_action_type_distribution.mode()
            if deterministic
            else dipl_action_type_distribution.sample()
        )
        dipl_action_type_log_prob = dipl_action_type_distribution.log_probs(
            dipl_action_type
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

        # tech action type head
        tech_action_type_distribution = self.tech_action_head(
            rnn_out, tech_action_type_mask
        )
        tech_action_type = (
            tech_action_type_distribution.mode()
            if deterministic
            else tech_action_type_distribution.sample()
        )
        tech_action_type_log_prob = tech_action_type_distribution.log_probs(
            tech_action_type
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
            dipl_id,
            dipl_id_log_prob,
            dipl_action_type,
            dipl_action_type_log_prob,
            gov_action_type,
            gov_action_type_log_prob,
            tech_action_type,
            tech_action_type_log_prob,
            value_pred,
            rnn_hidden_state,
        )

    def evaluate_actions(
        self,
        rules_batch,  # (episode_length * num_envs_per_batch, rules_dim)
        player_batch,  # (episode_length * num_envs_per_batch, player_dim)
        others_player_batch,  # (episode_length * num_envs_per_batch, n_max_others_player, others_player_dim)
        unit_batch,  # (episode_length * num_envs_per_batch, n_max_unit, unit_dim)
        city_batch,  # (episode_length * num_envs_per_batch, n_max_city, city_dim)
        dipl_batch,  # (episode_length * num_envs_per_batch, n_max_dipl, dipl_dim)
        others_unit_batch,  # (episode_length * num_envs_per_batch, n_max_others_unit, others_unit_dim)
        others_city_batch,  # (episode_length * num_envs_per_batch, n_max_others_city, others_city_dim)
        map_batch,  # (episode_length * num_envs_per_batch, x_size, y_size, map_channels)
        others_player_masks_batch,  # (episode_length * num_envs_per_batch, n_max_others_player, 1)
        unit_masks_batch,  # (episode_length * num_envs_per_batch, n_max_unit, 1)
        city_masks_batch,  # (episode_length * num_envs_per_batch, n_max_city, 1)
        others_unit_masks_batch,  # (episode_length * num_envs_per_batch, n_max_others_unit, 1)
        others_city_masks_batch,  # (episode_length * num_envs_per_batch, n_max_others_city, 1)
        rnn_hidden_states_batch,  # (1 * num_envs_per_batch, n_rnn_layers, rnn_hidden_dim)
        actor_type_batch,  # (episode_length * num_envs_per_batch, 1)
        actor_type_masks_batch,  # (episode_length * num_envs_per_batch, actor_type_dim)
        city_id_batch,  # (episode_length * num_envs_per_batch, 1)
        city_id_masks_batch,  # (episode_length * num_envs_per_batch, n_max_city, 1)
        city_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        city_action_type_masks_batch,  # (episode_length * num_envs_per_batch, city_action_type_dim)
        unit_id_batch,  # (episode_length * num_envs_per_batch, 1)
        unit_id_masks_batch,  # (episode_length * num_envs_per_batch, n_max_unit, 1)
        unit_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        unit_action_type_masks_batch,  # (episode_length * num_envs_per_batch, unit_action_type_dim)
        dipl_id_batch,  # (episode_length * num_envs_per_batch, 1)
        dipl_id_masks_batch,  # (episode_length * num_envs_per_batch, n_max_dipl, 1)
        dipl_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        dipl_action_type_masks_batch,  # (episode_length * num_envs_per_batch, dipl_action_type_dim)
        gov_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        gov_action_type_masks_batch,  # (episode_length * num_envs_per_batch, gov_action_type_dim)
        tech_action_type_batch,  # (episode_length * num_envs_per_batch, 1)
        tech_action_type_masks_batch,  # (episode_length * num_envs_per_batch, tech_action_type_dim)
        masks_batch,  # (episode_length * num_envs_per_batch, 1)
    ):
        # encoding step
        global_encoding_processed, unit_encoded, city_encoded, dipl_encoded = self.encoding_step(
            rules_batch,
            player_batch,
            others_player_batch,
            unit_batch,
            city_batch,
            dipl_batch,
            others_unit_batch,
            others_city_batch,
            map_batch,
            others_player_masks_batch,
            unit_masks_batch,
            city_masks_batch,
            others_unit_masks_batch,
            others_city_masks_batch,
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
            rnn_out, city_encoded, city_id_masks_batch, city_id_batch
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
        (
            unit_id_log_probs_batch,
            unit_chosen_encoded,
        ) = self.unit_id_head.evaluate_actions(
            rnn_out, unit_encoded, unit_id_masks_batch, unit_id_batch
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

        # dipl id head
        (
            dipl_id_log_probs_batch,
            dipl_chosen_encoded,
        ) = self.dipl_id_head.evaluate_actions(
            rnn_out, dipl_encoded, dipl_id_masks_batch, dipl_id_batch
        )  # (batch_size, 1), (batch_size, hidden_dim)

        # dipl action type head
        dipl_action_input = torch.cat([rnn_out, dipl_chosen_encoded], dim=-1)
        chosen_dipl_action_type_mask = dipl_action_type_masks_batch[
            torch.arange(batch_size), dipl_id_batch.squeeze(), :
        ]  # TODO: check whether gradient is correct
        dipl_action_type_distribution = self.dipl_action_head(
            dipl_action_input, chosen_dipl_action_type_mask
        )
        dipl_action_type_log_probs_batch = dipl_action_type_distribution.log_probs(
            dipl_action_type_batch
        )
        dipl_action_type_dist_entropy = dipl_action_type_distribution.entropy().mean()

        # gov action type head
        gov_action_type_distribution = self.gov_action_head(
            rnn_out, gov_action_type_masks_batch
        )
        gov_action_type_log_probs_batch = gov_action_type_distribution.log_probs(
            gov_action_type_batch
        )
        gov_action_type_dist_entropy = gov_action_type_distribution.entropy().mean()

        # tech action type head
        tech_action_type_distribution = self.tech_action_head(
            rnn_out, tech_action_type_masks_batch
        )
        tech_action_type_log_probs_batch = tech_action_type_distribution.log_probs(
            tech_action_type_batch
        )
        tech_action_type_dist_entropy = tech_action_type_distribution.entropy().mean()

        return (
            actor_type_log_probs_batch,
            actor_type_dist_entropy,
            city_id_log_probs_batch,
            city_action_type_log_probs_batch,
            city_action_type_dist_entropy,
            unit_id_log_probs_batch,
            unit_action_type_log_probs_batch,
            unit_action_type_dist_entropy,
            dipl_id_log_probs_batch,
            dipl_action_type_log_probs_batch,
            dipl_action_type_dist_entropy,
            gov_action_type_log_probs_batch,
            gov_action_type_dist_entropy,
            tech_action_type_log_probs_batch,
            tech_action_type_dist_entropy,
            value_preds_batch,
        )
