import torch
import torch.nn as nn
from civtensor.models.base.distributions import FixedCategorical


class PointerNetwork(nn.Module):
    def __init__(self, hidden_dim, rnn_hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.w_q = nn.Linear(self.rnn_hidden_dim, self.hidden_dim)
        self.w_k = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, rnn_output, hidden_state, mask, deterministic):
        """
        Args:
            rnn_outputs: (batch_size, rnn_hidden_dim)
            hidden_state: (batch_size, max_length, hidden_dim)
            mask: (batch_size, max_length, 1)
            deterministic: bool

        Returns:
            ids: (batch_size, 1)
            log_probs: (batch_size, 1)
            chosen_encoded: (batch_size, hidden_dim)
        """
        q = self.w_q(rnn_output)  # (batch_size, hidden_dim)
        k = self.w_k(hidden_state)  # (batch_size, max_length, hidden_dim)
        scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2)  # (batch_size, max_length)
        scores[mask.squeeze(2) == 0] = -1e10
        id_distribution = FixedCategorical(logits=scores)
        if deterministic:
            ids = id_distribution.mode()
        else:
            ids = id_distribution.sample()  # (batch_size, 1)
        log_probs = id_distribution.log_probs(ids)  # (batch_size, 1)
        chosen_encoded = torch.gather(
            hidden_state, dim=1, index=ids.unsqueeze(1).expand(-1, -1, self.hidden_dim)
        ).squeeze(1)
        return ids, log_probs, chosen_encoded

    def evaluate_actions(self, rnn_output, hidden_state, mask, chosen_id):
        """
        Args:
            rnn_outputs: (batch_size, rnn_hidden_dim)
            hidden_state: (batch_size, max_length, hidden_dim)
            mask: (batch_size, max_length, 1)
            chosen_id: (batch_size, 1)

        Returns:
            log_probs: (batch_size, 1)
            chosen_encoded: (batch_size, hidden_dim)
        """
        q = self.w_q(rnn_output)  # (batch_size, hidden_dim)
        k = self.w_k(hidden_state)  # (batch_size, max_length, hidden_dim)
        scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2)  # (batch_size, max_length)
        scores[mask.squeeze(2) == 0] = -1e10
        id_distribution = FixedCategorical(logits=scores)
        log_probs = id_distribution.log_probs(chosen_id)  # (batch_size, 1)
        chosen_encoded = torch.gather(
            hidden_state,
            dim=1,
            index=chosen_id.unsqueeze(1).expand(-1, -1, self.hidden_dim),
        ).squeeze(1)
        return log_probs, chosen_encoded
