import torch
import torch.nn as nn
from torch.distributions import Categorical


class PointerNetwork(nn.Module):
    def __init__(self, hidden_dim, lstm_hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.w_q = nn.Linear(self.lstm_hidden_dim, self.hidden_dim)
        self.w_k = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, lstm_output, hidden_state, mask):
        """
        Args:
            lstm_outputs: (batch_size, lstm_hidden_dim)
            hidden_state: (batch_size, max_length, hidden_dim)
            mask: (batch_size, max_length, 1)

        Returns:
            ids: (batch_size, 1)
            chosen_encoded: (batch_size, hidden_dim)
        """
        q = self.w_q(lstm_output)  # (batch_size, hidden_dim)
        k = self.w_k(hidden_state)  # (batch_size, max_length, hidden_dim)
        scores = torch.bmm(k, q.unsqueeze(2)).squeeze(2)  # (batch_size, max_length)
        scores = scores.masked_fill(mask.squeeze(2) == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)  # (batch_size, max_length)
        id_distribution = Categorical(scores)
        ids = id_distribution.sample()  # (batch_size, 1)
        chosen_encoded = torch.gather(
            hidden_state, dim=1, index=ids.unsqueeze(1).expand(-1, -1, self.hidden_dim)
        ).squeeze(1)
        return ids, chosen_encoded
