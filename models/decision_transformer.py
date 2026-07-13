import torch
import torch.nn as nn
import math


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for off-road autonomous navigation.
    Based on Chen et al. (2021) — adapted for continuous vehicle control.

    Input sequence per timestep: (return-to-go, state, action)
    Output: predicted action at each timestep
    """

    def __init__(
        self,
        state_dim=8,
        action_dim=3,
        hidden_dim=128,
        n_layers=3,
        n_heads=1,
        context_length=20,
        dropout=0.1,
        max_timestep=500
    ):
        super(DecisionTransformer, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length

        # Token embeddings — one per token type
        self.embed_rtg    = nn.Linear(1, hidden_dim)
        self.embed_state  = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)

        # Layer normalisation for each embedding
        self.norm_rtg    = nn.LayerNorm(hidden_dim)
        self.norm_state  = nn.LayerNorm(hidden_dim)
        self.norm_action = nn.LayerNorm(hidden_dim)

        # Timestep embedding — shared across token types
        self.embed_timestep = nn.Embedding(max_timestep, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Output activation — tanh to bound actions
        self.action_tanh = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        """Initialise weights following GPT-2 convention."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, states, actions, rtg, timesteps):
        """
        Forward pass through the Decision Transformer.

        Args:
            states:    (batch, context_length, state_dim)
            actions:   (batch, context_length, action_dim)
            rtg:       (batch, context_length, 1)
            timesteps: (batch, context_length)

        Returns:
            action_preds: (batch, context_length, action_dim)
        """
        batch_size, seq_len, _ = states.shape

        # Timestep embeddings
        time_embeddings = self.embed_timestep(timesteps)

        # Token embeddings + timestep embeddings
        rtg_embeddings    = self.norm_rtg(
            self.embed_rtg(rtg)
        ) + time_embeddings

        state_embeddings  = self.norm_state(
            self.embed_state(states)
        ) + time_embeddings

        action_embeddings = self.norm_action(
            self.embed_action(actions)
        ) + time_embeddings

        # Interleave tokens: (rtg_1, s_1, a_1, rtg_2, s_2, a_2, ...)
        # Shape: (batch, 3*seq_len, hidden_dim)
        token_sequence = torch.stack(
            [rtg_embeddings, state_embeddings, action_embeddings],
            dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_dim)

        # Causal mask — upper triangular, True = ignore
        mask = torch.triu(
            torch.ones(3 * seq_len, 3 * seq_len, device=states.device),
            diagonal=1
        ).bool()

        # Transformer forward pass
        transformer_output = self.transformer(
            token_sequence,
            mask=mask
        )

        # Extract state token outputs (indices 1, 4, 7, ...)
        # State tokens are at positions 1, 4, 7 ... (every 3rd, offset 1)
        state_token_outputs = transformer_output[
            :, 1::3, :
        ]  # (batch, seq_len, hidden_dim)

        # Predict actions from state token outputs
        action_preds = self.action_tanh(
            self.action_head(state_token_outputs)
        )

        return action_preds


class DTDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Decision Transformer training.
    Samples random subsequences of length context_length
    from the collected episodes.
    """

    def __init__(self, episodes, context_length=20):
        self.episodes = episodes
        self.context_length = context_length

        # Build index of valid (episode_idx, start_idx) pairs
        self.indices = []
        for ep_idx, episode in enumerate(episodes):
            ep_len = episode['length']
            for start in range(ep_len - context_length):
                self.indices.append((ep_idx, start))

        print(f"Dataset: {len(episodes)} episodes, "
              f"{len(self.indices)} subsequences")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start = self.indices[idx]
        episode = self.episodes[ep_idx]

        end = start + self.context_length

        states    = torch.tensor(
            episode['states'][start:end],
            dtype=torch.float32
        )
        actions   = torch.tensor(
            episode['actions'][start:end],
            dtype=torch.float32
        )
        rtg       = torch.tensor(
            episode['rtg'][start:end],
            dtype=torch.float32
        ).unsqueeze(-1)
        timesteps = torch.tensor(
            episode['timesteps'][start:end],
            dtype=torch.long
        )

        return states, actions, rtg, timesteps