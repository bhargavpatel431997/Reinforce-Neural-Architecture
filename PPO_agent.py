# ... (Keep all imports and class definitions the same as the previous PPO_agent.py) ...
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical # Removed Normal as no continuous actions now
import math
import os
import json # <-- Add import for JSON saving
import traceback # <-- Add import for error printing

# --- Reusing Transformer Building Blocks ---
# ... (MultiHeadSelfAttention, EnhancedRBFLayer, PositionalEncoding, FeedForward, EnhancedTransformerLayer, GraphAttention, EnhancedGraphTransformerEncoder classes remain the same) ...
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with enhanced capabilities.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, mask=None, return_attention=False): # Modified to accept separate Q, K, V inputs
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1) # Key/Value sequence length might differ

        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (batch, nhead, seq_len_q, seq_len_kv)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided (mask shape should broadcast: e.g., (batch, 1, 1, seq_len_kv))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute attention output (batch, nhead, seq_len_q, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # Final projection
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output


class EnhancedRBFLayer(nn.Module):
    """
    Enhanced Radial Basis Function Layer with adaptive centers and multiple kernels.
    """
    def __init__(self, in_features, out_features, num_kernels=3):
        super(EnhancedRBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_kernels = num_kernels

        # Centers for RBFs - learnable parameters
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))

        # Multiple bandwidth parameters for different kernels
        self.log_sigmas = nn.Parameter(torch.Tensor(num_kernels, out_features))

        # Kernel mixture weights
        self.kernel_weights = nn.Parameter(torch.Tensor(num_kernels, out_features))

        # Adaptive scaling factor
        self.scaling = nn.Parameter(torch.Tensor(out_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize centers using Xavier initialization
        nn.init.xavier_uniform_(self.centres)

        # Initialize log sigmas
        nn.init.uniform_(self.log_sigmas, -1, 0)

        # Initialize kernel weights (ensure they sum to 1 after softmax)
        # Initialize uniformly before softmax for better starting point
        nn.init.uniform_(self.kernel_weights)

        # Initialize scaling
        nn.init.ones_(self.scaling)

    def forward(self, x):
        # x shape: (batch_size * seq_len, in_features) or (batch_size, in_features)
        is_batched_seq = len(x.shape) == 2 and x.size(0) > 1 # Check if input is flattened batch*seq
        original_shape = x.shape
        if is_batched_seq:
            batch_seq_size = x.size(0)
            size = (batch_seq_size, self.out_features, self.in_features)
            x_expanded = x.unsqueeze(1).expand(size)
            c_expanded = self.centres.unsqueeze(0).expand(size)
        else: # Single item or batch without sequence dim flattened
            batch_size = x.size(0)
            size = (batch_size, self.out_features, self.in_features)
            x_expanded = x.unsqueeze(1).expand(size)
            c_expanded = self.centres.unsqueeze(0).expand(size)


        # Calculate squared Euclidean distances
        # Use torch.cdist for potentially better efficiency if available and suitable
        # distances = torch.cdist(x_expanded, c_expanded, p=2) # This might not work directly with expanded dims
        distances = (x_expanded - c_expanded).pow(2).sum(-1).sqrt() # (batch_size or batch_seq_size, out_features)

        # Apply multiple kernels with different bandwidths
        output = torch.zeros_like(distances) # Shape: (batch_size or batch_seq_size, out_features)

        # Ensure kernel weights are probabilities
        kernel_probs = F.softmax(self.kernel_weights, dim=0) # Softmax over kernels

        for k in range(self.num_kernels):
            # Get sigma for this kernel
            sigma = torch.exp(self.log_sigmas[k]).unsqueeze(0) # Shape: (1, out_features)

            # Calculate kernel output (Gaussian RBF)
            kernel_out = torch.exp(-(distances * sigma).pow(2))

            # Apply kernel weight (probability)
            kernel_weight = kernel_probs[k].unsqueeze(0) # Shape: (1, out_features)
            output += kernel_out * kernel_weight

        # Apply scaling
        output = output * self.scaling.unsqueeze(0) # Shape: (batch_size or batch_seq_size, out_features)

        # Reshape if input was flattened batch*seq
        if is_batched_seq and len(original_shape) > 2: # Make sure original had sequence dim
             output = output.view(original_shape[0], original_shape[1], -1)

        return output


class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with learnable parameters."""
    def __init__(self, d_model, max_len=100, dropout=0.1): # Increased max_len for grid
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Make positional encoding learnable
        self.register_buffer('pe_base', pe)
        self.pe_weight = nn.Parameter(torch.ones(1, max_len, d_model))
        self.pe_bias = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Apply learnable transformation to base positional encoding
        # Ensure max_len is sufficient
        if seq_len > self.pe_base.size(1):
             # Dynamically extend positional encoding if needed
             print(f"Warning: Extending PositionalEncoding max_len from {self.pe_base.size(1)} to {seq_len}")
             new_max_len = seq_len
             new_pe = torch.zeros(new_max_len, self.pe_weight.size(2)) # d_model
             new_position = torch.arange(0, new_max_len, dtype=torch.float).unsqueeze(1)
             new_div_term = torch.exp(torch.arange(0, self.pe_weight.size(2), 2).float() * (-math.log(10000.0) / self.pe_weight.size(2)))
             new_pe[:, 0::2] = torch.sin(new_position * new_div_term)
             new_pe[:, 1::2] = torch.cos(new_position * new_div_term)
             self.pe_base = new_pe.unsqueeze(0).to(x.device)
             # Also need to resize pe_weight and pe_bias, potentially re-initializing the new parts
             old_max_len = self.pe_weight.size(1)
             new_weight = torch.ones(1, new_max_len, self.pe_weight.size(2), device=x.device)
             new_bias = torch.zeros(1, new_max_len, self.pe_weight.size(2), device=x.device)
             new_weight[:, :old_max_len, :] = self.pe_weight.data
             new_bias[:, :old_max_len, :] = self.pe_bias.data
             self.pe_weight = nn.Parameter(new_weight)
             self.pe_bias = nn.Parameter(new_bias)


        pe = self.pe_base[:, :seq_len, :] * self.pe_weight[:, :seq_len, :] + self.pe_bias[:, :seq_len, :]

        # Add positional encoding to input
        x = x + pe
        return self.dropout(x)


class FeedForward(nn.Module):
    """Enhanced feed-forward network with Gated Linear Units."""
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # GLU components
        self.gate_linear = nn.Linear(d_model, d_ff)

        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        # Apply Gated Linear Unit
        gate = torch.sigmoid(self.gate_linear(x))
        x_ff = self.linear1(x)
        x_activated = self.activation(x_ff) * gate # Apply activation and gate

        # Second linear layer and dropout
        x = self.dropout(self.linear2(x_activated))
        return x


class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with advanced normalization and attention."""
    def __init__(self, d_model, nhead, d_ff, dropout=0.1, activation='gelu',
                 norm_type='layer', pre_norm=True):
        super(EnhancedTransformerLayer, self).__init__()

        # Self-attention
        # Use the modified MultiHeadSelfAttention that accepts query, key, value
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)

        # Normalization layers
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            # BatchNorm needs number of features (d_model)
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        else:
            raise ValueError(f"Unsupported normalization: {norm_type}")

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Flag for pre-normalization vs post-normalization
        self.pre_norm = pre_norm
        self.norm_type = norm_type

    def _apply_batch_norm(self, x, norm_layer):
        """Helper for applying BatchNorm1d to sequence data."""
        # Input x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        # BatchNorm1d expects input shape (batch_size, channels/d_model, seq_len)
        x = x.transpose(1, 2)
        x = norm_layer(x)
        x = x.transpose(1, 2) # Back to (batch_size, seq_len, d_model)
        return x

    def _apply_norm(self, x, norm_layer):
        """Apply appropriate normalization based on type."""
        if self.norm_type == 'batch':
            return self._apply_batch_norm(x, norm_layer)
        # LayerNorm applies over the last dimension (d_model)
        return norm_layer(x)

    def forward(self, x, mask=None):
        # Self-attention block
        if self.pre_norm:
            # Pre-normalization
            attn_input = self._apply_norm(x, self.norm1)
            # Pass attn_input as query, key, value
            attn_output = self.self_attn(attn_input, attn_input, attn_input, mask)
            x = x + self.dropout(attn_output) # Residual connection

            # Feed-forward block
            ff_input = self._apply_norm(x, self.norm2)
            ff_output = self.feed_forward(ff_input)
            x = x + self.dropout(ff_output) # Residual connection
        else:
            # Post-normalization
            # Pass x as query, key, value
            attn_output = self.self_attn(x, x, x, mask)
            x = self._apply_norm(x + self.dropout(attn_output), self.norm1) # Add -> Norm

            # Feed-forward block
            ff_output = self.feed_forward(x)
            x = self._apply_norm(x + self.dropout(ff_output), self.norm2) # Add -> Norm

        return x


class GraphAttention(nn.Module):
    """Graph Attention Network (GAT) for processing the graph state."""
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1, alpha=0.2):
        super(GraphAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = out_features // num_heads
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"

        # Linear transformation for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, self.head_dim))

        # Attention parameters
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim, 1))

        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialize parameters
        for i in range(num_heads):
            nn.init.xavier_uniform_(self.W[i])
            nn.init.xavier_uniform_(self.a[i])

    def forward(self, x, adj=None):
        """
        x: Node features (batch_size, num_nodes, in_features)
        adj: Adjacency matrix (batch_size, num_nodes, num_nodes) or None for fully connected
        """
        batch_size, num_nodes, _ = x.size()

        # Linear transformation for each head: (batch, num_nodes, in) @ (in, head_dim) -> (batch, num_nodes, head_dim)
        # Stack results for heads: (batch, num_heads, num_nodes, head_dim)
        Wh = torch.stack([torch.matmul(x, self.W[i]) for i in range(self.num_heads)], dim=1)

        # Prepare for attention mechanism
        Wh1 = Wh.unsqueeze(3) # (batch, nhead, N, 1, head_dim)
        Wh2 = Wh.unsqueeze(2) # (batch, nhead, 1, N, head_dim)
        Wh_cat = torch.cat([Wh1.expand(-1, -1, num_nodes, num_nodes, -1),
                            Wh2.expand(-1, -1, num_nodes, num_nodes, -1)], dim=-1)
        # Shape: (batch, nhead, N, N, 2 * head_dim)

        # Calculate attention coefficients
        e_list = []
        for i in range(self.num_heads):
             Wh_cat_head = Wh_cat[:, i, :, :, :] # (batch, N, N, 2*head_dim)
             a_head = self.a[i] # (2*head_dim, 1)
             e_head = torch.matmul(Wh_cat_head, a_head).squeeze(-1) # (batch, N, N)
             e_list.append(e_head)
        e = torch.stack(e_list, dim=1) # Shape: (batch, nhead, N, N)

        # Apply LeakyReLU
        e = self.leakyrelu(e)

        # Apply adjacency mask if provided
        if adj is not None:
            e = e.masked_fill(adj.unsqueeze(1) == 0, -9e15)

        # Apply softmax to get attention weights
        attention = F.softmax(e, dim=-1) # Softmax over the last dimension (source nodes j)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # Shape: (batch, nhead, N, N)

        # Apply attention to node features (value projections, which are Wh)
        h_prime = torch.matmul(attention, Wh)
        # Shape: (batch, nhead, N, head_dim)

        # Combine heads by concatenating
        h_prime = h_prime.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, self.out_features)

        return F.elu(h_prime)


class EnhancedGraphTransformerEncoder(nn.Module):
    """
    Enhanced transformer encoder for graph data (grid) with GAT and advanced attention.
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, d_ff, grid_size, # Added grid_size
                 dropout=0.1, activation='gelu', norm_type='layer', pre_norm=True):
        super(EnhancedGraphTransformerEncoder, self).__init__()

        self.grid_size = grid_size
        num_nodes = grid_size * grid_size # Total nodes in the grid

        # Initial embedding
        self.embedding = nn.Linear(input_dim, d_model)

        # Layer normalization for embedding
        if norm_type == 'layer':
            self.embed_norm = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            self.embed_norm = nn.BatchNorm1d(d_model) # Use 1d for features

        # Graph Attention Layer for spatial processing
        self.graph_attention = GraphAttention(d_model, d_model, nhead, dropout)

        # Positional encoding (needs max_len >= num_nodes)
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_nodes, dropout=dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(
                d_model, nhead, d_ff, dropout, activation, norm_type, pre_norm
            ) for _ in range(num_layers)
        ])

        # RBF layer (optional, can be removed if causing issues)
        self.use_rbf = False # Flag to easily disable (Set to False as it might complicate things)
        if self.use_rbf:
            self.rbf = EnhancedRBFLayer(d_model, d_model)

        # Final layer normalization
        if norm_type == 'layer':
            self.final_norm = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            self.final_norm = nn.BatchNorm1d(d_model) # Use 1d for features

        self.dropout = nn.Dropout(dropout)
        self.norm_type = norm_type

    def _apply_batch_norm(self, x, norm_layer):
        """Helper for applying BatchNorm1d to graph/sequence data."""
        # Input x shape: (batch_size, num_nodes, d_model)
        batch_size, num_nodes, d_model = x.size()
        # BatchNorm1d expects input shape (batch_size, channels/d_model, num_nodes)
        x = x.transpose(1, 2)
        x = norm_layer(x)
        x = x.transpose(1, 2) # Back to (batch_size, num_nodes, d_model)
        return x

    def _apply_norm(self, x, norm_layer):
        """Apply appropriate normalization based on type."""
        if self.norm_type == 'batch':
            return self._apply_batch_norm(x, norm_layer)
        # LayerNorm applies over the last dimension (d_model)
        return norm_layer(x)

    def forward(self, src, src_mask=None):
        # src shape: (batch, height, width, channels)
        batch_size, height, width, channels = src.shape
        num_nodes = height * width
        # Reshape to (batch, num_nodes, channels)
        src = src.reshape(batch_size, num_nodes, channels)

        # Initial embedding
        x = self.embedding(src)
        x = self._apply_norm(x, self.embed_norm)

        # Apply graph attention for spatial relationships
        x = self.graph_attention(x)

        # Add positional encoding (encoding the flattened grid position)
        x = self.pos_encoder(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=None) # Assuming no padding mask needed for grid nodes

        # Final normalization before RBF
        x = self._apply_norm(x, self.final_norm)

        # Apply RBF layer (optional)
        if self.use_rbf:
            flat_x = x.reshape(-1, x.size(-1)) # Flatten batch and node dimensions
            rbf_x = self.rbf(flat_x)
            x = rbf_x.reshape(batch_size, num_nodes, -1) # Reshape back

        return x

# --- New Network for Self-Play ---
class SelfPlayTransformerPPONetwork(nn.Module):
    """
    Transformer-based PPO network adapted for MathSelfPlayEnv.
    Processes the board state using a Graph Transformer.
    Outputs distributions for operation_id and placement_strategy, plus value.
    """
    def __init__(self, state_dim, action_dims):
        """
        Args:
            state_dim: Dictionary containing 'board' shape (grid_size, grid_size, channels)
            action_dims: Dictionary containing 'operation_id' and 'placement_strategy' counts
        """
        super(SelfPlayTransformerPPONetwork, self).__init__()

        board_shape = state_dim['board']
        grid_size = board_shape[0]
        input_channels = board_shape[2]

        # Model dimensions (can be tuned)
        d_model = 128
        nhead = 4 # Reduced heads might be sufficient
        num_layers = 2 # Reduced layers
        d_ff = 256 # Reduced feed-forward dim
        dropout = 0.1

        # Transformer encoder for the board state
        self.board_transformer = EnhancedGraphTransformerEncoder(
            input_dim=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            d_ff=d_ff,
            grid_size=grid_size, # Pass grid_size
            dropout=dropout,
            activation='gelu',
            norm_type='layer', # LayerNorm is generally more stable for transformers
            pre_norm=True
        )

        # --- Output Heads ---
        # Use features from the transformer output (e.g., mean-pooled features)

        # Actor heads for the two discrete actions
        self.operation_head = nn.Linear(d_model, action_dims['operation_id'])
        self.placement_head = nn.Linear(d_model, action_dims['placement_strategy'])

        # Critic head for the value function
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Dictionary containing 'board' tensor (batch, grid, grid, channels)

        Returns:
            action_dists: Dictionary of action distributions {'operation_id', 'placement_strategy'}
            value: Value function estimate tensor (batch, 1)
        """
        board_state = state['board'] # Shape: (batch, grid, grid, channels)

        # Process board state with the graph transformer
        # Output shape: (batch, num_nodes, d_model) where num_nodes = grid*grid
        board_features_nodes = self.board_transformer(board_state)

        # Aggregate node features to get a global representation for decision making
        # Mean pooling over the node dimension
        aggregated_features = board_features_nodes.mean(dim=1) # Shape: (batch, d_model)

        # --- Actor Heads ---
        operation_logits = self.operation_head(aggregated_features)
        placement_logits = self.placement_head(aggregated_features)

        # Create categorical distributions
        operation_dist = Categorical(logits=operation_logits)
        placement_dist = Categorical(logits=placement_logits)

        action_dists = {
            'operation_id': operation_dist,
            'placement_strategy': placement_dist
        }

        # --- Critic Head ---
        value = self.value_head(aggregated_features) # Shape: (batch, 1)

        return action_dists, value


# --- New Agent Class for Self-Play ---
class SelfPlayPPOAgent:
    """
    PPO agent adapted for the MathSelfPlayEnv using a shared policy network.
    """
    def __init__(
        self,
        env, # The MathSelfPlayEnv instance
        state_dim, # From env.observation_space
        action_dims, # From env.action_space
        lr=3e-4,
        gamma=0.99,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        buffer_size=2048,
        batch_size=64,
        update_epochs=10,
        lr_schedule=True
    ):
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.lr_schedule = lr_schedule

        # Use the new network architecture
        self.policy = SelfPlayTransformerPPONetwork(state_dim, action_dims)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        # Learning rate scheduler (optional)
        if lr_schedule:
            # Adjust T_max based on expected training duration
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=lr/10
            )

        # Initialize buffer (simplified structure)
        self.buffer = self.PPOBuffer(buffer_size)

    class PPOBuffer:
        """Buffer for storing trajectories in self-play."""
        def __init__(self, size):
            # Store observations directly (dictionaries)
            self.observations = [None] * size
            # Store actions directly (dictionaries)
            self.actions = [None] * size
            # Store log probs as single values (sum of individual action log probs)
            self.log_probs = np.zeros(size, dtype=np.float32)
            self.rewards = np.zeros(size, dtype=np.float32)
            self.values = np.zeros(size, dtype=np.float32)
            self.dones = np.zeros(size, dtype=np.float32) # Use float for GAE calculation
            self.size = size
            self.ptr = 0
            self.current_size = 0

        def store(self, obs, action, log_prob, reward, value, done):
            """Store a transition."""
            self.observations[self.ptr] = obs
            self.actions[self.ptr] = action
            self.log_probs[self.ptr] = log_prob # Store the single combined log_prob
            self.rewards[self.ptr] = reward
            self.values[self.ptr] = value
            self.dones[self.ptr] = float(done) # Store done as float (0.0 or 1.0)

            self.ptr = (self.ptr + 1) % self.size
            self.current_size = min(self.current_size + 1, self.size)

        def get_batch(self):
            """Get all data currently in the buffer."""
            assert self.current_size == self.size, "Buffer must be full before getting data."
            # --- IMPORTANT: Return copies to avoid modifying buffer during update ---
            obs_batch = self.observations[:self.current_size]
            action_batch = self.actions[:self.current_size]
            log_probs_batch = self.log_probs[:self.current_size].copy()
            rewards_batch = self.rewards[:self.current_size].copy()
            values_batch = self.values[:self.current_size].copy()
            dones_batch = self.dones[:self.current_size].copy()

            self.ptr = 0 # Reset pointer after getting data
            self.current_size = 0 # Reset size

            return (
                obs_batch, # List of observation dicts
                action_batch,      # List of action dicts
                log_probs_batch,
                rewards_batch,
                values_batch,
                dones_batch
            )

        def is_full(self):
            return self.current_size == self.size

    def _obs_to_tensor(self, obs_list):
        """Converts a list of observation dictionaries to a tensor dictionary."""
        # Assumes all observations in the list have the same structure
        if not obs_list:
             return {} # Handle empty list case
        keys = obs_list[0].keys()
        batch_tensors = {}
        for key in keys:
            try:
                # Stack numpy arrays from each observation dict
                # Special handling for 'board' which is likely the main input
                if key == 'board':
                     # Ensure all boards have the same shape before stacking
                     first_shape = obs_list[0][key].shape
                     if not all(obs[key].shape == first_shape for obs in obs_list):
                          raise ValueError(f"Inconsistent board shapes in batch for key '{key}'")
                     batch_tensors[key] = torch.FloatTensor(np.stack([obs[key] for obs in obs_list]))
                elif key in ['current_player', 'steps_taken']: # Example: Handle scalar features if present
                    batch_tensors[key] = torch.LongTensor([obs[key] for obs in obs_list])
                # Add other keys if necessary
            except Exception as e:
                 print(f"Error processing key '{key}' in _obs_to_tensor: {e}")
                 # Optionally skip this key or raise error depending on importance
        return batch_tensors


    def get_action(self, obs, deterministic=False):
        """
        Sample an action from the policy based on the current observation.

        Args:
            obs: Dictionary containing the current observation ('board', 'current_player', 'steps_taken')
            deterministic: Whether to sample deterministically (take argmax)

        Returns:
            action: Dictionary of sampled actions {'operation_id', 'placement_strategy'}
            log_prob: Combined log probability of the sampled actions (scalar tensor)
            value: Value estimate (scalar)
        """
        # Convert observation to tensor batch of size 1
        obs_tensor = self._obs_to_tensor([obs]) # Pass as a list

        # Forward pass through the policy network
        with torch.no_grad():
            action_dists, value_tensor = self.policy.forward(obs_tensor)

        # Sample actions from distributions
        op_dist = action_dists['operation_id']
        place_dist = action_dists['placement_strategy']

        if deterministic:
            op_action = torch.argmax(op_dist.probs, dim=-1)
            place_action = torch.argmax(place_dist.probs, dim=-1)
        else:
            op_action = op_dist.sample()
            place_action = place_dist.sample()

        # Calculate combined log probability
        log_prob = op_dist.log_prob(op_action) + place_dist.log_prob(place_action)

        # Prepare action dictionary for the environment
        action = {
            'operation_id': op_action.item(),
            'placement_strategy': place_action.item()
        }

        return action, log_prob.item(), value_tensor.item() # Return scalar log_prob and value

    def train(self, num_episodes=1000, max_steps_per_episode=100, save_path=None, graph_save_path=None): # Added graph_save_path
        """
        Train the agent using self-play.
        """
        episode_rewards_p1 = []
        episode_rewards_p2 = []
        all_episode_losses = [] # Track loss per episode
        best_avg_reward = float('-inf')
        best_episode_final_loss = float('inf') # Track best loss achieved
        best_graph_structure = None # Store the best graph structure

        print(f"Starting training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            obs, info = self.env.reset() # Get initial info
            episode_reward_p1 = 0
            episode_reward_p2 = 0
            ep_steps = 0
            ep_losses = [] # Track losses within an episode update cycle
            done = False # Initialize done flag

            while not done: # Loop until episode ends
                current_player = obs['current_player'] # Get current player from observation

                # Sample action using the shared policy
                action, log_prob, value = self.get_action(obs)

                # Take step in environment
                try:
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                except Exception as env_step_e:
                     print(f"!!! Error during env.step: {env_step_e}")
                     traceback.print_exc()
                     # Treat as episode end? Or skip step? Let's end episode.
                     terminated = True
                     truncated = False
                     reward = -5.0 # Penalize heavily for env errors
                     next_obs = obs # Keep current obs as next_obs

                done = terminated or truncated
                ep_steps += 1

                # Store transition in the buffer
                # The reward received is for the action taken by 'current_player'
                self.buffer.store(obs, action, log_prob, reward, value, done)

                # Accumulate reward for the player who just moved
                if current_player == 1:
                    episode_reward_p1 += reward
                else:
                    episode_reward_p2 += reward

                # Update observation
                obs = next_obs

                # If buffer is full, update policy
                if self.buffer.is_full():
                    avg_loss = self._update_policy()
                    if avg_loss is not None:
                         ep_losses.append(avg_loss)
                    # Buffer is cleared inside _update_policy

                # Safety break if steps exceed limit (should be handled by env termination)
                if ep_steps >= max_steps_per_episode * 1.1:
                     print(f"Warning: Episode {episode} exceeded step limit {max_steps_per_episode}. Breaking.")
                     if not done: # Force done if not already set
                          terminated = True
                          done = True
                     break


            # --- End of Episode ---
            episode_rewards_p1.append(episode_reward_p1)
            episode_rewards_p2.append(episode_reward_p2)
            avg_ep_loss = np.mean(ep_losses) if ep_losses else 0
            all_episode_losses.append(avg_ep_loss)
            final_loss = info.get('last_loss', float('inf')) # Get final loss from info

            # --- Track and Save Best Graph Structure ---
            if np.isfinite(final_loss) and final_loss < best_episode_final_loss:
                best_episode_final_loss = final_loss
                try:
                    # Serialize the graph structure from the environment
                    best_graph_structure = self.env.graph.serialize_graph()
                    print(f"*** New best graph found with final loss: {best_episode_final_loss:.4f} at episode {episode} ***")
                    # Save the best graph structure immediately (optional)
                    if graph_save_path:
                         graph_filepath = os.path.join(graph_save_path, "best_graph_structure.json")
                         try:
                              with open(graph_filepath, 'w') as f:
                                   json.dump(best_graph_structure, f, indent=4)
                              print(f"Best graph structure saved to {graph_filepath}")
                         except Exception as json_e:
                              print(f"Error saving best graph structure: {json_e}")

                except AttributeError:
                     print("Warning: Could not serialize graph. `env.graph.serialize_graph()` method not found?")
                except Exception as serialize_e:
                     print(f"Error during graph serialization: {serialize_e}")


            # Update learning rate if scheduling is enabled
            if self.lr_schedule:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']


            # Track and report progress
            total_episode_reward = episode_reward_p1 + episode_reward_p2
            avg_reward_p1 = np.mean(episode_rewards_p1[-100:]) if episode_rewards_p1 else 0
            avg_reward_p2 = np.mean(episode_rewards_p2[-100:]) if episode_rewards_p2 else 0
            avg_total_reward = avg_reward_p1 + avg_reward_p2

            print(f"Ep {episode}: Steps={ep_steps}, FinalLoss={final_loss:.4f}, Tot R={total_episode_reward:.2f} (P1:{episode_reward_p1:.2f}, P2:{episode_reward_p2:.2f}), "
                  f"Avg R={avg_total_reward:.2f} (P1:{avg_reward_p1:.2f}, P2:{avg_reward_p2:.2f}), "
                  f"Avg Update Loss={avg_ep_loss:.4f}, LR={current_lr:.6f}")

            # Save best model policy based on average total reward
            if save_path and avg_total_reward > best_avg_reward and episode > 50: # Start saving best after some initial exploration
                best_avg_reward = avg_total_reward
                self.save(os.path.join(save_path, f"selfplay_transformer_ppo_policy_best.pth"))
                print(f"*** New best policy saved with avg reward: {best_avg_reward:.2f} ***")

            # Save policy checkpoint periodically
            if save_path and (episode + 1) % 100 == 0:
                self.save(os.path.join(save_path, f"selfplay_transformer_ppo_policy_{episode+1}.pth"))

        print("Training finished.")

        # --- Save the final best graph structure found during training ---
        if graph_save_path and best_graph_structure is not None:
            final_graph_filepath = os.path.join(graph_save_path, "final_best_graph_structure.json")
            try:
                with open(final_graph_filepath, 'w') as f:
                    json.dump(best_graph_structure, f, indent=4)
                print(f"Final best graph structure saved to {final_graph_filepath} (Loss: {best_episode_final_loss:.4f})")
            except Exception as json_e:
                print(f"Error saving final best graph structure: {json_e}")
        elif best_graph_structure is None:
             print("No valid best graph structure found during training to save.")


        return episode_rewards_p1, episode_rewards_p2, all_episode_losses


    def _update_policy(self):
        """
        Update the policy using PPO algorithm with data from the buffer.
        """
        # Get all data from the buffer
        obs_list, actions_list, old_log_probs_np, rewards_np, values_np, dones_np = self.buffer.get_batch()

        # --- Prepare Data for PyTorch ---
        # Convert observations to tensor dictionary
        obs_tensor = self._obs_to_tensor(obs_list)

        # Convert actions to tensors
        actions_tensor = self._actions_to_tensors(actions_list)

        # Convert other data to tensors
        old_log_probs = torch.FloatTensor(old_log_probs_np)
        rewards = torch.FloatTensor(rewards_np)
        values = torch.FloatTensor(values_np)
        dones = torch.FloatTensor(dones_np) # Already float

        # --- Calculate Advantages and Returns ---
        advantages = self._compute_advantages(rewards_np, values_np, dones_np, self.gamma, lam=0.95)
        advantages = torch.FloatTensor(advantages)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Calculate returns (targets for value function)
        returns = advantages + values # values are V(s_t) from buffer

        # --- PPO Update Loop ---
        total_loss_epoch = 0
        num_updates = 0

        # Iterate multiple epochs over the collected data
        indices = np.arange(len(obs_list))
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(obs_list), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Slice data for the mini-batch
                batch_obs = {key: val[batch_indices] for key, val in obs_tensor.items()}
                batch_actions = {key: val[batch_indices] for key, val in actions_tensor.items()}
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # --- Forward pass for the current policy ---
                action_dists, current_values = self.policy.forward(batch_obs)

                # --- Calculate Log Probs and Entropy for the current policy ---
                current_log_probs = self._evaluate_actions(action_dists, batch_actions)
                entropy = self._compute_entropy(action_dists)

                # --- Calculate PPO Loss ---
                # Ratio of new probabilities to old probabilities
                ratio = torch.exp(current_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                obj = ratio * batch_advantages
                obj_clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean()

                # Value function loss (MSE)
                value_loss = F.mse_loss(current_values.squeeze(-1), batch_returns) # Ensure shapes match

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # --- Gradient Update ---
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss_epoch += loss.item()
                num_updates += 1

        # Buffer is cleared automatically after get_batch in this implementation
        avg_loss = total_loss_epoch / num_updates if num_updates > 0 else None
        return avg_loss


    def _compute_advantages(self, rewards, values, dones, gamma, lam):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        Inputs are numpy arrays. Returns numpy array.
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        last_value = 0.0 # Assume value of terminal state is 0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t] # 0 if done, 1 if not done
            # If trajectory didn't end, use V(s_{t+1}) from buffer, else use 0
            # GAE requires V(s_{t+1}), which is the value of the state *after* the action at time t
            # In our buffer, values[t] is V(s_t). So we need values[t+1] if t is not the last step.
            # If t is the last step, the next state is terminal, value is 0.
            next_state_value = values[t + 1] * mask if t < len(rewards) - 1 else last_value
            # Correction: next_value should be 0 if the state at t was terminal (dones[t]==1)
            # The mask already handles this for the *next* step's value contribution.
            # Let's recalculate delta carefully:
            delta = rewards[t] + gamma * (values[t+1] if t < len(rewards)-1 else last_value) * mask - values[t]

            advantages[t] = delta + gamma * lam * mask * last_advantage
            last_advantage = advantages[t]

        return advantages

    def _actions_to_tensors(self, actions_list):
        """
        Convert a list of action dictionaries (simplified) to tensors.
        """
        batch_size = len(actions_list)
        op_ids = torch.zeros(batch_size, dtype=torch.long)
        placements = torch.zeros(batch_size, dtype=torch.long)

        for i, action in enumerate(actions_list):
             # Handle potential None actions if buffer wasn't filled correctly
             if action is None:
                  print(f"Warning: Found None action at index {i} in actions_list.")
                  # Assign default values or skip? Let's assign defaults (e.g., 0)
                  op_ids[i] = 0
                  placements[i] = 0
                  continue
             try:
                  op_ids[i] = action['operation_id']
                  placements[i] = action['placement_strategy']
             except KeyError as e:
                  print(f"Error accessing action key at index {i}: {e}. Action: {action}")
                  # Assign defaults
                  op_ids[i] = 0
                  placements[i] = 0
             except TypeError as e:
                  print(f"Error processing action at index {i}: {e}. Action: {action}")
                  # Assign defaults
                  op_ids[i] = 0
                  placements[i] = 0


        return {
            'operation_id': op_ids,
            'placement_strategy': placements
        }

    def _evaluate_actions(self, action_dists, actions_tensor):
        """
        Compute log probabilities for given actions (simplified).
        action_dists: Output from policy network forward pass.
        actions_tensor: Output from _actions_to_tensors.
        """
        op_log_prob = action_dists['operation_id'].log_prob(actions_tensor['operation_id'])
        place_log_prob = action_dists['placement_strategy'].log_prob(actions_tensor['placement_strategy'])

        # Sum log probabilities for the combined action
        total_log_prob = op_log_prob + place_log_prob
        return total_log_prob

    def _compute_entropy(self, action_dists):
        """
        Compute entropy for action distributions (simplified).
        """
        op_entropy = action_dists['operation_id'].entropy()
        place_entropy = action_dists['placement_strategy'].entropy()

        # Return the mean entropy across the batch
        total_entropy = (op_entropy + place_entropy).mean()
        return total_entropy

    def save(self, path):
        """Save the policy model and optimizer state."""
        print(f"Saving policy model to {path}...")
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.lr_schedule else None
            }, path)
            print("Policy model saved.")
        except Exception as e:
             print(f"Error saving policy model: {e}")

    def load(self, path):
        """Load the policy model and optimizer state."""
        print(f"Loading policy model from {path}...")
        try:
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.lr_schedule and checkpoint.get('scheduler_state_dict') is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Scheduler state loaded.")
            else:
                 print("Scheduler state not loaded (not found or lr_schedule=False).")
            print("Policy model loaded.")
        except FileNotFoundError:
             print(f"Error: Model file not found at {path}")
        except Exception as e:
             print(f"Error loading policy model: {e}")


    def test(self, num_episodes=10, render=True):
        """Test the trained agent deterministically."""
        print("\n--- Testing Agent ---")
        episode_rewards = []
        final_losses = []
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            terminated, truncated = False, False
            steps = 0
            while not terminated and not truncated:
                # Use deterministic=True for testing
                action, _, _ = self.get_action(obs, deterministic=True)
                try:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                except Exception as e:
                     print(f"Error during test step: {e}")
                     terminated = True # End episode on error
                     reward = -5.0
                     info = {'last_loss': float('inf')}

                episode_reward += reward
                steps += 1
                if render:
                    print(f"Test Ep {episode}, Step {steps}: Action=Op{action['operation_id']}({self.env.operation_types[action['operation_id']]}) Pl{action['placement_strategy']}, "
                          f"Reward={reward:.3f}, Loss={info.get('last_loss', 'N/A'):.4f}")
                    self.env.render()
                if steps >= self.env.max_steps * 2: # Safety break
                     print("Warning: Exceeded max test steps.")
                     break

            episode_rewards.append(episode_reward)
            final_losses.append(info.get('last_loss', float('inf')))
            print(f"Test Episode {episode} finished. Total Reward: {episode_reward:.2f}, Final Loss: {info.get('last_loss', 'N/A'):.4f}")

        avg_reward = np.mean(episode_rewards)
        avg_loss = np.mean([l for l in final_losses if np.isfinite(l)]) if any(np.isfinite(l) for l in final_losses) else float('inf')
        print(f"\nAverage Test Reward over {num_episodes} episodes: {avg_reward:.2f}")
        print(f"Average Final Loss over {num_episodes} episodes: {avg_loss:.4f}")
        return episode_rewards


# --- Example Usage ---
if __name__ == "__main__":
    # Import the environment
    from math_env import MathSelfPlayEnv # Use the correct env name

    # --- Configuration ---
    # Using defaults from math_env.py for grid_size (10) and max_steps (50)
    # GRID_SIZE = 10 # Or override default here
    # MAX_STEPS = 50 # Or override default here

    # --- Sequence-to-Sequence Specific ---
    FEATURE_DIM = 8     # Feature dimension for sequence embeddings (must match MathNode default or env init)
    SEQ_LEN = 15        # Sequence length for the task
    ENV_BATCH_SIZE = 64 # Batch size for sequence evaluation within env

    # --- PPO Hyperparameters ---
    BUFFER_SIZE = 2048
    PPO_BATCH_SIZE = 64
    UPDATE_EPOCHS = 10
    NUM_EPISODES = 1000 # Number of training episodes
    LR = 5e-4 # Learning rate (might need tuning for seq2seq)
    POLICY_SAVE_PATH = "ppo_seq2seq_policy_checkpoints" # Directory to save policy models
    GRAPH_SAVE_PATH = "ppo_seq2seq_graphs" # Directory to save best graph structure

    # Create save directories if they don't exist
    for path in [POLICY_SAVE_PATH, GRAPH_SAVE_PATH]:
        if path and not os.path.exists(path):
            os.makedirs(path)

    print("Initializing Environment for Sequence-to-Sequence Task...")
    try:
        env = MathSelfPlayEnv(
            # grid_size=GRID_SIZE, # Omit to use default
            # max_steps=MAX_STEPS, # Omit to use default
            feature_dim=FEATURE_DIM, # Pass sequence feature dim
            batch_size=ENV_BATCH_SIZE,
            sequence_length=SEQ_LEN
            # dataset_path is no longer needed
        )
        print(f"Environment Initialized.")
        print(f"Grid Size: {env.grid_size}, Max Steps: {env.max_steps}")
        print(f"Number of operations: {env.num_operations}")
        print(f"Sequence Feature Dim: {env.feature_dim}")
        print(f"Sequence Length: {env.sequence_length}")

    except Exception as env_init_e:
         print(f"!!! Error Initializing Environment: {env_init_e}")
         traceback.print_exc()
         exit()


    # --- Define State and Action Dimensions for the Agent ---
    # Get shapes directly from the environment's spaces
    state_dim = {
        'board': env.observation_space['board'].shape
    }
    action_dims = {
        'operation_id': env.action_space['operation_id'].n,
        'placement_strategy': env.action_space['placement_strategy'].n
    }
    print(f"State Dim (Board): {state_dim['board']}")
    print(f"Action Dims: {action_dims}")

    # --- Create and Train Agent ---
    agent = SelfPlayPPOAgent(
        env=env,
        state_dim=state_dim,
        action_dims=action_dims,
        lr=LR,
        buffer_size=BUFFER_SIZE,
        batch_size=PPO_BATCH_SIZE,
        update_epochs=UPDATE_EPOCHS,
        lr_schedule=True
    )

    print("Self-Play PPO Agent created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")

    # --- Training ---
    agent.train(
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=env.max_steps, # Use env's max_steps
        save_path=POLICY_SAVE_PATH,
        graph_save_path=GRAPH_SAVE_PATH # Pass graph save path
    )

    # --- Testing ---
    # Load the best policy model for testing
    best_policy_path = os.path.join(POLICY_SAVE_PATH, "selfplay_transformer_ppo_policy_best.pth")
    if os.path.exists(best_policy_path):
        agent.load(best_policy_path)
        agent.test(num_episodes=10, render=True) # Render can be useful for seq2seq grid
    else:
        print("Best policy model not found, testing with the final model.")
        agent.test(num_episodes=10, render=True)

    env.close()
