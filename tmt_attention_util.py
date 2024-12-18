import torch
import torch.nn as nn


def get_decoder_mask(seq_len, batch_size, device=None):
    """
    Returns causal mask for self-attention.
    Args:
        seq_len: Sequence length
        batch_size: Batch size
    Returns:
        Causal mask of shape (batch_size, seq_len, seq_len) where each matrix is lower triangular
    """
    # Create a lower triangular matrix with ones on the diagonal and below, zeros elsewhere
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
    if device is not None:
        mask = mask.to(device)
    return mask  # Shape: (batch_size, seq_len, seq_len)


# Scaled Dot Product Attention in PyTorch
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        if attn_dropout is not None:
            self.dropout = nn.Dropout(attn_dropout)
        else:
            self.dropout = None
        self.softmax = nn.Softmax(dim=-1)  # Softmax along the last dimension

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Queries (batch_size, seq_len, feat_dim)
            k: Keys (batch_size, seq_len, feat_dim)
            v: Values (batch_size, seq_len, feat_dim)
            mask: Optional mask (batch_size, seq_len, seq_len)
        Returns:
            Tuple of (output, attention weights)
        """
        # Calculate the dot product attention scores and scale by sqrt(d_k)
        d_k = q.size(-1)  # Get the dimensionality of the queries/keys
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32))  # (batch_size, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)  # Masking attention scores

        # Apply softmax to get the attention weights
        attn = self.softmax(attn)  # (batch_size, seq_len, seq_len)

        # Apply dropout if dropout is defined
        if self.dropout is not None:
            attn = self.dropout(attn)

        # Compute the final output as a weighted sum of values
        output = torch.matmul(attn, v)  # (batch_size, seq_len, feat_dim)

        return output, attn


# Interpretable Multi-Head Attention in PyTorch
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1, attn_dropout=None):
        """
        Initializes the interpretable multi-head attention layer.
        Args:
            n_head: Number of attention heads
            d_model: Model dimensionality (input feature size)
            dropout: Dropout rate
        """
        super(InterpretableMultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = self.d_v = d_model // n_head  # Dimension per head
        self.dropout = dropout

        # Query, Key, Value layers for each head
        vs_layer = nn.Linear(d_model, self.d_v, bias=False)
        self.qs_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.ks_layers = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_head)])
        self.vs_layers = nn.ModuleList([vs_layer for _ in range(n_head)])

        # Attention mechanism
        self.attention = ScaledDotProductAttention(attn_dropout=attn_dropout)

        # Output projection
        self.w_o = nn.Linear(self.d_v, d_model, bias=False)

        # Dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for multi-head attention.
        Args:
            q: Query tensor (batch_size, seq_len, d_model)
            k: Key tensor (batch_size, seq_len, d_model)
            v: Value tensor (batch_size, seq_len, d_model)
            mask: Masking tensor (optional)
        Returns:
            Tuple of (output, attention weights)
        """
        heads = []
        attns = []
        # print('     InterpretableMultiHeadAttention:')
        for i in range(self.n_head):
            qs = self.qs_layers[i](q)  # (batch_size, seq_len, d_k)
            ks = self.ks_layers[i](k)  # (batch_size, seq_len, d_k)
            vs = self.vs_layers[i](v)  # (batch_size, seq_len, d_v)

            # Apply scaled dot-product attention
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(self.output_dropout(head))
            attns.append(attn)
        # print('         len attns: ', len(attns), '   attns[0]', attns[0].shape, ' heads[0]: ', heads[0].shape)
        # # Concatenate heads (if n_head > 1)
        # head = torch.cat(heads, dim=-1) if self.n_head > 1 else heads[0]
        # attn = torch.stack(attns)
        head = torch.stack(heads, dim=0) if self.n_head > 1 else heads[0]  # Shape: [n_head, batch_size, seq_len, d_k]
        # print('         head: ', head.shape)

        attn = torch.stack(attns, dim=0)  # Shape: [n_head, batch_size, seq_len, seq_len]
        # print('         attn: ', attn.shape)

        # Project back to the original d_model dimension
        output = torch.mean(head, dim=0) if self.n_head > 1 else head  # Shape: [batch_size, seq_len, d_k]
        # print('         output1: ', output.shape)
        output = self.w_o(output)
        # print('         output2: ', output.shape)
        output = self.output_dropout(output)
        # print('         output3: ', output.shape)

        return output, attn


# This is original multi-head attention, used for comparing with the modified one
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initializes the MultiHeadAttention layer using PyTorch's built-in attention mechanism.
        Args:
            embed_dim: Model dimensionality (input feature size)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=num_heads,
                                                    dropout=dropout,
                                                    batch_first=True)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for the MultiHeadAttention layer.
        Args:
            q: Query tensor (batch_size, seq_len, feat_dim)
            k: Key tensor (batch_size, seq_len, feat_dim)
            v: Value tensor (batch_size, seq_len, feat_dim)
            mask: Masking tensor (optional) (batch_size, seq_len, seq_len)
        Returns:
            Tuple of:
                output: Attention output of shape (batch_size, seq_len, feat_dim)
                attn_weights: Attention weights of shape (n_head, batch_size, seq_len, seq_len)
        """
        # Prepare the attention mask if provided
        if mask is not None:
            # MultiheadAttention expects the mask to have shape (batch_size * num_heads, seq_len, seq_len)
            attn_mask = mask
        else:
            attn_mask = None

        # Use PyTorch's native MultiheadAttention
        output, attn_weights = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # PyTorch's attention weights have shape (batch_size, num_heads, seq_len, seq_len)
        # We need to return attention weights in the format (n_head, batch_size, seq_len, seq_len)
        attn_weights = attn_weights.permute(1, 0, 2, 3)  # Convert to (num_heads, batch_size, seq_len, seq_len)

        return output, attn_weights




