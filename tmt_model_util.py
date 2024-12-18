import torch
import torch.nn as nn


class AddAndNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(AddAndNorm, self).__init__()
        # Initialize layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x_list):
        """
        Args:
            x_list: List of tensors to sum (for skip connection)

        Returns:
            Output tensor after applying skip connection and layer normalization
        """
        # Apply skip connection (element-wise sum)
        tmp = torch.stack(x_list, dim=0).sum(dim=0)

        # Apply layer normalization
        tmp = self.layer_norm(tmp)

        return tmp


class GLU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 dropout_rate: float = None,
                 activation=None):
        """
        Applies a Gated Linear Unit (GLU) to an input.
        Args:
            hidden_layer_size: Dimension of GLU
            dropout_rate: Dropout rate to apply if any
            activation: Activation function to apply to the linear feature transform if necessary
        """
        super(GLU, self).__init__()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None
        self.activation_layer = nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = nn.Linear(input_size, hidden_layer_size)
        self.activation_fn = activation if activation is not None else nn.Identity()  # Default to no activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('     in glu x1.shape:', x.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        # print('     in glu x2.shape:', x.shape)
        # Apply the linear transformations and activation functions
        tmp1 = self.activation_layer(x)
        # print('     in glu tmp shape: ', tmp1.shape)
        activation_layer = self.activation_fn(tmp1)
        # print('     in glu activation_layer shape: ', activation_layer.shape)
        tmp2 = self.gated_layer(x)
        # print('     in glu tmp2 shape: ', tmp2.shape)
        gated_layer = self.sigmoid(tmp2)
        # print('     in glu gated_layer: ', gated_layer.shape)

        # Element-wise multiplication of activation and gated layer
        output = activation_layer * gated_layer
        # print('     in glu output shape: ', output.shape)

        return output, gated_layer


class GatedResidualNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int = None,
                 dropout_rate: float = None,
                 additional_context=None,
                 return_gate: bool = False,
                 activation=None):
        super(GatedResidualNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size if output_size is not None else hidden_layer_size
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context
        self.return_gate = return_gate

        # Skip connection layer (if output_size is provided)
        self.linear_skip = nn.Linear(input_size, self.output_size) if output_size is not None else nn.Identity()

        # First feedforward linear layer
        self.linear_hidden1 = nn.Linear(input_size, hidden_layer_size)

        # Optional context layer for additional context
        self.context_layer = nn.Linear(hidden_layer_size, hidden_layer_size, bias=False)

        # Second feedforward linear layer
        self.linear_hidden2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Gated Linear Unit (GLU)
        self.glu = GLU(input_size=hidden_layer_size,
                       hidden_layer_size=self.output_size,
                       dropout_rate=dropout_rate,
                       activation=activation)

        # Add and Layer Normalization
        self.add_and_norm = AddAndNorm(normalized_shape=self.output_size)

        # Activation function (default to ELU)
        self.activation_fn = activation if activation is not None else nn.ELU()

        self.batch_size, self.seq_len = None, None

    def forward(self, x, additional_context=None):
        """
        Args:
            x: Input tensor (shape can be 2D or 3D)
            additional_context: Optional additional context tensor (shape [batch_size, 1, hidden_layer_size])

        Returns:
            Tuple of (output tensor, gate tensor)
        """
        is_3d = (x.dim() == 3)
        # print('     in gru is_3d: ', is_3d, '   x.dim: ', x.dim())
        # print('     in gru x.shape: ', x.shape)

        if is_3d:
            batch_size, seq_len, feat_dim = x.shape
            self.batch_size, self.seq_len = batch_size, seq_len
            x = x.view(batch_size * seq_len, feat_dim)      # Flatten for linear layer
            # print('     in gru is 3d x.shape: ', x.shape)

        # Apply skip connection (or identity if output_size is None)
        skip = self.linear_skip(x)                          # (None, output_size)   # [19, 1]
        # print('     in gru skip shape: ', skip.shape)

        # First feedforward pass
        hidden = self.linear_hidden1(x)                     # (None, hidden_layer_size)
        # print('     in gru hidden shape1: ', hidden.shape)

        # If additional context is provided, adjust its shape and add to hidden
        if additional_context is not None:
            if is_3d:
                # print('     in gru is context layer none?: ', self.context_layer is None)
                # If input was 3D, we need to repeat and reshape the context
                additional_context = additional_context.repeat(1, self.seq_len, 1)
                additional_context = additional_context.view(self.batch_size * self.seq_len, -1)
                # print('     in gru additional_context.shape: ', additional_context.shape)

            hidden = hidden + self.context_layer(additional_context)

        # print('     in gru hidden shape2: ', hidden.shape)
        # Apply activation
        hidden = self.activation_fn(hidden)
        # print('     in gru hidden shape3: ', hidden.shape)
        # Second feedforward pass
        hidden = self.linear_hidden2(hidden)                # (None, hidden_layer_size)
        # print('     in gru hidden shape4: ', hidden.shape)
        # Apply GLU
        gating_layer, gate = self.glu(hidden)               # (None, output_size)
        # print('     in gru gating_layer shape: ', gating_layer.shape)
        # Apply Add and Layer Normalization
        output = self.add_and_norm([skip, gating_layer])    # (None, output_size)
        # print('     in gru skip shape: ', skip.shape)
        # print('     in gru gru output shape: ', output.shape)

        if is_3d:
            output = output.view(self.batch_size, self.seq_len, self.output_size)
            # output = output.view(self.batch_size, self.seq_len, -1)
            # print('     in gru 3d output: ', output.shape)
            gate = gate.view(self.batch_size, self.seq_len, -1)
            # print('     in gru 3d gate: ', gate.shape)

        return output, gate


class LstmEncoder(nn.Module):
    def __init__(self, hidden_layer_size, dropout=0):
        super(LstmEncoder, self).__init__()
        self.hidden_layer_size = hidden_layer_size

        # Initialize LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.hidden_layer_size,      # 对应embedding_dim
            hidden_size=self.hidden_layer_size,     # 对应hidden_layer_size
            num_layers=1,                           # 一层LSTM
            batch_first=True,                       # batch维度在第一维
            bias=True,                              # 使用bias
            dropout=dropout,
        )

    def forward(self, input_embeddings, state_h, state_c):
        """
        Args:
            input_embeddings: Input embeddings of shape [batch_size, seq_len, embedding_dim]
            state_h: Initial hidden state of shape [batch_size, hidden_layer_size]
            state_c: Initial cell state of shape [batch_size, hidden_layer_size]

        Returns:
            lstm_output: Output from LSTM, of shape [batch_size, seq_len, hidden_layer_size]
        """
        # Reshape initial hidden and cell states to match LSTM's expected input
        initial_state = (
            state_h.unsqueeze(0), state_c.unsqueeze(0)
        )   # [1, batch_size, hidden_layer_size]

        # Pass input_embeddings and initial states to the LSTM
        lstm_output, _ = self.lstm(input_embeddings, initial_state)

        return lstm_output

