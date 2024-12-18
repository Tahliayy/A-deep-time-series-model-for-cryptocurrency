import torch
import torch.nn as nn
import torch.nn.functional as F

from tmt_config import params
from model.tmt_model_util import GatedResidualNetwork


# Category Calculation
class CategoryEmbedding(nn.Module):
    def __init__(self,
                 dim: int,
                 hidden_layer_size=params.model_params.embedding_hidden_layer_size):
        super(CategoryEmbedding, self).__init__()
        self.embedding = nn.Embedding(dim, hidden_layer_size)
        self.hidden_layer_size = hidden_layer_size

    def forward(self, x):
        batch_size, sequence_length, num_categorical_variables = x.shape    # [19, 60, 1]
        assert x.max() < self.embedding.num_embeddings, "Index out of range in the embedding layer"

        # 对于每个 categorical_input 变量，获取其嵌入
        embedded_inputs = [
            self.embedding(x[..., i].long())  # 需要转换为 long 类型以适配 nn.Embedding
            for i in range(num_categorical_variables)
        ]  # embedded_inputs 是一个 list, 每个元素形状为 [batch_size, sequence_length, hidden_layer_size]

        # 对于每个 categorical_input 的第一个时间步的嵌入特征
        static_inputs = torch.stack(
            [embedded_inputs[i][:, 0, :] for i in range(num_categorical_variables)],
            dim=1   # 堆叠到维度 1
        )           # static_inputs 形状为 [batch_size, num_categorical_variables, hidden_layer_size]
        # print('static_inputs shape: ', static_inputs.shape)
        return static_inputs    # 这里是: [19, 1, 10]


class StaticEncoder(nn.Module):
    def __init__(self,
                 input_size: int = params.model_params.hidden_layer_size,
                 hidden_layer_size: int = params.model_params.hidden_layer_size,
                 dropout_rate: float = params.model_params.dropout_rate):
        """
        Static Encoder for processing static inputs.
        """
        super(StaticEncoder, self).__init__()

        # GatedResidualNetwork for static embedding transformation
        self.gated_residual_network = GatedResidualNetwork(
            input_size=input_size,
            hidden_layer_size=hidden_layer_size,
            output_size=1,  # As the output_size in the original code is num_static=1
            dropout_rate=dropout_rate,
            return_gate=False
        )
        self.gated_residual_network_trans = GatedResidualNetwork(
            input_size=hidden_layer_size,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            return_gate=False
        )

        self.softmax = nn.Softmax(dim=-1)  # Softmax for sparse weights

    def forward(self, static_inputs):
        """
        Args:
            embedding: Transformed static inputs, shape [batch_size, num_static, static_dim]

        Returns:
            static_vec: Combined static embeddings, shape [batch_size, static_dim]
            sparse_weights: Attention weights, shape [batch_size, num_static, 1]
            :param static_inputs:
        """
        batch_size, num_static, static_dim = static_inputs.size()   # 19, 1, 10

        # Flatten the static inputs (embedding) to [batch_size, num_static * static_dim]
        flatten = static_inputs.view(batch_size, num_static * static_dim)
        # print('flatten.shape: ', flatten.shape)

        # Apply the gated residual network to the flattened inputs
        mlp_outputs, _ = self.gated_residual_network(flatten)  # Output shape [batch_size, num_static]
        # print('mlp_outputs shape: ', mlp_outputs.shape)

        # Apply softmax to get the sparse weights
        sparse_weights = self.softmax(mlp_outputs)  # Shape [batch_size, num_static]
        # print('sparse_weights shape: ', sparse_weights.shape)

        # Unsqueeze to add an extra dimension at the end, making it [batch_size, num_static, 1]
        sparse_weights = sparse_weights.unsqueeze(-1)
        # print('sparse_weights2 shape: ', sparse_weights.shape)

        # List to store transformed embeddings
        trans_emb_list = []
        for i in range(num_static):
            e, _ = self.gated_residual_network_trans(static_inputs[:, i:i + 1, :])  # Shape [batch_size, 1, static_dim]
            trans_emb_list.append(e)
        # print('trans_emb_list.shape: ', trans_emb_list[0].shape, '   trans_emb_list.len: ', len(trans_emb_list))

        # Concatenate along the second dimension (if there is more than one static variable)
        if len(trans_emb_list) > 1:
            transformed_embedding = torch.cat(trans_emb_list, dim=1)  # Shape [batch_size, num_static, static_dim]
        else:
            transformed_embedding = trans_emb_list[0]  # [batch_size, 1, static_dim]

        # Element-wise multiplication of sparse weights and transformed embedding
        # print('before combine, sparse_weights: ', sparse_weights.shape,
        #       ' transformed_embedding: ', transformed_embedding.shape)
        combined = sparse_weights * transformed_embedding  # Shape [batch_size, num_static, static_dim]
        # print('combined shape: ', combined.shape)

        # Sum along the num_static dimension to get the final static vector
        static_vec = torch.sum(combined, dim=1)  # Shape [batch_size, static_dim]
        # print('static_vec: ', static_vec.shape)

        return static_vec, sparse_weights


# Feature Calculation
class FeatureEmbedding(nn.Module):
    def __init__(self,
                 feat_dim: int,  # 输入维度（feature维度大小）
                 hidden_layer_size: int = params.model_params.embedding_hidden_layer_size):  # 输出的 embedding 维度
        super(FeatureEmbedding, self).__init__()

        # 定义线性层，类似于 keras.layers.Dense(self.hidden_layer_size)
        self.linear = nn.Linear(1, hidden_layer_size)  # 输入是单个 feature，输出是 hidden size
        self.hidden_layer_size = hidden_layer_size
        self.feat_dim = feat_dim

    def convert_real_to_embedding(self, x):
        """
        Applies a linear transformation for time-varying inputs.
        Equivalent to keras.layers.TimeDistributed(Dense(hidden_layer_size)).
        """
        # x shape: [batch_size, seq_len, 1] -> [batch_size, seq_len, hidden_layer_size]
        # print('FeatureEmbedding convert_real_to_embedding x.shape: ', x.shape)
        batch_size, seq_len, _ = x.shape
        # print('FeatureEmbedding convert_real_to_embedding batch_size, seq_len: ', batch_size, ' ', seq_len)
        # print('x.shape: ', x.shape)
        x = x.contiguous().view(-1, 1)                                # Flatten batch and seq_len dimensions for Linear
        # print('FeatureEmbedding x.shape: ', x.shape)
        embedded_x = self.linear(x)                                   # Apply linear transformation
        # print('FeatureEmbedding embedded_x.shape: ', embedded_x.shape)
        output = embedded_x.view(batch_size, seq_len, self.hidden_layer_size)   # Reshape back
        # print('FeatureEmbedding output.shape: ', output.shape)
        return output

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, feat_dim]. It's "features" actually

        Returns:
            known_combined_layer: Tensor of shape [batch_size, seq_len, hidden_layer_size, feat_dim].
        """
        # print('FeatureEmbedding forward x.shape:', x.shape)
        # 对 regular_inputs 的每个特征进行 convert_real_to_embedding 操作
        known_regular_inputs = [
            self.convert_real_to_embedding(x[:, :, i:i + 1])
            for i in range(self.feat_dim)  # 遍历最后一维的所有特征
        ]
        # print('known_regular_inputs.shape: ', len(known_regular_inputs))

        # 拼接所有处理后的输入，输出形状为 [batch_size, seq_len, hidden_layer_size, feat_dim]
        # 这里是 [19, 60, 10, 42]
        known_combined_layer = torch.stack(known_regular_inputs, dim=-1)
        # print('known_combined_layer.shape: ', known_combined_layer.shape)

        return known_combined_layer


class FeatureVariableSelection(nn.Module):
    def __init__(self,
                 feat_dim: int = params.model_params.num_features,
                 hidden_layer_size: int = params.model_params.hidden_layer_size,
                 dropout_rate: float = params.model_params.dropout_rate):
        super(FeatureVariableSelection, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        # Define GatedResidualNetwork for static context processing
        self.static_variable_selection_network = GatedResidualNetwork(
            input_size=params.model_params.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate
        )

        # variable selection weights: step 2
        self.variable_selection_weights_gru = GatedResidualNetwork(
            input_size=params.model_params.hidden_layer_size * params.model_params.num_features,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            output_size=feat_dim
        )

        # feature_transform_network, do the transform per feature element: step 3
        self.feature_transform_networks = nn.ModuleList([
            GatedResidualNetwork(
                input_size=params.model_params.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate
            ) for _ in range(feat_dim)
        ])

    def forward(self, feat_embedding, static_encoder=None):
        # feat_embedding: [19, 60, 10, 42], static_encoder: None or [19, 10]
        """
        Args:
            feat_embedding: Transformed inputs [batch_size, seq_len, feat_dim, num_of_feat]
            static_encoder: Static inputs for variable selection, if any.

        Returns:
            temporal_ctx: Processed temporal context [batch_size, seq_len, feat_dim]
            sparse_weights: Sparse weights [batch_size, seq_len, 1, num_inputs]
            :param feat_embedding:
            :param static_encoder:
        """
        # 19,       60,      10,            42
        batch_size, seq_len, embedding_dim, num_feat = feat_embedding.size()
        # print(' in feature selection')

        # Static context processing (if static_encoder is provided)
        if static_encoder is not None:      # static_encoder: [batch_size, hidden_dim] -> [batch_size, hidden_dim]
            # Apply static context processing network to static_encoder, [19, 10] -> [19, 10]
            # print(' static_encoder.shape: ', static_encoder.shape)
            static_context_variable_selection, _ = self.static_variable_selection_network(static_encoder)
            # print(' static_context_variable_selection.shape: ', static_context_variable_selection.shape)

            # Expand static context [batch_size, 1, hidden_layer_size]
            expanded_static_context = static_context_variable_selection.unsqueeze(1)    # [19, 1, 10]
            # print(' expanded_static_context.shape: ', expanded_static_context.shape)
        else:
            expanded_static_context = None

        # Transformed feature dimension selection:
        # 1. flatten expanded feature: [bs, seq_len, hidden_dim, feat_dim] -> [bs, seq_len, hidden_dim * feat_dim]
        #                              [19, 60, 10, 42]                    -> [19, 60, 10 * 42]
        # 2. get weight for each transformed feature dimension/element:
        #    [bs, seq_len, hidden_dim * feat_dim] -> [bs, seq_len, feat_dim] -> [bs, seq_len, 1, feat_dim]
        #    [19, 60, 10 * 42] -> [19, 60, 42] -> [19, 60, 1, 42]
        # 3. get transformed feature
        #    [the feature has been transformed before actually at outer space, at FeatureEmbedding to be frank.]
        #    feat_dim * [bs, seq_len, hidden_dim] ->
        #    feat_dim * [bs, seq_len, hidden_dim] -> [bs, seq_len, hidden_dim, feat_dim]
        #    42 * [19, 60, 10] -> 42 * [19, 60, 10] -> [19, 60, 10, 42]
        # 4. get weighted feature elements, at transformed space: weights * transformed feature
        #    [bs, seq_len, 1, feat_dim] * [bs, seq_len, hidden_dim, feat_dim]
        #    [19, 60, 1, 42] * [19, 60, 10, 42] = [19, 60, 10, 42] -> [19, 60, 10]
        #
        # step 1. flatten embedding to [batch_size, seq_len, hidden_dim * feat_dim]  # [19, 60, 10, 42] ->
        flatten = feat_embedding.view(batch_size, seq_len, embedding_dim * num_feat)    # [19, 60, 10 * 42]
        # print(' flatten.shape: ', flatten.shape, '   expanded_static_context: ', expanded_static_context.shape)

        # step 2. get transformed feature dimensions' weights:
        feature_mlp_outputs, _ = self.variable_selection_weights_gru(flatten,
                                                                     additional_context=expanded_static_context)
        # print(' feature_mlp_outputs.shape: ', feature_mlp_outputs.shape)
        sparse_weights = F.softmax(feature_mlp_outputs, dim=-1)         # [batch_size, seq_len, feat_dim]
        # [batch_size, seq_len, feat_dim] -> [batch_size, seq_len, 1, feat_dim], [19, 60, 42] -> [19, 60, 1, 42]
        sparse_weights = sparse_weights.unsqueeze(-2)

        # step 3. get transformed feature
        trans_emb_list = []
        for i in range(num_feat):
            grn_output, _ = self.feature_transform_networks[i](
                feat_embedding[..., i]                                  # [batch_size, seq_len, hidden_dim]
            )
            # feat_dim * [batch_size, seq_len, embedding_dim],
            trans_emb_list.append(grn_output)                           # 42 * [19, 60, 10]

        # Stack along the last dimension to get [batch_size, seq_len, embedding_dim, feat_dim]
        transformed_embedding = torch.stack(trans_emb_list, dim=-1)     # [19, 60, 10, 42]

        # step 4. get weighted feature elements
        # Element-wise multiplication of sparse weights and transformed embedding
        #           [batch_size, seq_len, hidden_dim, num_inputs],        [19, 60, 10, 42]
        combined = sparse_weights * transformed_embedding

        # Sum along the last dimension to reduce to [batch_size, seq_len, embedding_dim]
        feature_selected_weighted = torch.sum(combined, dim=-1)         # [19, 60, 10]

        return feature_selected_weighted, sparse_weights


