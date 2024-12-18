import torch
import torch.nn as nn

from tmt_config import params
from model.tmt_model_util import GatedResidualNetwork, LstmEncoder, GLU, AddAndNorm
from model.tmt_encoders import CategoryEmbedding, FeatureEmbedding, StaticEncoder, FeatureVariableSelection
from model.tmt_attention_util import get_decoder_mask, InterpretableMultiHeadAttention


class TMT(nn.Module):
    def __init__(self, device):
        super(TMT, self).__init__()
        self.device = device

        self.src_mask = None

        ''' This is the first block: static variable selection '''
        # input embedding: category (static_inputs)
        self.category_tft_embedding = CategoryEmbedding(dim=params.model_params.category_cnt)

        # static encoder: corresponding to "static_combine_and_mask" in original implementation
        # static variable selection + static encoder
        self.static_encoder_network = StaticEncoder()

        ''' This is the second block: feature variable selection '''
        # input embedding: features
        self.regular_tft_embedding = FeatureEmbedding(feat_dim=params.model_params.num_features)
        # feature variable selection network
        self.feature_selection_network = FeatureVariableSelection()

        ''' This is the third block: LSTM encoder '''
        # lstm encoder
        self.static_state_h_network = GatedResidualNetwork(input_size=params.model_params.hidden_layer_size,
                                                           hidden_layer_size=params.model_params.hidden_layer_size,
                                                           dropout_rate=params.model_params.dropout_rate)
        self.static_state_c_network = GatedResidualNetwork(input_size=params.model_params.hidden_layer_size,
                                                           hidden_layer_size=params.model_params.hidden_layer_size,
                                                           dropout_rate=params.model_params.dropout_rate)
        self.lstm_encoders = LstmEncoder(hidden_layer_size=params.model_params.hidden_layer_size)

        ''' This is the fourth block: GLU + add & norm appending LSTM '''
        # lstm GLU
        self.lstm_glu = GLU(input_size=params.model_params.hidden_layer_size,
                            hidden_layer_size=params.model_params.hidden_layer_size,
                            dropout_rate=params.model_params.dropout_rate)
        # lstm add_or_norm
        self.lstm_add_or_norm = AddAndNorm(normalized_shape=params.model_params.hidden_layer_size)

        ''' This is the 5th block: GRN appending 4th block '''
        # static_feature_GRN, the result is called "enriched"
        self.static_context_enrichment = GatedResidualNetwork(input_size=params.model_params.hidden_layer_size,
                                                              hidden_layer_size=params.model_params.hidden_layer_size,
                                                              dropout_rate=params.model_params.dropout_rate)
        self.static_feature_GRN = GatedResidualNetwork(input_size=params.model_params.hidden_layer_size,
                                                       hidden_layer_size=params.model_params.hidden_layer_size,
                                                       dropout_rate=params.model_params.dropout_rate)

        ''' This is the 6th block: InterpretableMultiHeadAttention '''
        self.mask_train = get_decoder_mask(seq_len=params.seq_len_per_batch_train,
                                           batch_size=params.batch_size,
                                           device=self.device)
        self.mask_pred = get_decoder_mask(seq_len=params.seq_len_per_batch_predict,
                                          batch_size=params.batch_size,
                                          device=self.device)
        self.self_attn_layer = InterpretableMultiHeadAttention(n_head=params.model_params.num_heads,
                                                               d_model=params.model_params.hidden_layer_size,
                                                               dropout=params.model_params.dropout_rate)

        ''' 7th block: 2nd GLU + add_and_norm (GLU(x) + AddAndNorm(GLU(x) + enriched) '''
        self.attn_glu = GLU(input_size=params.model_params.hidden_layer_size,
                            hidden_layer_size=params.model_params.hidden_layer_size,
                            dropout_rate=params.model_params.dropout_rate)
        self.attn_add_or_norm = AddAndNorm(normalized_shape=params.model_params.hidden_layer_size)

        ''' 8th block: GRN for decoder '''
        self.decoder = GatedResidualNetwork(input_size=params.model_params.hidden_layer_size,
                                            hidden_layer_size=params.model_params.hidden_layer_size,
                                            dropout_rate=params.model_params.dropout_rate)

        ''' 9th: GLu + add_and_norm (GLU(decoder) + temporal_feature [, result of "lstm_add_or_norm"] '''
        self.decoder_glu = GLU(input_size=params.model_params.hidden_layer_size,
                               hidden_layer_size=params.model_params.hidden_layer_size)       # no dropout
        self.decoder_add_or_norm = AddAndNorm(normalized_shape=params.model_params.hidden_layer_size)

        ''' 10th: output - linear operation '''
        self.output_layer = nn.Linear(in_features=params.model_params.hidden_layer_size,
                                      out_features=1)

    def forward(self, feat, ids, phase):
        # 60: empirical sequence length
        # 19: batch size (all tickers except for 'AVAX' since it's lack of 2 months data)
        # 42: 42 features calculated by the team
        # 1: 1 column which indicates which ticker we are dealing with
        # 10: hidden layer size / hidden dim, the dimension used in the network
        #     plz remember, 10 is just a number for instance. one can change to any number who likes
        #
        # feat: [19, 60, 42]
        # ids: [19, 60, 1]

        # Embedded each category ID
        if ids is None:
            id_embedded, static_encoder = None, None
        else:
            id_embedded = self.category_tft_embedding(ids)                  # [19, 60, 1] -> [19, 1, 10]
            # print('id_embedded shape: ', id_embedded.shape)
            # static encoder
            static_encoder, _ = self.static_encoder_network(id_embedded)       # [19, 1, 10] -> [19, 10]
        # print('static_encoder: ', static_encoder.shape)
        # Embedded features for each feature elements
        # print('before feat_embedded, feat.shape: ', feat.shape)
        feat_embedded = self.regular_tft_embedding(feat)                    # [19, 60, 42] -> [19, 60, 10, 42]
        # print('after feat embedded: ', feat_embedded.shape)
        # feature variable selection:
        feature_hidden_states, _ = \
            self.feature_selection_network(feat_embedded, static_encoder)   # [19, 60, 10, 42], [19, 10] -> [19, 60, 10]
        # print('feature_hidden_states: ', feature_hidden_states.shape)

        # LSTM encoder
        static_context_state_h, _ = self.static_state_h_network(static_encoder)
        # print('static_context_state_h: ', static_context_state_h.shape)
        static_context_state_c, _ = self.static_state_c_network(static_encoder)
        # print('static_context_state_c: ', static_context_state_c.shape)
        lstm_states = self.lstm_encoders(feature_hidden_states,
                                         static_context_state_h,
                                         static_context_state_c)
        # print('lstm_states: ', lstm_states.shape)

        # LSTM glu
        lstm_states, _ = self.lstm_glu(lstm_states)
        # print('lstm_states2: ', lstm_states.shape)
        # LSTM add_and_norm
        temporal_feature = self.lstm_add_or_norm([lstm_states, feature_hidden_states])  # keep [19, 60, 10]
        # print('temporal_feature: ', temporal_feature.shape)

        # GRN: expanded static enrichment + LSTM glu
        static_enrichment, _ = self.static_context_enrichment(static_encoder)      # [19, 10] -> [19, 10]
        # print('static_enrichment: ', static_enrichment.shape)
        expanded_static_context = static_enrichment.unsqueeze(1)                # [19, 1, 10]
        # print('static_enrichment: ', expanded_static_context.shape)
        enriched, _ = self.static_feature_GRN(temporal_feature, expanded_static_context)   # [19, 60, 10]
        # print('enriched: ', enriched.shape)

        # Interpretable Multi-Head Attention
        if phase == 'train':
            mask = self.mask_train                                              # [19, 60, 60]
        else:
            mask = self.mask_pred
        x, _ = self.self_attn_layer(enriched, enriched, enriched, mask=mask)    # [19, 60, 10]
        # print('x: ', x.shape)

        # GLU + add_and_norm after attention
        x, _ = self.attn_glu(x)
        # print('after attn_glu x: ', x.shape)
        x = self.attn_add_or_norm([x, enriched])                                # [19, 60, 10]

        # decoder: GRN
        x, _ = self.decoder_glu(x)                                                 # [19, 60, 10]
        x = self.decoder_add_or_norm([x, temporal_feature])                     # [19, 60, 10]

        # output
        outputs = self.output_layer(x)                                          # [19, 60, 1]
        outputs = outputs.squeeze(-1)                                           # [19, 60]

        return outputs
