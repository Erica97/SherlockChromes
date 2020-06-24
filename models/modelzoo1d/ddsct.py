"""Dynamic Depth Separable Convolutional Transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modelzoo1d.dain import DAIN_Layer

class DynamicDepthSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3, 15],
        dilation=1,
        bias=False,
        intermediate_nonlinearity=False,
        pad=True
    ):
        super(DynamicDepthSeparableConv1d, self).__init__()
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            bias=bias
        )

        self.intermediate_nonlinearity = intermediate_nonlinearity

        if self.intermediate_nonlinearity:
            self.nonlinear_activation = nn.ReLU()

        # Create dynamic kernel gating mechanism
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)

        self.dynamic_gate = nn.Parameter(
          torch.Tensor([1.0 / self.num_kernels for _ in self.kernel_sizes])
        )

        self.dynamic_depthwise = nn.ModuleList([])
        for kernel_size in self.kernel_sizes:
            padding = ((kernel_size - 1) // 2 * dilation) if pad else 0
            conv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias
            )
            
            self.dynamic_depthwise.append(conv)

    def get_gate(self):
        return self.dynamic_gate

    def forward(self, x):
        out = self.pointwise(x)

        if self.intermediate_nonlinearity:
            out = self.nonlinear_activation(out)

        # Apply dynamically sized kernels to values
        dynamic_out = []
        for dynamic_conv in self.dynamic_depthwise:
            dynamic_out.append(dynamic_conv(out))

        if self.num_kernels > 1:
            out = torch.sum(
                torch.stack(
                    dynamic_out,
                    dim=-1
                ) * F.softmax(self.dynamic_gate, dim=-1),
                dim=-1
            )
        else:
            out = dynamic_out[0]

        return out

class DynamicDepthSeparableTimeSeriesSelfAttention(nn.Module):
    def __init__(
        self,
        c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False):
        super(DynamicDepthSeparableTimeSeriesSelfAttention, self).__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes
        self.save_attn = save_attn

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes
        )

        if share_encoder:
            self.to_keys = self.to_queries
        else:
            self.to_keys = DynamicDepthSeparableConv1d(
                c,
                c * heads,
                kernel_sizes=kernel_sizes
            )

        self.to_values = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes
        )

        # This unifies the outputs of the different heads into a single 
        # c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * c, c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None

    def get_attn(self):
        return self.attn

    def forward(self, x):
        b, c, l = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, h, c, l)
        keys = self.to_keys(x).view(b, h, c, l)
        values = self.to_values(x).view(b, h, c, l)

        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, l)
        keys = keys.view(b * h, c, l)
        values = values.view(b * h, c, l)

        # Get dot product of queries and keys, and scale
        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot now has size (b*h, l, l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, l)

        # Unify heads
        out = self.unify_heads(out)

        if self.save_attn:
            self.attn = dot

        return out

class DynamicDepthSeparableTimeSeriesTemplateAttention(nn.Module):
    def __init__(
        self,
        qk_c,
        v_c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False):
        super(
            DynamicDepthSeparableTimeSeriesTemplateAttention, self).__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes
        self.save_attn = save_attn

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries = DynamicDepthSeparableConv1d(
            qk_c,
            qk_c * heads,
            kernel_sizes=kernel_sizes
        )

        if share_encoder:
            self.to_keys = self.to_queries
        else:
            self.to_keys = DynamicDepthSeparableConv1d(
                qk_c,
                qk_c * heads,
                kernel_sizes=kernel_sizes
            )

        self.to_values = DynamicDepthSeparableConv1d(
            v_c,
            v_c * heads,
            kernel_sizes=kernel_sizes
        )

        # This unifies the outputs of the different heads into a single 
        # v_c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * v_c, v_c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None

    def get_attn(self):
        return self.attn

    def forward(self, queries, keys, values):
        if len(values.size()) == 2:
            values = values.unsqueeze(1)

        q_b, qk_c, l = queries.size()
        kv_b, v_c, _ = values.size()
        h = self.heads

        queries = self.to_queries(queries).view(q_b, h, qk_c, l)
        keys = self.to_keys(keys).view(kv_b, h, qk_c, l)
        values = self.to_values(values).view(kv_b, h, v_c, l)

        # Fold heads into the batch dimension
        queries = queries.view(q_b * h, qk_c, l)
        keys = keys.view(kv_b * h, qk_c, l)
        values = values.view(kv_b * h, v_c, l)

        # Get dot product of queries and key, and scale
        queries = queries / (qk_c ** (1 / 4))
        keys = keys / (v_c ** (1 / 4))

        if kv_b > 1:
            dot = torch.matmul(
                keys.transpose(1, 2).contiguous().view(kv_b * h, l, qk_c),
                queries
            )
            # dot now has size (q_b*h, kv_b*h, l, l) containing raw weights

            dot = F.softmax(dot, dim=2)
            # dot now has channel-wise self-attention probabilities

            # Apply the attention to the values
            out = torch.matmul(values, dot)
            # out now has size (q_b*h, kv_b*h, v_c, l)

            out = torch.sum(out, dim=1)
            # out now has size (q_b*h, v_c, l)
        else:
            dot = torch.matmul(keys.transpose(1, 2), queries)
            # dot now has size (q_b*h, l, l) containing raw weights

            dot = F.softmax(dot, dim=1)
            # dot now has channel-wise self-attention probabilities

            # Apply the attention to the values
            out = torch.matmul(values, dot)
            # out now has size (q_b*h, v_c, l)

        out = out.view(q_b, h * v_c, l)

        # Unify heads
        out = self.unify_heads(out)

        if self.save_attn:
            self.attn = dot

        return out

class TimeSeriesAttentionPooling(nn.Module):
    def __init__(
        self,
        num_queries,
        c,
        save_attn=False):
        super(TimeSeriesAttentionPooling, self).__init__()
        self.save_attn = save_attn

        # This represents the queries for the weak binary global label
        self.query_embeds = nn.Embedding(num_queries, c)

        # TODO: Implement Transformer decoder for multiple queries
        # Currently unstable during training, probably due to difficulty
        # finding orthogonal latent embeddings per query feature.
        if num_queries > 1:
            self.to_out_embed = DynamicDepthSeparableConv1d(
                c,
                c,
                kernel_sizes=[num_queries],
                bias=True,
                pad=False
            )
        else:
            self.to_out_embed = nn.Identity()

        self.attn = None

    def get_attn(self):
        return self.attn

    def forward(self, x):
        b, c, l = x.size()

        queries = self.query_embeds.weight.transpose(0, 1).unsqueeze(0)
        keys = values = x

        # Get dot product of queries and keys, and scale
        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        dot = torch.matmul(keys.transpose(1, 2), queries)
        # dot now has size (b, l, num_queries) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.matmul(values, dot)
        # out now has size (b, c, num_queries)

        out = self.to_out_embed(out)
        # out now has size (b, c, 1)

        if self.save_attn:
            self.attn = dot

        return out

class Conv1dFeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Conv1dFeedForwardNetwork, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(
                n,
                k,
                1
            ) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

class DynamicDepthSeparableTimeSeriesTransformerBlock(nn.Module):
    def __init__(
        self,
        c,
        heads,
        depth_multiplier=4,
        dropout=0.1,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False):
        super(DynamicDepthSeparableTimeSeriesTransformerBlock, self).__init__()
        self.attention = DynamicDepthSeparableTimeSeriesSelfAttention(
            c,
            heads=heads,
            kernel_sizes=kernel_sizes,
            share_encoder=share_encoder,
            save_attn=save_attn
        )

        # Instance norm instead of layer norm
        self.norm1 = nn.InstanceNorm1d(c, affine=True)
        self.norm2 = nn.InstanceNorm1d(c, affine=True)

        # 1D Convolutions instead of FC
        self.feed_forward = Conv1dFeedForwardNetwork(
            c,
            depth_multiplier * c,
            c,
            num_layers=2
        )

        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(self.dropout(attended) + x)

        fed_forward = self.feed_forward(x)
        x = self.norm2(self.dropout(fed_forward) + x)

        return x

class DynamicDepthSeparableConv1dResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3, 15],
        dilation=1,
        bias=False,
        intermediate_nonlinearity=False):
        super(DynamicDepthSeparableConv1dResBlock, self).__init__()
        self.network = nn.Sequential(
            nn.InstanceNorm1d(in_channels, affine=True),
            nn.ReLU(),
            DynamicDepthSeparableConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                dilation=dilation,
                bias=bias,
                intermediate_nonlinearity=intermediate_nonlinearity
            )
        )

        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(
                in_channels,
                out_channels,
                1,
                bias=False
            )
        )

    def forward(self, x):
        out = self.network(x) + self.residual(x)

        return out

class TimeSeriesExtractor(nn.Module):
    def __init__(
        self,
        data_height=420,
        extraction_win=10,
        aggregation_win=7,
        num_ts=6,
        kernel_sizes=[3, 15],
        normalize=False,
        normalization_mode='full',
        save_normalized=False):
        super(TimeSeriesExtractor, self).__init__()
        self.data_height = data_height
        self.extraction_win = extraction_win
        self.aggregation_win = aggregation_win
        self.num_ts = num_ts
        self.save_normalized = save_normalized

        if self.extraction_win > 1:
            self.time_series_generator = nn.Sequential(
                DynamicDepthSeparableConv1dResBlock(
                    extraction_win,
                    16,
                    kernel_sizes=kernel_sizes
                ),
                DynamicDepthSeparableConv1dResBlock(
                    16,
                    1,
                    kernel_sizes=kernel_sizes
                )
            )
        else:
            self.time_series_generator = nn.Identity()

        self.time_series_aggregator = nn.Sequential(
            DynamicDepthSeparableConv1dResBlock(
                data_height // extraction_win // aggregation_win,
                16,
                kernel_sizes=kernel_sizes
            ),
            DynamicDepthSeparableConv1dResBlock(
                16,
                1,
                kernel_sizes=kernel_sizes
            )
        )

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=self.extraction_win
            )
        else:
            self.normalization_layer = nn.Identity()

        self.normalized = None

    def forward(self, x):
        b, c, l = x.size()

        out = x[:, :self.data_height].contiguous().view(
            -1, self.extraction_win, l)
        out = self.normalization_layer(out)

        if self.save_normalized:
            self.normalized = out

        out = self.time_series_generator(out)
        out = out.view(
            -1,
            self.data_height // self.extraction_win // self.aggregation_win,
            l
        )
        out = self.time_series_aggregator(out)
        out = out.view(b, self.num_ts, l)
        out = torch.cat([out, x[:, self.data_height:]], axis=1)

        return out

class DDSCTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        transformer_channels=32,
        heads=8,
        depth_multiplier=4,
        dropout=0.1,
        depth=6,
        kernel_sizes=[3, 15],
        share_encoder=False,
        normalize=False,
        normalization_mode='full',
        save_normalized=False,
        extract_ts=False,
        extractor_data_height=420,
        extractor_extraction_win=10,
        extractor_aggregation_win=7,
        extractor_num_ts=6,
        use_templates=False,
        cat_templates=False,
        aggregator_num_queries=1,
        save_attn=False,
        output_num_layers=1,
        output_mode='strong',
        probs=True):
        super(DDSCTransformer, self).__init__()
        self.save_normalized = save_normalized
        self.use_templates = use_templates
        self.cat_templates = self.use_templates and cat_templates
        self.output_mode = output_mode
        self.probs = probs

        if extract_ts:
            self.time_series_extractor = TimeSeriesExtractor(
                data_height=extractor_data_height,
                extraction_win=extractor_extraction_win,
                aggregation_win=extractor_aggregation_win,
                num_ts=extractor_num_ts,
                kernel_sizes=kernel_sizes,
                normalize=normalize,
                normalization_mode=normalization_mode,
                save_normalized=save_normalized
            )
        else:
            self.time_series_extractor = nn.Identity()

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=in_channels
            )
        else:
            self.normalization_layer = nn.Identity()

        self.init_encoder = nn.Conv1d(
            in_channels,
            transformer_channels,
            1
        )

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        t_blocks = []
        for i in range(depth):
            t_blocks.append(
                DynamicDepthSeparableTimeSeriesTransformerBlock(
                    c=transformer_channels,
                    heads=heads,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout,
                    kernel_sizes=kernel_sizes,
                    share_encoder=share_encoder,
                    save_attn=save_attn
                )
            )
        self.t_blocks = nn.Sequential(*t_blocks)

        t_out_channels = transformer_channels

        if self.use_templates:
            self.templates_attn = (
                DynamicDepthSeparableTimeSeriesTemplateAttention(
                    qk_c=transformer_channels,
                    v_c=1,
                    heads=heads,
                    kernel_sizes=kernel_sizes,
                    share_encoder=share_encoder,
                    save_attn=save_attn
                )
            )

        if self.cat_templates:
            t_out_channels+= 1
        elif self.use_templates:
            t_out_channels = 1

        # Aggregates output to single value per batch item
        self.output_aggregator = TimeSeriesAttentionPooling(
            num_queries=aggregator_num_queries,
            c=t_out_channels,
            save_attn=save_attn
        )

        # Maps the final output state(s) to logits
        self.to_logits = Conv1dFeedForwardNetwork(
            t_out_channels,
            t_out_channels,
            1,
            num_layers=output_num_layers
        )
        
        # Maps the final logits to probabilities
        self.to_probs = nn.Sigmoid()

        self.normalized = None

    def get_normalized(self):
        return self.normalized

    def set_output_probs(self, val):
        self.probs = val

    def forward(self, x, templates=None, templates_label=None):
        b, _, _ = x.size()

        x = self.time_series_extractor(x)

        x = self.normalization_layer(x)

        if self.save_normalized:
            self.normalized = x

        out = self.init_encoder(x)
        out = self.t_blocks(out)

        if self.use_templates:
            templates = self.normalization_layer(templates)
            templates = self.init_encoder(templates)
            templates = self.t_blocks(templates)
            out_weighted = self.templates_attn(out, templates, templates_label)

            if self.cat_templates:
                out = torch.cat([out, out_weighted], dim=1)
            else:
                out = out_weighted
        
        out_dict = {}

        if self.output_mode == 'weak' or self.output_mode == 'both':
            out_dict['weak'] = self.to_logits(self.output_aggregator(out))

        if self.output_mode == 'strong' or self.output_mode == 'both':
            out_dict['strong'] = self.to_logits(out)

        if self.probs:
            for mode in out_dict:
                out_dict[mode] = self.to_probs(out_dict[mode])

        for mode in out_dict:
            out_dict[mode] = out_dict[mode].view(b, -1)

        if self.output_mode == 'weak':
            return out_dict['weak']
        elif self.output_mode == 'strong':
            return out_dict['strong']
        elif self.output_mode == 'both':
            return out_dict
