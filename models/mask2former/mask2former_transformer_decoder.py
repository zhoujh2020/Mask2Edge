# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from models.utils.Dwpart import Dwpart
from .position_encoding import PositionEmbeddingSine
from models.utils.right import GCBlock8



class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class SelfAttentionLayer2(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pelayer = PositionEmbeddingSine(128, normalize=True)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, tgt_k,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None, key_pos: Optional[Tensor] = None):
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(tgt_k, key_pos)
        tgt2 = self.self_attn(q, k, value=tgt_k, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, tgt_k,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, key_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_k, tgt_mask,
                                    tgt_key_padding_mask, query_pos, key_pos)
        return self.forward_post(tgt, tgt_k, tgt_mask,
                                 tgt_key_padding_mask, query_pos, key_pos)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
            self,
            in_channels=256,
            mask_classification=True,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=8,
            pre_norm=False,
            mask_dim=256,
            enforce_input_project=False
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.pe_layer2 = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.features_cross_attention_layers = nn.ModuleList()
        self.features_self_attention_layers = nn.ModuleList()
        self.Fpath_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.features_ffn_layers = nn.ModuleList()
        self.dwPart_layers = nn.ModuleList()
        self.q2q = nn.ModuleList()
        for i in range(self.num_layers):
            self.dwPart_layers.append(Dwpart())
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.Fpath_self_attention_layers.append(
                SelfAttentionLayer2(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.features_self_attention_layers.append(
                GCBlock8(256, 256, 9)
            )
            self.features_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.q2q.append(nn.Linear(512,256))
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_2 = nn.Embedding(num_queries, hidden_dim)
        self.pos2 = nn.Embedding(100, 18)
        self.pos3 = nn.Embedding(100, 18)
        # self.q2q = nn.Linear(512,256)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 4
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.out_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.0)
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        pos2 = self.pos2.weight.unsqueeze(0).repeat(bs, 1, 1)
        pos3 = self.pos3.weight.unsqueeze(0).repeat(bs, 1, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        output_class, outputs_mask, attn_mask, hard_mask = self.forward_prediction_heads(output, mask_features,
                                                                               attn_mask_target_size=size_list[0],
                                                                               out_features=x[0])
        for i in range(self.num_layers):
            # architecture 1: init_query
            init_query = output
            level_index = i % self.num_feature_levels
            bs, _, h, w = x[level_index].shape
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output_g = self.dwPart_layers[i](x[level_index], query=output, attn_mask=attn_mask)
            output_l, x[level_index] = self.features_self_attention_layers[i](query=init_query, value=x[level_index], hard_sigmoid_masks = hard_mask, pos2=pos2, pos3=pos3)
            output_2 = self.q2q[i](torch.cat((output_g, output_l), dim=-1))
            output = output + self.dropout(output_2) 
            output = self.norm(output)  # after
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            output_class, outputs_mask, attn_mask, hard_mask = self.forward_prediction_heads(output, mask_features,
                                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
                                                                                    out_features=x[level_index])

            if i >= self.num_layers - 4: predictions_class.append(output_class)
            if i == self.num_layers - 1: predictions_mask.append(outputs_mask)
        assert len(predictions_class) == self.num_feature_levels
        out = {
            'pred_logits': predictions_class,
            'pred_masks': predictions_mask[-1],
        }
        return out
        
    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, out_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        out_embed = self.out_embed(decoder_output)
        outputs_class = torch.einsum('bqc, bchw->bqhw', out_embed, out_features)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        bs, q, h, w = outputs_mask.size()
        region_size = (h//3, w//3)
        reg_mask = F.avg_pool2d(outputs_mask, kernel_size=region_size, ceil_mode=True, count_include_pad=False)
        
        reg_mask_up = F.interpolate(reg_mask, size= (outputs_mask.size()[2], outputs_mask.size()[3]), mode="nearest")  # same vector dimension
        
        reg_up_length = torch.sqrt((reg_mask_up ** 2).flatten(2).sum(2).unsqueeze(2).unsqueeze(3) + 1e-10)
        # reg_length = torch.sqrt((reg_mask ** 2).flatten(2).sum(2).unsqueeze(2).unsqueeze(3) + 1e-10) 
        outputs_length = torch.sqrt((outputs_mask**2).flatten(2).sum(2).unsqueeze(2).unsqueeze(3) + 1e-10)    
        outputs_cub_length = torch.sqrt((outputs_mask**6).flatten(2).sum(2).unsqueeze(2).unsqueeze(3) + 1e-10)
        
        nei_multi = (reg_mask_up.flatten(2) * (outputs_mask ** 3).flatten(2)).sum(2).unsqueeze(2).unsqueeze(3)
        cosine = nei_multi/(reg_up_length * outputs_cub_length)
        
        reg_mask_proj = outputs_length * cosine * (reg_mask_up/reg_up_length) 
        
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        hard_mask = (reg_mask.sigmoid() > 0.5).float()
        reg_mask = (reg_mask.sigmoid() < 0.5).float()
        attn_mask = F.interpolate(reg_mask_proj, size= attn_mask_target_size, mode="nearest")
        attn_mask = (attn_mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,1) > 0.5).bool()
        
        hard_mask = hard_mask.detach()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, hard_mask

