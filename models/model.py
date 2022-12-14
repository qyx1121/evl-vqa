#!/usr/bin/env python

from typing import Dict, Iterable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from collections import OrderedDict
from .bert import AModel

#from .vision_transformer import QuickGELU, Attention
from .weight_loaders import weight_loader_fn_dict
from .vision_transformer import (
    VisionTransformer2D, TransformerDecoderLayer,
    model_to_fp16, vit_presets,
)
from .transformer import lmodel
from .cmatt import CMAtten

class TemporalCrossAttention(nn.Module):

    def __init__(
        self,
        spatial_size: Tuple[int, int] = (14, 14),
        feature_dim: int = 768,
    ):
        super().__init__()

        self.spatial_size = spatial_size

        w_size = np.prod([x * 2 - 1 for x in spatial_size])
        self.w1 = nn.Parameter(torch.zeros([w_size, feature_dim]))
        self.w2 = nn.Parameter(torch.zeros([w_size, feature_dim]))

        idx_tensor = torch.zeros([np.prod(spatial_size) for _ in (0, 1)], dtype=torch.long)
        for q in range(np.prod(spatial_size)):
            qi, qj = q // spatial_size[1], q % spatial_size[1]
            for k in range(np.prod(spatial_size)):
                ki, kj = k // spatial_size[1], k % spatial_size[1]
                i_offs = qi - ki + spatial_size[0] - 1
                j_offs = qj - kj + spatial_size[1] - 1
                idx_tensor[q, k] = i_offs * (spatial_size[1] * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor


    def forward_half(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        q, k = q[:, :, 1:], k[:, :, 1:] # remove cls token

        assert q.size() == k.size()
        assert q.size(2) == np.prod(self.spatial_size)

        attn = torch.einsum('ntqhd,ntkhd->ntqkh', q / (q.size(-1) ** 0.5), k)
        attn = attn.softmax(dim=-2).mean(dim=-1) # L, L, N, T

        self.idx_tensor = self.idx_tensor.to(w.device)
        w_unroll = w[self.idx_tensor] # L, L, C
        ret = torch.einsum('ntqk,qkc->ntqc', attn, w_unroll)

        return ret


    def forward(self, q: torch.Tensor, k: torch.Tensor):
        N, T, L, H, D = q.size()
        assert L == np.prod(self.spatial_size) + 1

        ret = torch.zeros([N, T, L, self.w1.size(-1)], device='cuda')
        ret[:, 1:, 1:, :] += self.forward_half(q[:, 1:, :, :, :], k[:, :-1, :, :, :], self.w1)
        ret[:, :-1, 1:, :] += self.forward_half(q[:, :-1, :, :, :], k[:, 1:, :, :, :], self.w2)

        return ret


class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout) for _ in range(num_layers)]
        )

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1, padding=1, groups=in_feature_dim) for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList(
                [TemporalCrossAttention(spatial_size, in_feature_dim) for _ in range(num_layers)]
            )

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))


    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)


    def forward(self, in_features, q_feat):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)
        #x = q_feat.unsqueeze(1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']
            
            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1) # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous() # N, T, L, C
                frame_features += feat
            
            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)
            
            if self.enable_temporal_cross_attention:
                frame_features += self.cross_attention[i](in_features[i]['q'], in_features[i]['k'])

            frame_features = frame_features.flatten(1, 2) # N, T * L, C
            
            x = self.decoder_layers[i](x, frame_features)
        
        return x


class EVLTransformer(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        backbone_name: str = 'ViT-B/16',
        backbone_type: str = 'clip',
        backbone_path: str = 'ckpts/ViT-L-14.pt',
        backbone_mode: str = 'freeze_fp32',
        decoder_num_layers: int = 4,
        decoder_qkv_dim: int = 768,
        decoder_num_heads: int = 12,
        decoder_mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        decoder_mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.decoder_num_layers = decoder_num_layers

        backbone_config = self._create_backbone(backbone_name, backbone_type, backbone_path, backbone_mode)
        backbone_feature_dim = backbone_config['feature_dim']
        backbone_spatial_size = tuple(x // y for x, y in zip(backbone_config['input_size'], backbone_config['patch_size']))

        self.decoder = EVLDecoder(
            num_frames=num_frames,
            spatial_size=backbone_spatial_size,
            num_layers=decoder_num_layers,
            in_feature_dim=backbone_feature_dim,
            qkv_dim=decoder_qkv_dim,
            num_heads=decoder_num_heads,
            mlp_factor=decoder_mlp_factor,
            enable_temporal_conv=enable_temporal_conv,
            enable_temporal_pos_embed=enable_temporal_pos_embed,
            enable_temporal_cross_attention=enable_temporal_cross_attention,
            mlp_dropout=decoder_mlp_dropout,
        )
        '''
        self.proj = nn.Sequential(
            nn.LayerNorm(backbone_feature_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(backbone_feature_dim, num_classes),
        )'''
        self.answer_embeddings = None

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.l_encoder = AModel(tokenizer, out_dim=backbone_feature_dim)
        #self.l_encoder = self._create_transformer(backbone_path)
        self.cm_att = CMAtten()

    def _create_transformer(self, backbone_path):
        ckpt = torch.load(backbone_path)
        state_dict = OrderedDict()
        for k,v in ckpt.named_parameters():
            state_dict[k] = v
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
        model = lmodel(vocab_size, transformer_width, context_length, transformer_layers)
        model.load_state_dict(ckpt.state_dict(), strict=False)
        return model
    
    def _create_backbone(
        self,
        backbone_name: str,
        backbone_type: str,
        backbone_path: str,
        backbone_mode: str,
    ) -> dict:
        weight_loader_fn = weight_loader_fn_dict[backbone_type]
        state_dict = weight_loader_fn(backbone_path)

        backbone = VisionTransformer2D(return_all_features=True, **vit_presets[backbone_name])
        backbone.load_state_dict(state_dict, strict=True) # weight_loader_fn is expected to strip unused parameters

        assert backbone_mode in ['finetune', 'freeze_fp16', 'freeze_fp32']

        if backbone_mode == 'finetune':
            self.backbone = backbone
        else:
            backbone.eval().requires_grad_(False)
            if backbone_mode == 'freeze_fp16':
                model_to_fp16(backbone)
            self.backbone = [backbone] # avoid backbone parameter registration

        return vit_presets[backbone_name]


    def _get_backbone(self, x):
        if isinstance(self.backbone, list):
            # freeze backbone
            self.backbone[0] = self.backbone[0]
            return self.backbone[0]
        else:
            # finetune bakbone
            return self.backbone

    def _compute_answer_embedding(self, a2v):
        self.answer_embeddings = self.get_answer_embedding(a2v)

    def get_answer_embedding(self, answer):

        answer_g, answer = self.l_encoder(answer)
        return answer_g, answer 

    def forward(self, x, q, text_mask, answer = None, q_len = None):
        
        q_feat, q = self.l_encoder(q) # q_feat
        
        backbone = self._get_backbone(x).to(x.device)

        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
        features = backbone(x)[-self.decoder_num_layers:]
        features = [
            dict((k, v.float().view(B, T, *v.size()[1:])) for k, v in x.items())
            for x in features
        ]

        x = self.decoder(features, q_feat) # bs x 1 x 1024 - > bs x 1024

        x_len = torch.ones_like(q_len)
        qv_feat, _ = self.cm_att(x, x_len, q, q_len)
        qv_feat = qv_feat + x
        if answer == []:
            answer_g, _ = self.answer_embeddings
        else:
            answer_g, _ =self.get_answer_embedding(answer)
        
        answer_k = answer_g.clone().to(x.device)
        pred = x @ answer_k.t()
        pred = pred.squeeze(1)
        return pred
