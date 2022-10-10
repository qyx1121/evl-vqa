import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertConfig
from transformers.activations import gelu
from transformers.modeling_outputs import BaseModelOutput

import math
import copy
import numpy as np
class Bert(nn.Module):
    """ Finetuned *BERT module """

    def __init__(self, bert_tokenizer):
        super(Bert, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        
        self.bert.resize_token_embeddings(len(bert_tokenizer))
        
        # You can uncomment this to freeze the language model for the 2nd-stage finetuning
        # for name, param in self.bert.named_parameters():
        #     param.requires_grad = False
        
        for k,v in self.bert.named_parameters():
            v.requires_grad = False
    
    def forward(self, tokens):
        attention_mask = (tokens > 0).float()
        outs = self.bert(tokens, attention_mask=attention_mask)
        embds = outs[0]
        
        return embds, outs[1][-1]


class Sentence_Maxpool(nn.Module):
    """ Utilitary for the answer module """

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x_in):
        x = self.fc(x_in)
        x = torch.max(x, dim=1)[0]
        if self.relu:
            x = F.relu(x)
        return x


class FFN(nn.Module):
    def __init__(self, word_dim, hidden_dim, out_dim, dropout=0.3):
        super().__init__()
        activation = "gelu"
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=word_dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x

class AModel(nn.Module):
    """
    Answer embedding module
    """

    def __init__(self, bert_tokenizer, word_dim=768, out_dim=512):
        super(AModel, self).__init__()
        self.bert = Bert(bert_tokenizer)
        self.ans_proj = nn.Linear(word_dim, out_dim)
        self.que_proj = nn.Linear(word_dim, out_dim)
        
    def forward(self, answer, ltype = 'ans'):
        if len(answer.shape) == 3:
            #multi-choice
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer, hd_state = self.bert(answer)
            if ltype == 'ans':
                answer = self.ans_proj(answer)
            else:
                answer = self.que_proj(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
            answer_g = answer_g.view(bs, nans, -1)
        else:
            answer, hd_state = self.bert(answer)
            if ltype == 'ans':
                answer = self.ans_proj(answer)
            else:
                answer = self.que_proj(answer)
            answer_g = answer.mean(dim=1)
            # answer_g = answer[:, 0, :]
        
        return answer_g, answer

def create_sinusoidal_embeddings(n_pos, dim, out):
    with torch.no_grad():
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.num_attention_heads #config.n_heads
        self.dim = config.hidden_size #config.dim
        dp_rate = config.attention_probs_dropout_prob #config.attention_dropout
        self.dropout = nn.Dropout(p=dp_rate)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout, dim, hidden_dim = config.attention_probs_dropout_prob, config.hidden_size, config.intermediate_size
        activation = config.hidden_act

        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # dim = config.dim
        dim = config.hidden_size
        # assert config.dim % config.n_heads == 0
        assert dim % config.num_attention_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.n_layers = config.n_layers
        self.n_layers = config.num_hidden_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.n_layers)]
        )

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            #hidden_states=all_hidden_states,
            #attentions=all_attentions,
        )


class Embeddings(nn.Module):
    def __init__(
        self, d_model, language_len, vision_len, dropout, sinusoidal_pos_embds, d_pos=128
    ):
        super().__init__()
        max_position_embeddings = language_len + vision_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_model,
                # out=self.position_embeddings.weight,
                out=self.position_embeddings.weight,
            )
        self.modality_embedding = nn.Embedding(2, d_model)
        self.language_len = language_len
        self.vision_len = vision_len
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, embeddings):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        token_type_id = torch.tensor(
                [0] * (seq_length-self.vision_len) + [1] * self.vision_len, dtype=torch.long,device=embeddings.device
            )
        modality_embeddings = self.modality_embedding(token_type_id)
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)

    
        embeddings = (
            embeddings + position_embeddings + modality_embeddings
            )  # (bs, max_seq_length, dim)
        #else:
        #    embeddings = embeddings + position_embeddings # (bs, max_seq_length, dim)

        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        
        return embeddings


class POSEmbeddings(nn.Module):
    def __init__(
        self, d_model, max_seq_len, dropout, sinusoidal_pos_embds,d_pos=128
    ):
        super().__init__()
        max_position_embeddings = max_seq_len
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_pos)
        if sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=max_position_embeddings,
                dim=d_pos,
                out=self.position_embeddings.weight,
            )
        self.merge_pos = nn.Sequential(
            nn.Linear(d_model+d_pos, d_model),
            nn.ELU(inplace=True))

    def forward(self, embeddings, cid):
        seq_length = embeddings.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=embeddings.device
        )  # (max_seq_length)
        position_ids += cid*seq_length
        
        position_ids = position_ids.unsqueeze(0).expand_as(
            embeddings[:, :, 0]
        )  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)
        # print(position_embeddings.shape)
        embeddings = self.merge_pos(torch.cat([embeddings, position_embeddings],dim=-1)) # (bs, max_seq_length, dim)
        
        cpos_embed = position_embeddings.mean(dim=1) #(bs, dim)
        return embeddings, cpos_embed