import math, copy
import torch
import torch.nn as nn
import torch.nn.functional as functional

from playground.nlp.framework.torch.layers.embedding import PositionalEncoding


def clones(module, n):
    """
    Produce n identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Embeddings(nn.Module):
    def __init__(self, vocab, embed_size):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, embed_size)
        self.embed_size = embed_size

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_size)


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = functional.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.h = n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab,
                 n_layer=6,
                 d_model=512,
                 d_ff=2048,
                 n_head=8,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.embedding_layer = Embeddings(src_vocab, d_model)
        self.position_embedding_layer = PositionalEncoding(d_model, dropout)

        attn = MultiHeadedAttention(n_head, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), n_layer)
        self.linear = nn.Linear(d_model, 1)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, x):
        mask = torch.ones_like(x, device=x.device)
        mask[x == 0] = 0

        embed = self.embedding_layer(x)
        pos_embed = self.position_embedding_layer(embed)

        encodes = self.encoder(pos_embed, mask)
        encodes = torch.mean(encodes, dim=1)
        logits = self.linear(encodes)
        return logits
