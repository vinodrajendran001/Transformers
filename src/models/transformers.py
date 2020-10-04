from utils.util import mask_

import torch
from torch import nn
import torch.nn.functional as F
from utils.util import d
import random, math

class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """[summary]

        Args:
            emb ([type]): [dimension]
            heads (int, optional): [head]. Defaults to 8.
            mask (bool, optional): [mask]. Defaults to False.
        """
        super().__init__()
        
        self.emb = emb
        self.heads = heads
        self.mask = mask
        
        self.tokeys = nn.Linear(emb, emb*heads, bias=False)
        self.toqueries = nn.Linear(emb, emb*heads, bias=False)
        self.tovalues = nn.Linear(emb, emb*heads, bias=False)
        
        self.unifyheads = nn.Linear(heads*emb, emb)
        
    def forward(self, x):
        b,t,e = x.size()
        assert e == self.emb
        h = self.heads
        
        keys = self.tokeys(x).view(b,t,h,e)
        queries = self.toqueries(x).view(b,t,h,e)
        values = self.tovalues(x).view(b,t,h,e)
        
        #compute scaled dot product self-attention
        
        # fold heads in batch dimension
        keys = keys.transpose(1,2).contiguous().view(b*h,t,e)
        queries = queries.transpose(1,2).contiguous().view(b*h,t,e)
        values = values.transpose(1,2).contiguous().view(b*h,t,e)
        
        # instead of scaling the dot product, the queries and keys are scaled to make memory efficient computation
        queries = queries/(e**(1/4))
        keys = keys/(e**(1/4))
        
        # get dot product of queries and keys, scale
        dot = torch.bmm(queries, keys.transpose(1,2))
        
        assert dot.size() == (b*h, t, t)
        
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)
        
        # dot now has self-attention probabilities
        dot = F.softmax(dot, dim=2)
        
        # apply self-attention to the values
        out = torch.bmm(dot, values).view(b,h,t,e)
        
        out = out.transpose(1,2).contiguous().view(b,t,h*e)
        
        return self.unifyheads(out)

class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """[summary]

        Args:
            emb ([type]): [dimension]
            heads (int, optional): [heads]. Defaults to 8.
            mask (bool, optional): [mask]. Defaults to False.
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        """[summary]

        Args:
            emb (int): Embedding dimension
            heads (int): number of attention heads
            depth (int): number of transformer blocks
            seq_length (int): expected maximum sequence length
            num_tokens (int): number of words in the vocabulary
            num_classes (int): number of classes
            max_pool (bool, optional): Global max pooling or Global average pooling. Defaults to True.
            dropout (float, optional): Dropout percentage. Defaults to 0.0.
            wide (bool, optional): Wide or Narrow transformer block. Defaults to False.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)


class GTransformer(nn.Module):
    """
    Transformer for generating sequence
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, wide=False):
        """[summary]

        Args:
            emb (int): Embedding dimension
            heads (int): number of attention heads
            depth (int): number of transformer blocks
            seq_length (int): expected maximum sequence length
            num_tokens (int): number of words in the vocabulary
            wide (bool, optional): Wide or Narrow transformer block. Defaults to False.
        """

        super().__init__()
        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, mask=True, seq_length=seq_length, wide=wide)
            )

        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """[summary]

        Args:
            x : A (batch, sequence length) integer tensor of token indices.
        """

        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)        