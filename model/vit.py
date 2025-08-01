import torch
from torch import nn

from einops import rearrange
from timm.models.layers import trunc_normal_

import sys
sys.path.append('./utils/')
from utils.pose_embedding import PosEmbFactory

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, a=None):
        # a is the adjacency mask (B, N, N)
        return self.fn(self.norm(x), a) if a is not None else self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, a=None):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        project_out = not (heads == 1 and dim_head == dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, a=None):
        """
        x:        (B, N, D)
        a:        (B, N, N) adjacency mask (0/1 or bool)
        returns:  (B, N, D)
        """
        B, N, _ = x.shape
        # project to queries, keys, values
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # raw attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, heads, N, N)

        # mask out illegal edges before softmax
        if a is not None:
            mask = a.unsqueeze(1)  # (B, 1, N, N)
            scores = scores.masked_fill(mask == 0, float('-1e9'))

        # attention weights
        attn = self.attend(scores)  # (B, heads, N, N)
        attn = self.dropout(attn)

        # weighted sum of values
        out = torch.matmul(attn, v)  # (B, heads, N, head_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, a=None):
        for attn, ff in self.layers:
            x = attn(x, a) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        num_classes,
        pool='cls',
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        self.node_dim = node_dim
        self.to_node_embedding = PosEmbFactory(emb_type="fourier", d_pos=dim)

        # positional embedding per node
        self.pos_embedding = nn.Parameter(torch.randn(1, num_nodes, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer backbone
        self.transformer = Transformer(dim, depth, heads, dim, mlp_dim, dropout)

        # pooling and classification head
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        trunc_normal_(self.mlp_head[1].weight, std=.02)

        # initialize weights
        self.apply(self._init_weights)
        nn.init.constant_(self.mlp_head[1].weight, 0.)
        nn.init.constant_(self.mlp_head[1].bias, 0.)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x, a):
        """
        x: (B, N, node_dim)   node features (e.g., joint coords)
        a: (B, N, N)          adjacency mask for temporal edges
        """
        # embed nodes + positions
        x = self.to_node_embedding(x) + self.pos_embedding  # (B, N, dim)
        x = self.dropout(x)

        # apply Transformer with adjacency-aware attention
        x = self.transformer(x, a)                         # (B, N, dim)

        # pooling
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # final classification
        x = self.to_latent(x)
        return self.mlp_head(x)