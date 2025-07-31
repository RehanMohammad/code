import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataset.IPN import _TEMPORAL_RULES

class TemporalEdgeMAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder_dim: int,
        edge_masking_ratio: float = 0.5,
    ):
        """
        encoder: ViT-style encoder with:
            - to_node_embedding: nn.Module mapping (B,N,node_dim)->(B,N,enc_dim)
            - pos_embedding:     (1, N, enc_dim) positional embeddings
            - transformer:       the ViT Transformer block
        decoder_dim:         latent dimension for edge decoder
        edge_masking_ratio:  fraction of adjacency edges to mask
        """
        super().__init__()
        self.encoder = encoder
        self.edge_masking_ratio = edge_masking_ratio

        # project encoder output -> decoder_dim if needed
        enc_dim = encoder.pos_embedding.shape[-1]
        self.enc_to_dec = (
            nn.Linear(enc_dim, decoder_dim)
            if enc_dim != decoder_dim else nn.Identity()
        )

        # edge reconstruction head: concat embeddings of node pairs -> weight
        self.edge_decoder = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, 1)
        )

    def _generate_edge_mask(self, a: torch.Tensor):
        """
        a: (B, T, N, N) temporal adjacency
        Returns:
            a_masked: (B, T, N, N) with masked entries zeroed
            edge_mask: (B, T, N, N) bool mask of which entries were hidden
        """
        # collapse time into batch
        B, T, N, _ = a.shape
        a_flat = a.view(B * T, N, N)

        # now do exactly what you had for the 3D case
        flat = a_flat.view(B * T, -1)
        total = flat.size(1)
        k = int(self.edge_masking_ratio * total)

        masks = torch.zeros_like(flat, dtype=torch.bool)
        for i in range(B * T):
            idx = torch.randperm(total, device=a.device)[:k]
            masks[i, idx] = True

        flat_masked = flat.clone()
        flat_masked[masks] = 0

        # reshape back to (B,T,N,N)
        a_masked = flat_masked.view(B, T, N, N)
        edge_mask = masks.view(B, T, N, N)
        return a_masked, edge_mask

    def forward(self, x, a=None):
        # 1) force (B, D, T, V) → (B, N=T*V, D)
        B, C, T, V = x.shape
        N = T * V
        x = x.permute(0,2,3,1).reshape(B, N, C)

        # 2) if no adjacency given, build the temporal one
        if a is None:
            mask = torch.zeros(N, N, dtype=torch.bool, device=x.device)
            for t in range(T-1):
                for src, dests in _TEMPORAL_RULES.items():
                    i = t*V + src
                    for dst in dests:
                        j = (t+1)*V + dst
                        mask[i, j] = True
            a = mask.unsqueeze(0).expand(B, N, N)

        # 3) mask some edges
        a_masked, edge_mask = self._generate_edge_mask(a)  # (B,N,N),(B,N,N)

        # 4) encode + pos-embed
        emb = self.encoder.to_node_embedding(x)            # (B,N,enc_dim)
        emb = emb.permute(0, 2, 1)                         # → (B, N, enc_dim)
        # emb = emb + self.encoder.pos_embedding             # (1,N,enc_dim)

        enc = self.encoder.transformer(emb, a_masked)      # (B,N,enc_dim)
        dec = self.enc_to_dec(enc)                         # (B,N,dec_dim)

        # 5) reconstruct edges
        Zi = dec.unsqueeze(2).expand(B,N,N,-1)
        Zj = dec.unsqueeze(1).expand(B,N,N,-1)
        feats = torch.cat([Zi, Zj], dim=-1)                # (B,N,N,2*dec)
        edge_pred = self.edge_decoder(feats).squeeze(-1)   # (B,N,N)

        # 6) loss over the masked entries
        loss = F.mse_loss(edge_pred[edge_mask], a[edge_mask])
        return edge_pred, loss

    def inference(self, x: torch.Tensor, a: torch.Tensor):
        """
        Compute embeddings without masking and return edge predictions.
        """
        B, N, _ = x.shape
        # emb = self.encoder.to_node_embedding(x) + self.encoder.pos_embedding
        emb = self.encoder.to_node_embedding(x)
        enc = self.encoder.transformer(emb, a)
        dec = self.enc_to_dec(enc)
        Zi = dec.unsqueeze(2).expand(B, N, N, -1)
        Zj = dec.unsqueeze(1).expand(B, N, N, -1)
        edge_feats = torch.cat([Zi, Zj], dim=-1)
        return self.edge_decoder(edge_feats).squeeze(-1)  # (B, N, N)
