import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from kdel.gpu_data import _ranges
from kdel.featurize import smiles_to_arrays


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class SynStore:
    """Do thi cua toan bo synthon (~1.7k) nam tren GPU; lay theo id."""
    def __init__(self, smiles, device="cuda"):
        arrs = [smiles_to_arrays(s) for s in smiles]
        nn_ = np.array([a[0].shape[0] for a in arrs]); ee = np.array([a[1].shape[1] for a in arrs])
        T = lambda a, dt=None: torch.from_numpy(a).to(device)
        self.x    = T(np.concatenate([a[0] for a in arrs], 0))
        self.ei   = T(np.concatenate([a[1] for a in arrs], 1))
        self.ea   = T(np.concatenate([a[2] for a in arrs], 0))
        self.nptr = T(np.r_[0, np.cumsum(nn_)].astype(np.int64))
        self.eptr = T(np.r_[0, np.cumsum(ee)].astype(np.int64))
        self.dev  = device
        self.n = len(smiles)

    def batch(self, ids):
        ids = ids.long()
        n0, n1 = self.nptr[ids], self.nptr[ids + 1]
        e0, e1 = self.eptr[ids], self.eptr[ids + 1]
        nl, el = n1 - n0, e1 - e0
        x = self.x[_ranges(n0, nl)].float()
        eid = _ranges(e0, el)
        shift = torch.repeat_interleave(torch.cumsum(nl, 0) - nl, el)
        ei = self.ei[:, eid].long() + shift.unsqueeze(0)
        ea = self.ea[eid].float()
        b = torch.repeat_interleave(torch.arange(len(ids), device=self.dev), nl)
        return x, ei, ea, b


class GINEEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, n_layers, dropout):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList([GINEConv(mlp(hidden, hidden, hidden), edge_dim=hidden)
                                    for _ in range(n_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, ei, ea, batch, n):
        h = self.node_in(x); e = self.edge_in(ea)
        for conv, bn in zip(self.convs, self.bns):
            h = h + self.drop(F.relu(bn(conv(h, ei, e))))      # residual
        return global_add_pool(h, batch, size=n)


class HierDEL(nn.Module):
    """7 token phan cap: A,B,C (GNN tren synthon) ; AB,AC,BC (MLP ghep cap) ; ABC (GNN tren phan tu day du)
       -> 1 lop self-attention -> attention pooling -> dau NB."""
    def __init__(self, node_dim=28, edge_dim=6, hidden=256, n_layers=5, syn_layers=3,
                 dropout=0.2, heads=4):
        super().__init__()
        H = hidden
        self.syn_enc  = GINEEncoder(node_dim, edge_dim, H, syn_layers, dropout)
        self.full_enc = GINEEncoder(node_dim, edge_dim, H, n_layers, dropout)
        self.pos  = nn.Embedding(3, H)
        self.pair = nn.ModuleList([mlp(2 * H, H, H) for _ in range(3)])   # AB, AC, BC
        self.tri  = nn.Linear(H, H)
        self.tok_type = nn.Parameter(torch.randn(7, H) * 0.02)
        self.tf   = nn.TransformerEncoderLayer(H, heads, 2 * H, dropout, batch_first=True, norm_first=True)
        self.q    = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.attn = nn.MultiheadAttention(H, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(H)
        self.score = mlp(H, H, 2)       # log lambda: target, control
        self.zero  = mlp(H, H, 2)       # logit pi (chi dung khi zero_inflated)
        self.log_disp = nn.Parameter(torch.zeros(2))

    def forward(self, d, syn_store):
        B = d.syn.shape[0]
        u, inv = torch.unique(d.syn, return_inverse=True)            # inv: [B,3]
        zs = self.syn_enc(*syn_store.batch(u), n=len(u))             # [U,H]
        zA, zB, zC = [zs[inv[:, k]] + self.pos.weight[k] for k in range(3)]
        zf = self.full_enc(d.x, d.edge_index, d.edge_attr, d.batch, n=B)
        tok = torch.stack([zA, zB, zC,
                           self.pair[0](torch.cat([zA, zB], -1)),
                           self.pair[1](torch.cat([zA, zC], -1)),
                           self.pair[2](torch.cat([zB, zC], -1)),
                           self.tri(zf)], dim=1)  + self.tok_type        # [B,7,H]
        tok = self.tf(tok)
        z, _ = self.attn(self.q.expand(B, -1, -1), tok, tok, need_weights=False)
        z = self.norm(z.squeeze(1))
        return {"log_lambda": self.score(z), "pi_logit": self.zero(z),
                "log_disp": self.log_disp, "z": z}
