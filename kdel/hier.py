import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool
from kdel.gpu_data import _ranges
from kdel.featurize import smiles_to_arrays


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


def clean_syn(s):
    """Muc 'X, Y' = 2 thanh phan -> 'X.Y' (1 do thi 2 manh); 'BLANK'/rong -> '' (nut rong)."""
    if s is None or str(s).strip().upper() in ("BLANK", "NAN", ""):
        return ""
    return ".".join(p.strip() for p in str(s).split(","))


class LMTable:
    """Bang embedding ChemBERTa theo CHUOI SYNTHON THO (da chuan hoa tung chieu; BLANK = 0)."""
    def __init__(self, lm_dir, device="cuda"):
        d = np.load(f"{lm_dir}/synthon_chemberta.npz")
        Z = (d["emb"].astype(np.float32) - d["mean"]) / d["std"]
        Z[d["blank"]] = 0.0
        sm = pd.read_parquet(f"{lm_dir}/synthon_chemberta_smiles.parquet")["smiles"].tolist()
        self.idx = {s: i for i, s in enumerate(sm)}
        self.Z = torch.from_numpy(Z).to(device); self.dim = Z.shape[1]; self.dev = device

    def lookup(self, smiles):
        rows = [self.idx.get(s, -1) for s in smiles]
        miss = sum(r < 0 for r in rows)
        if miss: print(f"[LMTable] canh bao: {miss}/{len(rows)} synthon khong co embedding -> vector 0", flush=True)
        out = self.Z[torch.tensor([max(r, 0) for r in rows], device=self.dev)].clone()
        out[torch.tensor([r < 0 for r in rows], device=self.dev)] = 0.0
        return out


class SynStore:
    """Do thi cua cac synthon nam tren GPU; lay theo id. lm: LMTable (tuy chon)."""
    def __init__(self, smiles, device="cuda", lm=None):
        arrs = [smiles_to_arrays(clean_syn(s)) for s in smiles]
        nn_ = np.array([a[0].shape[0] for a in arrs]); ee = np.array([a[1].shape[1] for a in arrs])
        T = lambda a: torch.from_numpy(a).to(device)
        self.x    = T(np.concatenate([a[0] for a in arrs], 0))
        self.ei   = T(np.concatenate([a[1] for a in arrs], 1))
        self.ea   = T(np.concatenate([a[2] for a in arrs], 0))
        self.nptr = T(np.r_[0, np.cumsum(nn_)].astype(np.int64))
        self.eptr = T(np.r_[0, np.cumsum(ee)].astype(np.int64))
        self.dev  = device
        self.n = len(smiles)
        self.lm = lm.lookup(list(smiles)) if lm is not None else None     # [n, lm_dim]

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


def readout_graph(h, batch, n, proj=None):
    """proj=None -> chi tong (nhu PA1-PA3); nguoc lai noi [sum || mean || max] roi chieu ve H."""
    if proj is None:
        return global_add_pool(h, batch, size=n)
    return proj(torch.cat([global_add_pool(h, batch, size=n),
                           global_mean_pool(h, batch, size=n),
                           global_max_pool(h, batch, size=n)], -1))


class DMPNNEncoder(nn.Module):
    """Truyen tin theo CANH CO HUONG (kieu Chemprop). featurize luu moi lien ket thanh hai
       canh nguoc nhau LIEN TIEP va moi phan tu co so canh chan, nen canh nguoc cua canh k
       trong batch luon la k^1."""
    def __init__(self, node_dim, edge_dim, hidden, n_layers, dropout, readout="sum"):
        super().__init__()
        self.W_i = nn.Linear(node_dim + edge_dim, hidden)
        self.W_h = nn.Linear(hidden, hidden)
        self.W_o = nn.Linear(node_dim + hidden, hidden)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.hidden = hidden
        self.out = nn.Linear(3 * hidden, hidden) if readout == "smm" else None

    def forward(self, x, ei, ea, batch, n):
        src, dst = ei[0], ei[1]
        m = src.numel()
        if m == 0:
            S = x.new_zeros(x.shape[0], self.hidden)
            hv = F.relu(self.W_o(torch.cat([x.to(S.dtype), S], -1)))
            return readout_graph(hv, batch, n, self.out)
        h0 = F.relu(self.W_i(torch.cat([x[src], ea], -1)))        # [m,H] tin tren canh
        h, rev = h0, torch.arange(m, device=x.device) ^ 1
        for bn in self.bns:
            S = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=h.dtype)
            S = S.index_add_(0, dst, h)                           # S[a] = tong canh di VAO a
            h = F.relu(h0 + self.drop(bn(self.W_h(S[src] - h[rev]))))   # tru canh nguoc
        S = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=h.dtype)
        S = S.index_add_(0, dst, h)
        hv = F.relu(self.W_o(torch.cat([x.to(S.dtype), S], -1)))
        return readout_graph(hv, batch, n, self.out)


class GINEEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden, n_layers, dropout, readout="sum"):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList([GINEConv(mlp(hidden, hidden, hidden), edge_dim=hidden)
                                    for _ in range(n_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)
        self.out   = nn.Linear(3 * hidden, hidden) if readout == "smm" else None

    def forward(self, x, ei, ea, batch, n):
        h = self.node_in(x); e = self.edge_in(ea)
        for conv, bn in zip(self.convs, self.bns):
            h = h + self.drop(F.relu(bn(conv(h, ei, e))))      # residual
        return readout_graph(h, batch, n, self.out)


class HierDEL(nn.Module):
    """7 token phan cap: A,B,C (GNN tren synthon [+ ChemBERTa]) ; AB,AC,BC (MLP ghep cap) ; ABC (GNN tren phan tu day du)
       -> 1 lop self-attention -> attention pooling -> dau NB.
       use_lm: cong embedding ChemBERTa (da chieu) vao token synthon.
       fusion: 'add' = khong cong ; 'gate' = tanh(alpha) * proj(emb)  (kieu MolProphecy, alpha la so vo huong hoc duoc).
       aux_sizes: (nA,nB,nC) -> them dau doan ID synthon tu vector tong hop z."""
    def __init__(self, node_dim=28, edge_dim=6, hidden=256, n_layers=5, syn_layers=3,
                 dropout=0.2, heads=4, use_lm=False, lm_dim=384, fusion="add", aux_sizes=None,
                 mol_enc="gine", readout="sum", use_mol_lm=False, mol_lm_dim=384):
        super().__init__()
        H = hidden
        self.use_lm, self.fusion = use_lm, fusion
        self.use_mol_lm = use_mol_lm
        self.syn_enc  = GINEEncoder(node_dim, edge_dim, H, syn_layers, dropout, readout)
        self.full_enc = (DMPNNEncoder(node_dim, edge_dim, H, n_layers, dropout, readout)
                         if mol_enc == "dmpnn" else
                         GINEEncoder(node_dim, edge_dim, H, n_layers, dropout, readout))
        if use_mol_lm:
            self.mol_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(mol_lm_dim, H))
        if use_lm:
            self.lm_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(lm_dim, H))
            if fusion == "gate":
                self.lm_gate = nn.Parameter(torch.zeros(1))
        self.pos  = nn.Embedding(3, H)
        self.pair = nn.ModuleList([mlp(2 * H, H, H) for _ in range(3)])   # AB, AC, BC
        self.tri  = nn.Linear(H, H)
        self.n_tok = 8 if use_mol_lm else 7
        self.tok_type = nn.Parameter(torch.randn(self.n_tok, H) * 0.02)
        self.tf   = nn.TransformerEncoderLayer(H, heads, 2 * H, dropout, batch_first=True, norm_first=True)
        self.q    = nn.Parameter(torch.randn(1, 1, H) * 0.02)
        self.attn = nn.MultiheadAttention(H, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(H)
        self.score = mlp(H, H, 2)       # log lambda: target, control
        self.zero  = mlp(H, H, 2)       # logit pi (chi dung khi zero_inflated)
        self.log_disp = nn.Parameter(torch.zeros(2))
        self.aux = nn.Linear(H, int(sum(aux_sizes))) if aux_sizes else None

    def forward(self, d, syn_store, drop_syn=None):
        B = d.syn.shape[0]
        u, inv = torch.unique(d.syn, return_inverse=True)            # inv: [B,3]
        zs = self.syn_enc(*syn_store.batch(u), n=len(u))             # [U,H]
        if self.use_lm:
            e = self.lm_proj(syn_store.lm[u])
            if self.fusion == "gate":
                e = torch.tanh(self.lm_gate) * e
            zs = zs + e
        zA, zB, zC = [zs[inv[:, k]] + self.pos.weight[k] for k in range(3)]
        zf = self.full_enc(d.x, d.edge_index, d.edge_attr, d.batch, n=B)
        toks = [zA, zB, zC,
                self.pair[0](torch.cat([zA, zB], -1)),
                self.pair[1](torch.cat([zA, zC], -1)),
                self.pair[2](torch.cat([zB, zC], -1)),
                self.tri(zf)]
        if self.use_mol_lm:
            toks.append(self.mol_proj(d.lm_mol))      # token 8: KHONG bi che boi synthon-dropout
        tok = torch.stack(toks, dim=1) + self.tok_type        # [B,n_tok,H]
        kpm = None
        if drop_syn is not None:              # che 6 token synthon -> chi con token ABC
            kpm = torch.zeros(B, self.n_tok, dtype=torch.bool, device=tok.device)
            kpm[:, :6] = drop_syn.view(-1, 1)
        tok = self.tf(tok, src_key_padding_mask=kpm)
        z, _ = self.attn(self.q.expand(B, -1, -1), tok, tok, need_weights=False, key_padding_mask=kpm)
        z = self.norm(z.squeeze(1))
        out = {"log_lambda": self.score(z), "pi_logit": self.zero(z),
               "log_disp": self.log_disp, "z": z}
        if self.aux is not None:
            out["aux"] = self.aux(z)
        return out
