import numpy as np, torch

def _ranges(starts, lens):
    tot = int(lens.sum())
    new0 = torch.cumsum(lens, 0) - lens
    return (torch.arange(tot, device=starts.device)
            - torch.repeat_interleave(new0, lens)
            + torch.repeat_interleave(starts, lens))

class Batch:
    pass

class GPUGraphStore:
    def __init__(self, root, device="cuda", mol_lm=None):
        L = lambda n: torch.from_numpy(np.load(f"{root}/{n}")).to(device)
        self.x    = L("graphs_x_fp16.npy")
        self.ei   = L("graphs_ei_int16.npy")
        self.ea   = L("graphs_ea_fp16.npy")
        self.nptr = L("graph_nptr.npy").long()
        self.eptr = L("graph_eptr.npy").long()
        self.cnt  = L("counts_int32.npy").long()
        self.w    = L("conf_weight_fp16.npy")
        self.dev  = device
        self.lm = None
        if mol_lm:                                    # embedding LM muc phan tu (PA4)
            st = np.load(f"{mol_lm}/stats.npz")
            self.lm = torch.from_numpy(np.load(f"{mol_lm}/mol_emb_fp16.npy")).to(device)
            self.lm_mean = torch.from_numpy(st["mean"]).to(device)
            self.lm_std  = torch.from_numpy(st["std"]).to(device)

    def batch(self, idx):
        idx = idx.to(self.dev).long()
        n0, n1 = self.nptr[idx], self.nptr[idx + 1]
        e0, e1 = self.eptr[idx], self.eptr[idx + 1]
        nl, el = n1 - n0, e1 - e0
        B = Batch()
        B.x = self.x[_ranges(n0, nl)].float()
        eid = _ranges(e0, el)
        shift = torch.repeat_interleave(torch.cumsum(nl, 0) - nl, el)
        B.edge_index = self.ei[:, eid].long() + shift.unsqueeze(0)
        B.edge_attr  = self.ea[eid].float()
        B.batch = torch.repeat_interleave(torch.arange(len(idx), device=self.dev), nl)
        c = self.cnt[idx]
        B.target_counts, B.control_counts = c[:, 0:3], c[:, 3:6]
        B.pre = c[:, 6].float() + 1.0
        B.weight = self.w[idx].float()
        B.idx = idx
        if self.lm is not None:
            B.lm_mol = (self.lm[idx].float() - self.lm_mean) / self.lm_std
        return B

def epoch_batches(store, indices, bs, shuffle, seed=0):
    ind = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        ind = ind[torch.randperm(len(ind), generator=g)]
    n = len(ind) // bs if shuffle else (len(ind) + bs - 1) // bs
    for i in range(n):
        yield store.batch(ind[i*bs:(i+1)*bs])
