import numpy as np, torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class KinDELGraphDataset(Dataset):
    def __init__(self, indices, root):
        super().__init__()
        self.idx  = np.asarray(indices, np.int64)
        self.root = root
        self._h   = None                      # mo memmap lazy, an toan voi worker

    def _open(self):
        if self._h is not None: return
        r = self.root
        self._h = dict(
            x    = np.load(f"{r}/graphs_x_fp16.npy",   mmap_mode="r"),
            ei   = np.load(f"{r}/graphs_ei_int16.npy", mmap_mode="r"),
            ea   = np.load(f"{r}/graphs_ea_fp16.npy",  mmap_mode="r"),
            nptr = np.load(f"{r}/graph_nptr.npy"),
            eptr = np.load(f"{r}/graph_eptr.npy"),
            cnt  = np.load(f"{r}/counts_int32.npy"),
        )

    def len(self): return len(self.idx)

    def get(self, k):
        self._open(); h = self._h; i = int(self.idx[k])
        a, b = h["nptr"][i], h["nptr"][i+1]
        c, d = h["eptr"][i], h["eptr"][i+1]
        g = Data(
            x          = torch.from_numpy(np.array(h["x"][a:b])).float(),
            edge_index = torch.from_numpy(np.array(h["ei"][:, c:d])).long(),
            edge_attr  = torch.from_numpy(np.array(h["ea"][c:d])).float(),
        )
        cnt = h["cnt"][i]
        g.target_counts  = torch.from_numpy(cnt[0:3].astype(np.int64)).view(1,3)
        g.control_counts = torch.from_numpy(cnt[3:6].astype(np.int64)).view(1,3)
        g.pre = torch.tensor([float(cnt[6]) + 1.0])
        return g

def make_loader(ds, bs, shuffle, num_workers=0):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle)
