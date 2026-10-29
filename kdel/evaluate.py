import argparse, os, json
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from kdel.featurize import smiles_to_arrays
from kdel.hier import HierDEL, SynStore, LMTable


class MolLMTable:
    """Embedding LM muc PHAN TU cho phan tu held-out; tra theo chuoi SMILES."""
    def __init__(self, mol_lm_dir, device="cuda"):
        d  = np.load(f"{mol_lm_dir}/heldout_emb.npz", allow_pickle=True)
        st = np.load(f"{mol_lm_dir}/stats.npz")
        self.idx = {s: i for i, s in enumerate(d["smiles"].tolist())}
        Z = (d["emb"].astype(np.float32) - st["mean"]) / st["std"]
        self.Z = torch.from_numpy(Z).to(device); self.dev = device

    def lookup(self, smiles):
        rows = [self.idx.get(s, -1) for s in smiles]
        miss = sum(r < 0 for r in rows)
        if miss: print(f"[MolLMTable] canh bao: {miss}/{len(rows)} phan tu thieu embedding -> vector 0", flush=True)
        out = self.Z[torch.tensor([max(r, 0) for r in rows], device=self.dev)].clone()
        out[torch.tensor([r < 0 for r in rows], device=self.dev)] = 0.0
        return out


class _B: pass


def to_batch(smiles, syn_ids, dev):
    xs, eis, eas, bs, off = [], [], [], [], 0
    for k, s in enumerate(smiles):
        x, ei, ea = smiles_to_arrays(s)
        xs.append(x); eas.append(ea); eis.append(ei.astype(np.int64) + off)
        bs += [k] * x.shape[0]; off += x.shape[0]
    b = _B()
    b.x = torch.from_numpy(np.concatenate(xs)).float().to(dev)
    b.edge_index = torch.from_numpy(np.concatenate(eis, 1)).long().to(dev)
    b.edge_attr = torch.from_numpy(np.concatenate(eas)).float().to(dev)
    b.batch = torch.tensor(bs, device=dev)
    b.syn = torch.as_tensor(syn_ids, dtype=torch.long, device=dev)
    return b


@torch.no_grad()
def score(model, df, dev, bs=64, lm=None, drop_syn=False, mol_lm=None):
    """-> dict ten_score -> np.array. Synthon duoc featurize truc tiep tu SMILES (khong can nam trong vocab)."""
    cols = ["smiles_a", "smiles_b", "smiles_c"]
    if drop_syn:                      # khong dung synthon: do thi rong + che token (cham duoc phan tu ngoai DEL)
        ids = np.zeros((len(df), 3), dtype=np.int64); syn = SynStore([""], dev, lm=lm)
    else:
        uniq = pd.unique(df[cols].to_numpy().ravel()).tolist()
        mp = {s: i for i, s in enumerate(uniq)}
        ids = np.stack([df[c].map(mp).to_numpy() for c in cols], 1)
        syn = SynStore(uniq, dev, lm=lm)
    smi = df["smiles"].tolist(); s_t, s_d = [], []
    ME = mol_lm.lookup(smi) if mol_lm is not None else None
    model.eval()
    for i in range(0, len(df), bs):
        b = to_batch(smi[i:i + bs], ids[i:i + bs], dev)
        dsm = torch.ones(b.syn.shape[0], dtype=torch.bool, device=dev) if drop_syn else None
        if ME is not None: b.lm_mol = ME[i:i + bs]
        o = model(b, syn, drop_syn=dsm)
        lt, lc = o["log_lambda"].float().unbind(-1)
        s_t.append(lt.cpu()); s_d.append((lt - lc).cpu())
    return {"logλ_target": torch.cat(s_t).numpy(), "logλ_t−logλ_c": torch.cat(s_d).numpy()}


def load_heldout(root, target, cond):
    df = pd.read_csv(f"{root}/{target}_{cond}dna.csv")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for c in ("molecule_hash", "smiles_a", "smiles_b", "smiles_c"):
        if c not in df.columns: df[c] = None
    return df.dropna(subset=["kd"])


def eval_ckpt(model, root, target, dev, lm=None, mol_lm=None):
    """3 o: in-library (co synthon), in-library-nosyn (che synthon -> do muc phu thuoc),
       extended (ca phan tu ngoai DEL, bat buoc che synthon)."""
    rows = []
    for cond in ("on", "off"):
        full = load_heldout(root, target, cond)
        has = full["molecule_hash"].notna() & full[["smiles_a", "smiles_b", "smiles_c"]].notna().all(1)
        jobs = [("in-library", full[has], False), ("in-library-nosyn", full[has], True)]
        if not has.all():
            jobs.append(("extended", full, True))
        for lib, df, dsyn in jobs:
            if len(df) < 5: continue
            df = df.reset_index(drop=True)
            for name, sc in score(model, df, dev, lm=lm, drop_syn=dsyn, mol_lm=mol_lm).items():
                rho = spearmanr(sc, df["kd"].to_numpy()).statistic
                rows.append(dict(cond=cond, lib=lib, n=len(df), score=name, rho=-rho))   # -rho: dau theo bai bao
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)      # thu muc chua best.pt + last.pt
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="eval.csv")
    ap.add_argument("--lm_dir", default=None); ap.add_argument("--mol_lm_dir", default=None)
    a = ap.parse_args()
    allr = []
    for k, d in enumerate(a.ckpts):
        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]
        m = HierDEL(**cfg["model"]).to(a.device)
        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=a.device))
        lm = LMTable(a.lm_dir, a.device) if cfg["model"].get("use_lm") else None
        ml = MolLMTable(a.mol_lm_dir, a.device) if cfg["model"].get("use_mol_lm") else None
        r = eval_ckpt(m, a.root, a.target, a.device, lm=lm, mol_lm=ml); r["ckpt"] = os.path.basename(d.rstrip("/"))
        allr.append(r)
    R = pd.concat(allr); R.to_csv(a.out, index=False)
    print(R.to_string(index=False))
    g = R.groupby(["cond", "lib", "score"], dropna=False).agg(n=("n", "first"), mean=("rho", "mean"),
                                                              std=("rho", "std"), k=("rho", "count"))
    print("\nTRUNG BINH QUA CAC SPLIT (rho theo dau bai bao: cao = tot)")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
