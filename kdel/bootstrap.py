import argparse, os
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
from kdel.hier import HierDEL, LMTable
from kdel.evaluate import load_heldout, score

# Spearman tot nhat cua bai bao KinDEL (disynthon, in-library) - de doi chieu
BASELINE = {"ddr1":   {"on": 0.663, "off": 0.463},
            "mapk14": {"on": 0.402, "off": 0.748}}


def spearman_rows(a, b):
    """a,b: [B,n] -> spearman tung hang."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra = ra - ra.mean(1, keepdims=True); rb = rb - rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def rankcorr(x, y):
    rx, ry = rankdata(x), rankdata(y); rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--out", default="bootstrap.csv"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm_dir", default=None)
    a = ap.parse_args()
    models, lms = [], []
    for d in a.ckpts:
        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]
        m = HierDEL(**cfg["model"]).to(a.device)
        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=a.device)); models.append(m)
        lms.append(LMTable(a.lm_dir, a.device) if cfg["model"].get("use_lm") else None)
    rng = np.random.default_rng(a.seed); rows = []
    for cond in ("on", "off"):
        full = load_heldout(a.root, a.target, cond)
        has = full["molecule_hash"].notna() & full[["smiles_a", "smiles_b", "smiles_c"]].notna().all(1)
        df = full[has].reset_index(drop=True); n = len(df); kd = df["kd"].to_numpy()
        S = [score(m, df, a.device, lm=l) for m, l in zip(models, lms)]               # moi phan tu: dict rule -> [n]
        idx = rng.integers(0, n, size=(a.B, n))                    # CUNG mot bo resample cho moi cach cham/ckpt
        for rule in S[0]:
            per = np.stack([s[rule] for s in S])                    # [K,n]
            mean_boot = np.mean([-spearman_rows(p[idx], kd[idx]) for p in per], axis=0)   # TB 5 split moi resample
            ens = per.mean(0); ens_boot = -spearman_rows(ens[idx], kd[idx])
            for name, boot, point in (("TB 5 split", mean_boot,
                                       np.mean([-rankcorr(p, kd) for p in per])),
                                      ("ensemble 5 model", ens_boot, -rankcorr(ens, kd))):
                lo, hi = np.percentile(boot, [2.5, 97.5])
                base = BASELINE[a.target][cond]
                rows.append(dict(target=a.target, cond=cond, n=n, rule=rule, agg=name, rho=point,
                                 ci_lo=lo, ci_hi=hi, baseline=base,
                                 baseline_in_ci=bool(lo <= base <= hi),
                                 p_below_baseline=float((boot < base).mean())))
    R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
