import argparse, glob, os
import numpy as np, pandas as pd, torch
from kdel.hier import HierDEL, LMTable
from kdel.evaluate import load_heldout, score
from kdel.bootstrap import spearman_rows


def load_set(pattern, dev, lm_dir):
    dirs = sorted(d for d in glob.glob(pattern) if os.path.exists(f"{d}/best.pt"))
    assert dirs, f"khong tim thay checkpoint: {pattern}"
    models, lms = [], []
    for d in dirs:
        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]
        m = HierDEL(**cfg["model"]).to(dev)
        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev)); models.append(m)
        lms.append(LMTable(lm_dir, dev) if cfg["model"].get("use_lm") else None)
    return models, lms, [os.path.basename(d) for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_pat", required=True)       # mo hinh A (doi chung), co {t}
    ap.add_argument("--b_pat", required=True)       # mo hinh B (de xuat), co {t}
    ap.add_argument("--root_pat", required=True)    # duong dan toi file {t}_ondna.csv, co {t}
    ap.add_argument("--lm_dir", default=None)
    ap.add_argument("--targets", nargs="+", default=["ddr1", "mapk14"])
    ap.add_argument("--rules", nargs="+", default=["logλ_target", "logλ_t−logλ_c"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="paired.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); rows = []; pool = {r: [] for r in a.rules}
    for t in a.targets:
        root = os.path.dirname(glob.glob(a.root_pat.format(t=t), recursive=True)[0])
        MA, LA, NA = load_set(a.a_pat.format(t=t), a.device, a.lm_dir)
        MB, LB, NB = load_set(a.b_pat.format(t=t), a.device, a.lm_dir)
        print(f"[{t}] A: {len(MA)} ckpt | B: {len(MB)} ckpt", flush=True)
        for cond in ("on", "off"):
            full = load_heldout(root, t, cond)
            has = full["molecule_hash"].notna() & full[["smiles_a", "smiles_b", "smiles_c"]].notna().all(1)
            df = full[has].reset_index(drop=True); n = len(df); kd = df["kd"].to_numpy()
            SA = [score(m, df, a.device, lm=l) for m, l in zip(MA, LA)]
            SB = [score(m, df, a.device, lm=l) for m, l in zip(MB, LB)]
            idx = rng.integers(0, n, size=(a.B, n))                 # CUNG resample cho A va B -> ghep cap
            for rule in a.rules:
                bootA = np.mean([-spearman_rows(s[rule][idx], kd[idx]) for s in SA], axis=0)
                bootB = np.mean([-spearman_rows(s[rule][idx], kd[idx]) for s in SB], axis=0)
                d = bootB - bootA
                pa = float(np.mean([-spearman_rows(s[rule][None], kd[None])[0] for s in SA]))
                pb = float(np.mean([-spearman_rows(s[rule][None], kd[None])[0] for s in SB]))
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append(dict(target=t, cond=cond, n=n, rule=rule, A=pa, B=pb, diff=pb - pa,
                                 ci_lo=lo, ci_hi=hi, p_B_not_better=float((d <= 0).mean()),
                                 sig=bool(lo > 0 or hi < 0)))
                pool[rule].append(d)
    # gop 4 o (cac tap held-out doc lap -> resample doc lap, lay TB theo o)
    for rule, ds in pool.items():
        D = np.mean(ds, axis=0); lo, hi = np.percentile(D, [2.5, 97.5])
        rows.append(dict(target="TB cac o", cond="-", n=0, rule=rule,
                         A=np.nan, B=np.nan, diff=float(np.mean([r["diff"] for r in rows if r["rule"] == rule and r["target"] != "TB cac o"])),
                         ci_lo=lo, ci_hi=hi, p_B_not_better=float((D <= 0).mean()), sig=bool(lo > 0 or hi < 0)))
    R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
