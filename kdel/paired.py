import argparse, glob, os
import numpy as np, pandas as pd, torch
from kdel.hier import HierDEL, LMTable
from kdel.evaluate import load_heldout, score, MolLMTable
from kdel.bootstrap import spearman_rows


def load_set(pattern, dev, lm_dir, mol_lm_dir=None):
    dirs = sorted(d for d in glob.glob(pattern, recursive=True) if os.path.exists(f"{d}/best.pt"))
    assert dirs, f"khong tim thay checkpoint: {pattern}"
    models, lms, mls = [], [], []
    for d in dirs:
        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]
        m = HierDEL(**cfg["model"]).to(dev)
        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev)); models.append(m)
        lms.append(LMTable(lm_dir, dev) if cfg["model"].get("use_lm") else None)
        mls.append(MolLMTable(mol_lm_dir, dev) if cfg["model"].get("use_mol_lm") else None)
    return models, lms, mls, [os.path.basename(d) for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_pat", required=True)
    ap.add_argument("--b_pat", required=True)
    ap.add_argument("--root_pat", required=True)
    ap.add_argument("--lm_dir", default=None); ap.add_argument("--mol_lm_dir", default=None)
    ap.add_argument("--targets", nargs="+", default=["ddr1", "mapk14"])
    ap.add_argument("--rules", nargs="+", default=["logλ_target", "logλ_t−logλ_c"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="paired.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); rows = []; pool = {}
    for t in a.targets:
        root = os.path.dirname(glob.glob(a.root_pat.format(t=t), recursive=True)[0])
        mld = a.mol_lm_dir.format(t=t) if a.mol_lm_dir else None   # cho phep mau co {t}
        MA, LA, GA, NA = load_set(a.a_pat.format(t=t), a.device, a.lm_dir, mld)
        MB, LB, GB, NB = load_set(a.b_pat.format(t=t), a.device, a.lm_dir, mld)
        print(f"[{t}] A: {len(MA)} ckpt | B: {len(MB)} ckpt", flush=True)
        for cond in ("on", "off"):
            full = load_heldout(root, t, cond)
            has = full["molecule_hash"].notna() & full[["smiles_a", "smiles_b", "smiles_c"]].notna().all(1)
            jobs = [("in-library", full[has], False), ("in-library-nosyn", full[has], True)]
            if not has.all():
                jobs.append(("extended", full, True))
            for lib, dfj, dsyn in jobs:
                df = dfj.reset_index(drop=True); n = len(df); kd = df["kd"].to_numpy()
                if n < 5: continue
                SA = [score(m, df, a.device, lm=l, drop_syn=dsyn, mol_lm=g) for m, l, g in zip(MA, LA, GA)]
                SB = [score(m, df, a.device, lm=l, drop_syn=dsyn, mol_lm=g) for m, l, g in zip(MB, LB, GB)]
                idx = rng.integers(0, n, size=(a.B, n))
                for rule in a.rules:
                    bootA = np.mean([-spearman_rows(s[rule][idx], kd[idx]) for s in SA], axis=0)
                    bootB = np.mean([-spearman_rows(s[rule][idx], kd[idx]) for s in SB], axis=0)
                    d = bootB - bootA
                    pa = float(np.mean([-spearman_rows(s[rule][None], kd[None])[0] for s in SA]))
                    pb = float(np.mean([-spearman_rows(s[rule][None], kd[None])[0] for s in SB]))
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    rows.append(dict(target=t, cond=cond, lib=lib, n=n, rule=rule, A=pa, B=pb,
                                     diff=pb - pa, ci_lo=lo, ci_hi=hi,
                                     p_B_not_better=float((d <= 0).mean()), sig=bool(lo > 0 or hi < 0)))
                    pool.setdefault((lib, rule), []).append(d)
    for (lib, rule), ds in pool.items():
        D = np.mean(ds, axis=0); lo, hi = np.percentile(D, [2.5, 97.5])
        dm = float(np.mean([r["diff"] for r in rows if r["rule"] == rule and r["lib"] == lib
                            and r["target"] != "TB cac o"]))
        rows.append(dict(target="TB cac o", cond="-", lib=lib, n=0, rule=rule, A=np.nan, B=np.nan,
                         diff=dm, ci_lo=lo, ci_hi=hi,
                         p_B_not_better=float((D <= 0).mean()), sig=bool(lo > 0 or hi < 0)))
    R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
