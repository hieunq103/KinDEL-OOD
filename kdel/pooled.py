"""Phep kiem DA CHOT TRUOC (xem claude/pa1_results.md): gop hai o Extended on-DNA
(DDR1 n=54, MAPK14 n=41), quy tac chinh logλ_target, so voi moc GOP cua bai bao
(0.598 DNN + 0.421 DEL-Compose(M)) / 2 = 0.5095.

Moi target duoc lay mau lai DOC LAP (hai tap held-out doc lap), thong ke gop la
trung binh hai o tren moi lan lay mau -> CI hep hon khoang √2 lan so voi mot o.
"""
import argparse, glob, os
import numpy as np, pandas as pd, torch
from kdel.hier import HierDEL, LMTable
from kdel.evaluate import load_heldout, score, MolLMTable
from kdel.bootstrap import spearman_rows, rankcorr

PAPER = {"ddr1": 0.598, "mapk14": 0.421}      # Extended on-DNA, split disynthon, mo hinh ML tot nhat


def load_models(pattern, dev, lm_dir, mol_lm_dir):
    dirs = sorted(d for d in glob.glob(pattern, recursive=True) if os.path.exists(f"{d}/best.pt"))
    assert dirs, f"khong tim thay checkpoint: {pattern}"
    out = []
    for d in dirs:
        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]
        m = HierDEL(**cfg["model"]).to(dev)
        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev))
        lm = LMTable(lm_dir, dev) if cfg["model"].get("use_lm") else None
        ml = MolLMTable(mol_lm_dir, dev) if cfg["model"].get("use_mol_lm") else None
        out.append((m, lm, ml))
    return out, [os.path.basename(d) for d in dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_pat", required=True)      # co {t}
    ap.add_argument("--root_pat", required=True)      # duong dan toi {t}_ondna.csv, co {t}
    ap.add_argument("--mol_lm_pat", default=None)     # thu muc embedding, co {t}
    ap.add_argument("--lm_dir", default=None)
    ap.add_argument("--targets", nargs="+", default=["ddr1", "mapk14"])
    ap.add_argument("--rule", default="logλ_target")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--B", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="pooled.csv")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    boots, rows = [], []
    for t in a.targets:
        root = os.path.dirname(glob.glob(a.root_pat.format(t=t), recursive=True)[0])
        ml_dir = a.mol_lm_pat.format(t=t) if a.mol_lm_pat else None
        M, names = load_models(a.ckpt_pat.format(t=t), a.device, a.lm_dir, ml_dir)
        df = load_heldout(root, t, "on").reset_index(drop=True)      # Extended = TOAN BO tap on-DNA
        n = len(df); kd = df["kd"].to_numpy()
        S = [score(m, df, a.device, lm=lm, drop_syn=True, mol_lm=mm)[a.rule] for m, lm, mm in M]
        idx = rng.integers(0, n, size=(a.B, n))
        boot = np.mean([-spearman_rows(s[idx], kd[idx]) for s in S], axis=0)
        point = float(np.mean([-rankcorr(s, kd) for s in S]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(o=t, n=n, k=len(M), rho=point, ci_lo=lo, ci_hi=hi, moc=PAPER[t],
                         moc_trong_ci=bool(lo <= PAPER[t] <= hi),
                         p_duoi_moc=float((boot < PAPER[t]).mean())))
        boots.append(boot)
        print(f"[{t}] n={n} ckpt={len(M)} rho={point:.4f} CI=[{lo:.3f},{hi:.3f}] moc={PAPER[t]}", flush=True)

    D = np.mean(boots, axis=0)
    base = float(np.mean([PAPER[t] for t in a.targets]))
    point = float(np.mean([r["rho"] for r in rows]))
    lo, hi = np.percentile(D, [2.5, 97.5])
    rows.append(dict(o="GOP (phep kiem da chot)", n=0, k=0, rho=point, ci_lo=lo, ci_hi=hi, moc=base,
                     moc_trong_ci=bool(lo <= base <= hi), p_duoi_moc=float((D < base).mean())))
    R = pd.DataFrame(rows); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 220)
    print("\n" + R.round(4).to_string(index=False))
    r = rows[-1]
    print(f"\nKET LUAN: rho gop {r['rho']:.4f} vs moc {base:.4f} | CI95 [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
          f" | p mot phia = {r['p_duoi_moc']:.4f} | VUOT CO Y NGHIA: {r['ci_lo'] > base}")


if __name__ == "__main__":
    main()
