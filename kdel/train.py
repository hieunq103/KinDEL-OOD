import argparse, os, time, math, json, yaml
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from kdel.gpu_data import GPUGraphStore
from kdel.hier import HierDEL, SynStore, LMTable
from kdel.losses import nb_nll, total_loss, rank_loss


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp); os.replace(tmp, path)


def lr_at(step, total, base, warm):
    if step < warm: return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def evaluate(model, store, syn, synidx, idx, zi, bs=1024):
    model.eval()
    nll_t = nll_c = 0.0; n = 0; preds = []; ys = []
    for i in range(0, len(idx), bs):
        b = store.batch(idx[i:i + bs]); b.syn = synidx[b.idx]
        with torch.autocast("cuda", dtype=torch.float16, enabled=store.dev == "cuda"):
            out = model(b, syn)
        out = {k: v.float() for k, v in out.items()}
        pt = pc = None
        ll_t, ll_c = out["log_lambda"].unbind(-1)
        d_t, d_c = out["log_disp"].unbind(-1)
        if zi: pt, pc = out["pi_logit"].unbind(-1)
        k = len(b.pre)
        nll_t += nb_nll(b.target_counts, ll_t, d_t, b.pre, pt).sum().item()
        nll_c += nb_nll(b.control_counts, ll_c, d_c, b.pre, pc).sum().item()
        n += k; preds.append(ll_t.cpu())
        ys.append(torch.log((b.target_counts.float().mean(1) + 0.5) / b.pre).cpu())
    model.train()
    rho = spearmanr(torch.cat(preds).numpy(), torch.cat(ys).numpy())[0]
    return nll_t / n, nll_c / n, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)          # thu muc Dataset (co graphs_*.npy ...)
    ap.add_argument("--split", type=int, default=1)   # 1..5
    ap.add_argument("--config", default="configs/pa1.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=0)   # >0: chay thu
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm_dir", default=None)   # thu muc co synthon_chemberta.npz (bat buoc neu model.use_lm)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config)); M, T = cfg["model"], cfg["train"]
    os.makedirs(a.out, exist_ok=True)
    dev = a.device; t_start = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    store = GPUGraphStore(a.root, dev)
    voc = pd.read_parquet(f"{a.root}/synthon_vocab.parquet").sort_values("id")
    assert (voc["id"].to_numpy() == np.arange(len(voc))).all()
    lm = LMTable(a.lm_dir, dev) if M.get("use_lm") else None
    if M.get("use_lm"): M["lm_dim"] = lm.dim
    syn = SynStore(voc["smiles"].tolist(), dev, lm=lm)
    # loss phu doan ID synthon: kich thuoc theo tung vi tri A/B/C (id co offset theo vi tri)
    g = voc.groupby("pos")["id"].agg(["min", "count"]).loc[["A", "B", "C"]]
    offs = torch.tensor(g["min"].to_numpy(), device=dev); sizes = g["count"].tolist()
    lam_rank, lam_aux = float(T.get("lambda_rank", 0.0)), float(T.get("lambda_aux", 0.0))
    p_sdrop = float(T.get("syn_drop", 0.0))      # xac suat xoa token synthon (PA3)
    if lam_aux > 0: M["aux_sizes"] = sizes
    synidx = torch.from_numpy(np.load(f"{a.root}/synthon_idx.npy").astype(np.int64)).to(dev)
    S = np.load(f"{a.root}/splits_disynthon.npy")[:, a.split - 1]
    tr = torch.from_numpy(np.where(S == 0)[0]); va = torch.from_numpy(np.where(S == 1)[0])
    va = va[torch.randperm(len(va), generator=torch.Generator().manual_seed(123))[:T["val_size"]]]
    zi = bool(T["zero_inflated"])

    model = HierDEL(**M).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    bs, acc = T["batch_size"], T["accum"]
    spe = (len(tr) // bs) // acc                      # optimizer step / epoch
    total = spe * T["epochs"]
    st = dict(epoch=0, mb=0, step=0, best=1e9, bad=0)
    last, best_p = f"{a.out}/last.pt", f"{a.out}/best.pt"
    if os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"]); st = ck["st"]
        print(f"RESUME epoch {st['epoch']} mb {st['mb']} step {st['step']}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M | train {len(tr)} | val {len(va)} "
          f"| steps/epoch {spe} | total {total}", flush=True)

    def save():
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(),
                         scaler=scaler.state_dict(), st=st, cfg=cfg), last)

    logf = open(f"{a.out}/log.jsonl", "a")
    model.train(); t0 = time.time(); seen = 0; run = [0.0, 0.0, 0, 0.0, 0.0]
    stop = False
    while st["epoch"] < T["epochs"] and not stop:
        g = torch.Generator().manual_seed(a.seed * 1000 + st["epoch"])
        perm = tr[torch.randperm(len(tr), generator=g)]
        nmb = spe * acc
        while st["mb"] < nmb:
            if st["mb"] % acc == 0:
                for gr in opt.param_groups: gr["lr"] = lr_at(st["step"], total, T["lr"], T["warmup_steps"])
            i = st["mb"] * bs
            b = store.batch(perm[i:i + bs]); b.syn = synidx[b.idx]
            ds = (torch.rand(b.syn.shape[0], device=dev) < p_sdrop) if p_sdrop > 0 else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                out = model(b, syn, drop_syn=ds)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = total_loss(out, b, beta=T["beta"], zero_inflated=zi)
            if lam_rank > 0:
                S_ = b.target_counts.sum(1).float()
                y_ = torch.log((b.target_counts.float().mean(1) + 0.5) / b.pre)
                w_ = b.weight * (S_ / (S_ + 3.0))                  # phan tu it read -> trong so ~0
                lr_ = rank_loss(out["log_lambda"][:, 0], y_, w_, T.get("rank_margin", 0.5))
                loss = loss + lam_rank * lr_; parts["rank"] = lr_.item()
            if lam_aux > 0:
                lg = out["aux"].split(sizes, dim=1)
                kp = slice(None) if ds is None else ~ds        # bo mau da xoa synthon khoi loss phu
                la_ = sum(F.cross_entropy(lg[k][kp], b.syn[kp, k] - offs[k]) for k in range(3)) / 3
                loss = loss + lam_aux * la_; parts["aux"] = la_.item()
            scaler.scale(loss / acc).backward()
            st["mb"] += 1; seen += bs; run[0] += parts["target"]; run[1] += parts["control"]; run[2] += 1
            run[3] += parts.get("rank", 0.0); run[4] += parts.get("aux", 0.0)
            if st["mb"] % acc == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                st["step"] += 1
                if st["step"] % 100 == 0:
                    dt = time.time() - t0
                    rec = dict(step=st["step"], epoch=st["epoch"], nll_t=run[0] / run[2], nll_c=run[1] / run[2], rank=run[3] / run[2], aux=run[4] / run[2],
                               lr=opt.param_groups[0]["lr"], mol_s=seen / dt,
                               vram=torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0)
                    print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
                    run = [0.0, 0.0, 0, 0.0, 0.0]; t0 = time.time(); seen = 0
                if st["step"] % T["save_every"] == 0: save()
                if a.max_steps and st["step"] >= a.max_steps: stop = True; break
                if (time.time() - t_start) / 3600 > T["wall_limit_hours"]:
                    save(); print("WALL LIMIT -> da luu checkpoint, chay lai de resume", flush=True); return
        if stop: break
        st["epoch"] += 1; st["mb"] = 0
        vt, vc, rho = evaluate(model, store, syn, synidx, va, zi)
        rec = dict(epoch=st["epoch"], val_nll_t=vt, val_nll_c=vc, val_rho_proxy=rho)
        print("VAL", json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if vt < st["best"] - 1e-4:
            st["best"] = vt; st["bad"] = 0; atomic_save(model.state_dict(), best_p)
        else:
            st["bad"] += 1
        save()
        if st["bad"] >= T["patience"]:
            print("EARLY STOP", flush=True); break
        t0 = time.time(); seen = 0
    if a.max_steps:
        vt, vc, rho = evaluate(model, store, syn, synidx, va, zi)
        print("VAL(smoke)", vt, vc, rho)
    save(); print("DONE", flush=True)


if __name__ == "__main__":
    main()
