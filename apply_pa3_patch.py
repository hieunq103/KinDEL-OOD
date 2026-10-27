"""PA3 = PA1 + synthon-dropout. Chay nhieu lan khong sao (idempotent)."""
import sys

E = []
def sub(path, old, new, tag):
    s = open(path).read()
    if new in s: print(f"  [bo qua] {tag} (da co)"); return
    if s.count(old) != 1:
        E.append(f"{tag}: tim thay {s.count(old)} lan, can dung 1"); return
    open(path, "w").write(s.replace(old, new)); print(f"  [da sua] {tag}")

print("kdel/hier.py")
sub("kdel/hier.py",
    "    def forward(self, d, syn_store):\n        B = d.syn.shape[0]",
    "    def forward(self, d, syn_store, drop_syn=None):\n        B = d.syn.shape[0]",
    "forward nhan drop_syn")
sub("kdel/hier.py",
    "        tok = self.tf(tok)\n"
    "        z, _ = self.attn(self.q.expand(B, -1, -1), tok, tok, need_weights=False)",
    "        kpm = None\n"
    "        if drop_syn is not None:              # che 6 token synthon -> chi con token ABC\n"
    "            kpm = torch.zeros(B, 7, dtype=torch.bool, device=tok.device)\n"
    "            kpm[:, :6] = drop_syn.view(-1, 1)\n"
    "        tok = self.tf(tok, src_key_padding_mask=kpm)\n"
    "        z, _ = self.attn(self.q.expand(B, -1, -1), tok, tok, need_weights=False, key_padding_mask=kpm)",
    "mask token synthon")

print("kdel/train.py")
sub("kdel/train.py",
    '    lam_rank, lam_aux = float(T.get("lambda_rank", 0.0)), float(T.get("lambda_aux", 0.0))',
    '    lam_rank, lam_aux = float(T.get("lambda_rank", 0.0)), float(T.get("lambda_aux", 0.0))\n'
    '    p_sdrop = float(T.get("syn_drop", 0.0))      # xac suat xoa token synthon (PA3)',
    "doc syn_drop tu config")
sub("kdel/train.py",
    '            b = store.batch(perm[i:i + bs]); b.syn = synidx[b.idx]\n'
    '            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):\n'
    '                out = model(b, syn)',
    '            b = store.batch(perm[i:i + bs]); b.syn = synidx[b.idx]\n'
    '            ds = (torch.rand(b.syn.shape[0], device=dev) < p_sdrop) if p_sdrop > 0 else None\n'
    '            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):\n'
    '                out = model(b, syn, drop_syn=ds)',
    "lay mau drop_syn moi batch")
sub("kdel/train.py",
    '                lg = out["aux"].split(sizes, dim=1)\n'
    '                la_ = sum(F.cross_entropy(lg[k], b.syn[:, k] - offs[k]) for k in range(3)) / 3',
    '                lg = out["aux"].split(sizes, dim=1)\n'
    '                kp = slice(None) if ds is None else ~ds        # bo mau da xoa synthon khoi loss phu\n'
    '                la_ = sum(F.cross_entropy(lg[k][kp], b.syn[kp, k] - offs[k]) for k in range(3)) / 3',
    "loss phu bo qua mau da xoa synthon")

print("kdel/evaluate.py")
sub("kdel/evaluate.py",
    'def score(model, df, dev, bs=64, lm=None):',
    'def score(model, df, dev, bs=64, lm=None, drop_syn=False):',
    "score nhan drop_syn")
sub("kdel/evaluate.py",
    '    cols = ["smiles_a", "smiles_b", "smiles_c"]\n'
    '    uniq = pd.unique(df[cols].to_numpy().ravel()).tolist()\n'
    '    mp = {s: i for i, s in enumerate(uniq)}\n'
    '    ids = np.stack([df[c].map(mp).to_numpy() for c in cols], 1)\n'
    '    syn = SynStore(uniq, dev, lm=lm)',
    '    cols = ["smiles_a", "smiles_b", "smiles_c"]\n'
    '    if drop_syn:                      # khong dung synthon: do thi rong + che token (cham duoc phan tu ngoai DEL)\n'
    '        ids = np.zeros((len(df), 3), dtype=np.int64); syn = SynStore([""], dev, lm=lm)\n'
    '    else:\n'
    '        uniq = pd.unique(df[cols].to_numpy().ravel()).tolist()\n'
    '        mp = {s: i for i, s in enumerate(uniq)}\n'
    '        ids = np.stack([df[c].map(mp).to_numpy() for c in cols], 1)\n'
    '        syn = SynStore(uniq, dev, lm=lm)',
    "score: nhanh khong synthon")
sub("kdel/evaluate.py",
    '        b = to_batch(smi[i:i + bs], ids[i:i + bs], dev)\n'
    '        o = model(b, syn)',
    '        b = to_batch(smi[i:i + bs], ids[i:i + bs], dev)\n'
    '        dsm = torch.ones(b.syn.shape[0], dtype=torch.bool, device=dev) if drop_syn else None\n'
    '        o = model(b, syn, drop_syn=dsm)',
    "score: truyen drop_syn")

OLD = '''def eval_ckpt(model, root, target, dev, lm=None):
    rows = []
    for cond in ("on", "off"):
        full = load_heldout(root, target, cond)
        has = full["molecule_hash"].notna() & full[["smiles_a", "smiles_b", "smiles_c"]].notna().all(1)
        sets = {"in-library": full[has]}
        sets["extended"] = full if has.all() else None          # co phan tu ngoai DEL -> khong co synthon
        for lib, df in sets.items():
            if df is None or len(df) < 5:
                rows.append(dict(cond=cond, lib=lib, n=int((~has).sum() + has.sum()) if df is None else len(df),
                                 score="n/a (phan tu ngoai DEL khong co synthon)", rho=np.nan)); continue
            for name, sc in score(model, df.reset_index(drop=True), dev, lm=lm).items():
                rho = spearmanr(sc, df["kd"].to_numpy()).statistic
                rows.append(dict(cond=cond, lib=lib, n=len(df), score=name, rho=-rho))   # -rho: dau theo bai bao
    return pd.DataFrame(rows)'''
NEW = '''def eval_ckpt(model, root, target, dev, lm=None):
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
            for name, sc in score(model, df, dev, lm=lm, drop_syn=dsyn).items():
                rho = spearmanr(sc, df["kd"].to_numpy()).statistic
                rows.append(dict(cond=cond, lib=lib, n=len(df), score=name, rho=-rho))   # -rho: dau theo bai bao
    return pd.DataFrame(rows)'''
sub("kdel/evaluate.py", OLD, NEW, "eval_ckpt: them o nosyn + extended")

if E:
    print("\nLOI, KHONG SUA GI THEM:"); [print(" -", e) for e in E]; sys.exit(1)
print("\nXONG")
