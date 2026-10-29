"""PA4 = PA3 + (1) encoder phan tu D-MPNN, (2) readout sum|mean|max, (3) token thu 8 = LM muc phan tu.
Moi thay doi la MOT CO TRONG CONFIG, mac dinh TAT -> checkpoint PA1/PA3 cu van nap duoc nguyen ven.
Chay nhieu lan khong sao (idempotent). Chay tu thu muc goc repo: python apply_pa4_patch.py
"""
import os, sys

E = []


def sub(path, old, new, tag):
    s = open(path).read()
    if new in s:
        print(f"  [bo qua] {tag} (da co)"); return
    if s.count(old) != 1:
        E.append(f"{tag}: tim thay {s.count(old)} lan, can dung 1"); return
    open(path, "w").write(s.replace(old, new)); print(f"  [da sua] {tag}")


# ------------------------------------------------------------------ kdel/hier.py
print("kdel/hier.py")
sub("kdel/hier.py",
    "from torch_geometric.nn import GINEConv, global_add_pool",
    "from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool, global_max_pool",
    "import them mean/max pool")

sub("kdel/hier.py",
    "class GINEEncoder(nn.Module):\n"
    "    def __init__(self, node_dim, edge_dim, hidden, n_layers, dropout):",
    'def readout_graph(h, batch, n, proj=None):\n'
    '    """proj=None -> chi tong (nhu PA1-PA3); nguoc lai noi [sum || mean || max] roi chieu ve H."""\n'
    '    if proj is None:\n'
    '        return global_add_pool(h, batch, size=n)\n'
    '    return proj(torch.cat([global_add_pool(h, batch, size=n),\n'
    '                           global_mean_pool(h, batch, size=n),\n'
    '                           global_max_pool(h, batch, size=n)], -1))\n'
    '\n'
    '\n'
    'class DMPNNEncoder(nn.Module):\n'
    '    """Truyen tin theo CANH CO HUONG (kieu Chemprop). featurize luu moi lien ket thanh hai\n'
    '       canh nguoc nhau LIEN TIEP va moi phan tu co so canh chan, nen canh nguoc cua canh k\n'
    '       trong batch luon la k^1."""\n'
    '    def __init__(self, node_dim, edge_dim, hidden, n_layers, dropout, readout="sum"):\n'
    '        super().__init__()\n'
    '        self.W_i = nn.Linear(node_dim + edge_dim, hidden)\n'
    '        self.W_h = nn.Linear(hidden, hidden)\n'
    '        self.W_o = nn.Linear(node_dim + hidden, hidden)\n'
    '        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])\n'
    '        self.drop = nn.Dropout(dropout)\n'
    '        self.hidden = hidden\n'
    '        self.out = nn.Linear(3 * hidden, hidden) if readout == "smm" else None\n'
    '\n'
    '    def forward(self, x, ei, ea, batch, n):\n'
    '        src, dst = ei[0], ei[1]\n'
    '        m = src.numel()\n'
    '        if m == 0:\n'
    '            S = x.new_zeros(x.shape[0], self.hidden)\n'
    '            hv = F.relu(self.W_o(torch.cat([x.to(S.dtype), S], -1)))\n'
    '            return readout_graph(hv, batch, n, self.out)\n'
    '        h0 = F.relu(self.W_i(torch.cat([x[src], ea], -1)))        # [m,H] tin tren canh\n'
    '        h, rev = h0, torch.arange(m, device=x.device) ^ 1\n'
    '        for bn in self.bns:\n'
    '            S = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=h.dtype)\n'
    '            S = S.index_add_(0, dst, h)                           # S[a] = tong canh di VAO a\n'
    '            h = F.relu(h0 + self.drop(bn(self.W_h(S[src] - h[rev]))))   # tru canh nguoc\n'
    '        S = torch.zeros(x.shape[0], self.hidden, device=x.device, dtype=h.dtype)\n'
    '        S = S.index_add_(0, dst, h)\n'
    '        hv = F.relu(self.W_o(torch.cat([x.to(S.dtype), S], -1)))\n'
    '        return readout_graph(hv, batch, n, self.out)\n'
    '\n'
    '\n'
    'class GINEEncoder(nn.Module):\n'
    '    def __init__(self, node_dim, edge_dim, hidden, n_layers, dropout, readout="sum"):',
    "them readout_graph + DMPNNEncoder")

sub("kdel/hier.py",
    "        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])\n"
    "        self.drop  = nn.Dropout(dropout)\n",
    "        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])\n"
    "        self.drop  = nn.Dropout(dropout)\n"
    '        self.out   = nn.Linear(3 * hidden, hidden) if readout == "smm" else None\n',
    "GINEEncoder: them projection readout")

sub("kdel/hier.py",
    "            h = h + self.drop(F.relu(bn(conv(h, ei, e))))      # residual\n"
    "        return global_add_pool(h, batch, size=n)",
    "            h = h + self.drop(F.relu(bn(conv(h, ei, e))))      # residual\n"
    "        return readout_graph(h, batch, n, self.out)",
    "GINEEncoder: dung readout_graph")

sub("kdel/hier.py",
    "    def __init__(self, node_dim=28, edge_dim=6, hidden=256, n_layers=5, syn_layers=3,\n"
    '                 dropout=0.2, heads=4, use_lm=False, lm_dim=384, fusion="add", aux_sizes=None):',
    "    def __init__(self, node_dim=28, edge_dim=6, hidden=256, n_layers=5, syn_layers=3,\n"
    '                 dropout=0.2, heads=4, use_lm=False, lm_dim=384, fusion="add", aux_sizes=None,\n'
    '                 mol_enc="gine", readout="sum", use_mol_lm=False, mol_lm_dim=384):',
    "HierDEL: them 4 tham so PA4")

sub("kdel/hier.py",
    "        self.use_lm, self.fusion = use_lm, fusion\n"
    "        self.syn_enc  = GINEEncoder(node_dim, edge_dim, H, syn_layers, dropout)\n"
    "        self.full_enc = GINEEncoder(node_dim, edge_dim, H, n_layers, dropout)",
    "        self.use_lm, self.fusion = use_lm, fusion\n"
    "        self.use_mol_lm = use_mol_lm\n"
    "        self.syn_enc  = GINEEncoder(node_dim, edge_dim, H, syn_layers, dropout, readout)\n"
    '        self.full_enc = (DMPNNEncoder(node_dim, edge_dim, H, n_layers, dropout, readout)\n'
    '                         if mol_enc == "dmpnn" else\n'
    '                         GINEEncoder(node_dim, edge_dim, H, n_layers, dropout, readout))\n'
    "        if use_mol_lm:\n"
    "            self.mol_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(mol_lm_dim, H))",
    "HierDEL: chon encoder phan tu + nhanh LM")

sub("kdel/hier.py",
    "        self.tok_type = nn.Parameter(torch.randn(7, H) * 0.02)",
    "        self.n_tok = 8 if use_mol_lm else 7\n"
    "        self.tok_type = nn.Parameter(torch.randn(self.n_tok, H) * 0.02)",
    "HierDEL: so token 7 hoac 8")

sub("kdel/hier.py",
    "        tok = torch.stack([zA, zB, zC,\n"
    "                           self.pair[0](torch.cat([zA, zB], -1)),\n"
    "                           self.pair[1](torch.cat([zA, zC], -1)),\n"
    "                           self.pair[2](torch.cat([zB, zC], -1)),\n"
    "                           self.tri(zf)], dim=1)  + self.tok_type        # [B,7,H]",
    "        toks = [zA, zB, zC,\n"
    "                self.pair[0](torch.cat([zA, zB], -1)),\n"
    "                self.pair[1](torch.cat([zA, zC], -1)),\n"
    "                self.pair[2](torch.cat([zB, zC], -1)),\n"
    "                self.tri(zf)]\n"
    "        if self.use_mol_lm:\n"
    "            toks.append(self.mol_proj(d.lm_mol))      # token 8: KHONG bi che boi synthon-dropout\n"
    "        tok = torch.stack(toks, dim=1) + self.tok_type        # [B,n_tok,H]",
    "HierDEL: dung danh sach token")

sub("kdel/hier.py",
    "            kpm = torch.zeros(B, 7, dtype=torch.bool, device=tok.device)",
    "            kpm = torch.zeros(B, self.n_tok, dtype=torch.bool, device=tok.device)",
    "mask rong theo n_tok")

# ------------------------------------------------------------------ kdel/gpu_data.py
print("kdel/gpu_data.py")
sub("kdel/gpu_data.py",
    '    def __init__(self, root, device="cuda"):',
    '    def __init__(self, root, device="cuda", mol_lm=None):',
    "GPUGraphStore: tham so mol_lm")

sub("kdel/gpu_data.py",
    '        self.w    = L("conf_weight_fp16.npy")\n'
    "        self.dev  = device",
    '        self.w    = L("conf_weight_fp16.npy")\n'
    "        self.dev  = device\n"
    "        self.lm = None\n"
    "        if mol_lm:                                    # embedding LM muc phan tu (PA4)\n"
    '            st = np.load(f"{mol_lm}/stats.npz")\n'
    '            self.lm = torch.from_numpy(np.load(f"{mol_lm}/mol_emb_fp16.npy")).to(device)\n'
    '            self.lm_mean = torch.from_numpy(st["mean"]).to(device)\n'
    '            self.lm_std  = torch.from_numpy(st["std"]).to(device)',
    "GPUGraphStore: nap bang embedding")

sub("kdel/gpu_data.py",
    "        B.idx = idx\n"
    "        return B",
    "        B.idx = idx\n"
    "        if self.lm is not None:\n"
    "            B.lm_mol = (self.lm[idx].float() - self.lm_mean) / self.lm_std\n"
    "        return B",
    "GPUGraphStore: tra embedding theo batch")

# ------------------------------------------------------------------ kdel/train.py
print("kdel/train.py")
sub("kdel/train.py",
    '    ap.add_argument("--lm_dir", default=None)',
    '    ap.add_argument("--lm_dir", default=None)\n'
    '    ap.add_argument("--mol_lm_dir", default=None)   # thu muc co mol_emb_fp16.npy + stats.npz (PA4)',
    "train: tham so --mol_lm_dir")

sub("kdel/train.py",
    "    store = GPUGraphStore(a.root, dev)",
    "    store = GPUGraphStore(a.root, dev, mol_lm=a.mol_lm_dir if M.get(\"use_mol_lm\") else None)",
    "train: truyen mol_lm vao store")

# ------------------------------------------------------------------ kdel/evaluate.py
print("kdel/evaluate.py")
sub("kdel/evaluate.py",
    "from kdel.hier import HierDEL, SynStore, LMTable",
    "from kdel.hier import HierDEL, SynStore, LMTable\n"
    "\n"
    "\n"
    "class MolLMTable:\n"
    '    """Embedding LM muc PHAN TU cho phan tu held-out; tra theo chuoi SMILES."""\n'
    '    def __init__(self, mol_lm_dir, device="cuda"):\n'
    '        d  = np.load(f"{mol_lm_dir}/heldout_emb.npz", allow_pickle=True)\n'
    '        st = np.load(f"{mol_lm_dir}/stats.npz")\n'
    '        self.idx = {s: i for i, s in enumerate(d["smiles"].tolist())}\n'
    '        Z = (d["emb"].astype(np.float32) - st["mean"]) / st["std"]\n'
    "        self.Z = torch.from_numpy(Z).to(device); self.dev = device\n"
    "\n"
    "    def lookup(self, smiles):\n"
    "        rows = [self.idx.get(s, -1) for s in smiles]\n"
    "        miss = sum(r < 0 for r in rows)\n"
    '        if miss: print(f"[MolLMTable] canh bao: {miss}/{len(rows)} phan tu thieu embedding -> vector 0", flush=True)\n'
    "        out = self.Z[torch.tensor([max(r, 0) for r in rows], device=self.dev)].clone()\n"
    "        out[torch.tensor([r < 0 for r in rows], device=self.dev)] = 0.0\n"
    "        return out",
    "them MolLMTable")

sub("kdel/evaluate.py",
    "def score(model, df, dev, bs=64, lm=None, drop_syn=False):",
    "def score(model, df, dev, bs=64, lm=None, drop_syn=False, mol_lm=None):",
    "score: tham so mol_lm")

sub("kdel/evaluate.py",
    '    smi = df["smiles"].tolist(); s_t, s_d = [], []',
    '    smi = df["smiles"].tolist(); s_t, s_d = [], []\n'
    "    ME = mol_lm.lookup(smi) if mol_lm is not None else None",
    "score: tra embedding phan tu")

sub("kdel/evaluate.py",
    "        dsm = torch.ones(b.syn.shape[0], dtype=torch.bool, device=dev) if drop_syn else None\n"
    "        o = model(b, syn, drop_syn=dsm)",
    "        dsm = torch.ones(b.syn.shape[0], dtype=torch.bool, device=dev) if drop_syn else None\n"
    "        if ME is not None: b.lm_mol = ME[i:i + bs]\n"
    "        o = model(b, syn, drop_syn=dsm)",
    "score: gan lm_mol cho batch")

sub("kdel/evaluate.py",
    "def eval_ckpt(model, root, target, dev, lm=None):",
    "def eval_ckpt(model, root, target, dev, lm=None, mol_lm=None):",
    "eval_ckpt: tham so mol_lm")

sub("kdel/evaluate.py",
    "            for name, sc in score(model, df, dev, lm=lm, drop_syn=dsyn).items():",
    "            for name, sc in score(model, df, dev, lm=lm, drop_syn=dsyn, mol_lm=mol_lm).items():",
    "eval_ckpt: truyen mol_lm")

sub("kdel/evaluate.py",
    '    ap.add_argument("--lm_dir", default=None)\n'
    "    a = ap.parse_args()",
    '    ap.add_argument("--lm_dir", default=None); ap.add_argument("--mol_lm_dir", default=None)\n'
    "    a = ap.parse_args()",
    "evaluate main: tham so --mol_lm_dir")

sub("kdel/evaluate.py",
    '        lm = LMTable(a.lm_dir, a.device) if cfg["model"].get("use_lm") else None\n'
    '        r = eval_ckpt(m, a.root, a.target, a.device, lm=lm); r["ckpt"] = os.path.basename(d.rstrip("/"))',
    '        lm = LMTable(a.lm_dir, a.device) if cfg["model"].get("use_lm") else None\n'
    '        ml = MolLMTable(a.mol_lm_dir, a.device) if cfg["model"].get("use_mol_lm") else None\n'
    '        r = eval_ckpt(m, a.root, a.target, a.device, lm=lm, mol_lm=ml); r["ckpt"] = os.path.basename(d.rstrip("/"))',
    "evaluate main: dung MolLMTable")

# ------------------------------------------------------------------ kdel/bootstrap.py
print("kdel/bootstrap.py")
sub("kdel/bootstrap.py",
    "from kdel.evaluate import load_heldout, score",
    "from kdel.evaluate import load_heldout, score, MolLMTable",
    "bootstrap: import MolLMTable")

sub("kdel/bootstrap.py",
    '    ap.add_argument("--lm_dir", default=None)\n'
    "    a = ap.parse_args()\n"
    "    models, lms = [], []",
    '    ap.add_argument("--lm_dir", default=None); ap.add_argument("--mol_lm_dir", default=None)\n'
    "    a = ap.parse_args()\n"
    "    models, lms, mls = [], [], []",
    "bootstrap: tham so + danh sach mls")

sub("kdel/bootstrap.py",
    '        lms.append(LMTable(a.lm_dir, a.device) if cfg["model"].get("use_lm") else None)\n'
    "    rng = np.random.default_rng(a.seed); rows = []",
    '        lms.append(LMTable(a.lm_dir, a.device) if cfg["model"].get("use_lm") else None)\n'
    '        mls.append(MolLMTable(a.mol_lm_dir, a.device) if cfg["model"].get("use_mol_lm") else None)\n'
    "    rng = np.random.default_rng(a.seed); rows = []",
    "bootstrap: nap MolLMTable")

sub("kdel/bootstrap.py",
    "            S = [score(m, df, a.device, lm=l, drop_syn=dsyn) for m, l in zip(models, lms)]",
    "            S = [score(m, df, a.device, lm=l, drop_syn=dsyn, mol_lm=g)\n"
    "                 for m, l, g in zip(models, lms, mls)]",
    "bootstrap: truyen mol_lm")

# ------------------------------------------------------------------ kdel/paired.py
print("kdel/paired.py")
sub("kdel/paired.py",
    "from kdel.evaluate import load_heldout, score",
    "from kdel.evaluate import load_heldout, score, MolLMTable",
    "paired: import MolLMTable")

sub("kdel/paired.py",
    "def load_set(pattern, dev, lm_dir):\n"
    '    dirs = sorted(d for d in glob.glob(pattern, recursive=True) if os.path.exists(f"{d}/best.pt"))\n'
    '    assert dirs, f"khong tim thay checkpoint: {pattern}"\n'
    "    models, lms = [], []\n"
    "    for d in dirs:\n"
    '        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]\n'
    "        m = HierDEL(**cfg[\"model\"]).to(dev)\n"
    '        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev)); models.append(m)\n'
    '        lms.append(LMTable(lm_dir, dev) if cfg["model"].get("use_lm") else None)\n'
    "    return models, lms, [os.path.basename(d) for d in dirs]",
    "def load_set(pattern, dev, lm_dir, mol_lm_dir=None):\n"
    '    dirs = sorted(d for d in glob.glob(pattern, recursive=True) if os.path.exists(f"{d}/best.pt"))\n'
    '    assert dirs, f"khong tim thay checkpoint: {pattern}"\n'
    "    models, lms, mls = [], [], []\n"
    "    for d in dirs:\n"
    '        cfg = torch.load(f"{d}/last.pt", map_location="cpu")["cfg"]\n'
    "        m = HierDEL(**cfg[\"model\"]).to(dev)\n"
    '        m.load_state_dict(torch.load(f"{d}/best.pt", map_location=dev)); models.append(m)\n'
    '        lms.append(LMTable(lm_dir, dev) if cfg["model"].get("use_lm") else None)\n'
    '        mls.append(MolLMTable(mol_lm_dir, dev) if cfg["model"].get("use_mol_lm") else None)\n'
    "    return models, lms, mls, [os.path.basename(d) for d in dirs]",
    "paired: load_set tra ve mls")

sub("kdel/paired.py",
    '    ap.add_argument("--lm_dir", default=None)',
    '    ap.add_argument("--lm_dir", default=None); ap.add_argument("--mol_lm_dir", default=None)',
    "paired: tham so --mol_lm_dir")

sub("kdel/paired.py",
    "        MA, LA, NA = load_set(a.a_pat.format(t=t), a.device, a.lm_dir)\n"
    "        MB, LB, NB = load_set(a.b_pat.format(t=t), a.device, a.lm_dir)",
    "        MA, LA, GA, NA = load_set(a.a_pat.format(t=t), a.device, a.lm_dir, a.mol_lm_dir)\n"
    "        MB, LB, GB, NB = load_set(a.b_pat.format(t=t), a.device, a.lm_dir, a.mol_lm_dir)",
    "paired: nhan them mls")

sub("kdel/paired.py",
    "                SA = [score(m, df, a.device, lm=l, drop_syn=dsyn) for m, l in zip(MA, LA)]\n"
    "                SB = [score(m, df, a.device, lm=l, drop_syn=dsyn) for m, l in zip(MB, LB)]",
    "                SA = [score(m, df, a.device, lm=l, drop_syn=dsyn, mol_lm=g) for m, l, g in zip(MA, LA, GA)]\n"
    "                SB = [score(m, df, a.device, lm=l, drop_syn=dsyn, mol_lm=g) for m, l, g in zip(MB, LB, GB)]",
    "paired: truyen mol_lm")

# ------------------------------------------------------------------ configs/pa4.yaml
print("configs/pa4.yaml")
CFG = """name: pa4c_dmpnn_mollm
model:
  node_dim: 28
  edge_dim: 6
  hidden: 256
  n_layers: 5
  syn_layers: 3
  heads: 4
  dropout: 0.2
  mol_enc: dmpnn          # (1) encoder phan tu theo canh co huong
  readout: smm            # (2) readout [sum || mean || max]
  use_mol_lm: true        # (3) token thu 8 = embedding LM muc phan tu
  mol_lm_dim: 384
train:
  batch_size: 256
  accum: 4
  lr: 3.0e-4
  weight_decay: 1.0e-5
  warmup_steps: 300
  epochs: 20
  beta: 1.0
  zero_inflated: false
  patience: 4
  save_every: 500
  val_size: 100000
  wall_limit_hours: 11.0
  syn_drop: 0.3
"""
if os.path.exists("configs/pa4.yaml") and open("configs/pa4.yaml").read() == CFG:
    print("  [bo qua] configs/pa4.yaml (da co)")
else:
    open("configs/pa4.yaml", "w").write(CFG); print("  [da tao] configs/pa4.yaml")

if E:
    print("\nLOI, KHONG SUA GI THEM:"); [print(" -", x) for x in E]; sys.exit(1)
print("\nXONG")
