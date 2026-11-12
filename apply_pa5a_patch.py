"""PA5-A: tham so hoa phan du logL_t = logL_c + Delta ; diem xep hang = Delta ; (tuy chon) phat L2 tren Delta. Idempotent."""
import pathlib
R = pathlib.Path(".")
def sub(path, old, new, tag):
    p = R / path; s = p.read_text()
    if tag in s: print(f"[skip] {path} ({tag})"); return
    assert old in s, f"khong thay doan can va trong {path}: {old[:40]}"
    p.write_text(s.replace(old, new, 1)); print(f"[ok] {path} ({tag})")

# model
sub("kdel/hier.py", "use_mol_lm=False, mol_lm_dim=384", "use_mol_lm=False, mol_lm_dim=384, residual=False", "residual=False")
sub("kdel/hier.py", "        self.score = mlp(H, H, 2)       # log lambda: target, control\n",
    "        self.score = mlp(H, H, 2)       # log lambda: target, control\n"
    "        self.residual = residual        # PA5A: dau 0 = Delta (khoi tao 0), dau 1 = logL_c ; logL_t = logL_c + Delta\n"
    "        if residual:\n"
    "            nn.init.zeros_(self.score[-1].weight[0]); nn.init.zeros_(self.score[-1].bias[0])\n",
    "self.residual = residual")
sub("kdel/hier.py", 'out = {"log_lambda": self.score(z),',
    'll = self.score(z)\n'
    '        if self.residual: ll = torch.stack([ll[:, 1] + ll[:, 0], ll[:, 1]], -1)   # PA5A\n'
    '        out = {"log_lambda": ll,', "if self.residual: ll")
# loss
sub("kdel/losses.py", "def total_loss(out, b, beta=1.0, zero_inflated=True):",
    "def total_loss(out, b, beta=1.0, zero_inflated=True, delta_l2=0.0):  # PA5A", "delta_l2=0.0")
sub("kdel/losses.py", '    return l_t + beta*l_c, {"target": l_t.item(), "control": l_c.item()}',
    '    loss = l_t + beta*l_c\n'
    '    if delta_l2 > 0: loss = loss + delta_l2 * (ll_t - ll_c).pow(2).mean()   # PA5A: phat L2 tren Delta\n'
    '    return loss, {"target": l_t.item(), "control": l_c.item()}', "if delta_l2 > 0")
sub("kdel/train.py", 'total_loss(out, b, beta=T["beta"], zero_inflated=zi)',
    'total_loss(out, b, beta=T["beta"], zero_inflated=zi, delta_l2=float(T.get("delta_l2", 0.0)))', 'delta_l2=float(')
y = R / "configs/pa5a.yaml"
if not y.exists():
    s = (R / "configs/pa4.yaml").read_text()
    s = s.replace("name: pa4c_dmpnn_mollm", "name: pa5a_dmpnn_mollm_resid")
    s = s.replace("  mol_lm_dim: 384\n", "  mol_lm_dim: 384\n  residual: true          # PA5-A: logL_t = logL_c + Delta\n")
    s = s.replace("  syn_drop: 0.3", "  syn_drop: 0.3\n  delta_l2: 0.05          # phat L2 tren Delta (co dinh truoc, chua tune)")
    y.write_text(s); print("[ok] configs/pa5a.yaml")
else: print("[skip] configs/pa5a.yaml")
