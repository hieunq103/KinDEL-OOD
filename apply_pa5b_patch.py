"""PA5-B: phan ra cong tinh trong khong gian toc do (DEL-Dock, Shmilovich et al. arXiv:2212.00136).

    lambda_control = lambda_matrix                  (chi bam nen / matrix)
    lambda_target  = lambda_matrix + lambda_spec    (nen + bam dac hieu)
    diem xep hang  = log lambda_spec

Khac PA5-A: PA5-A cong trong khong gian log (tuc NHAN toc do); PA5-B cong trong khong gian
toc do, dung voi viec so dem target la tong cua hai qua trinh. Idempotent.
"""
import pathlib
R = pathlib.Path(".")

def sub(path, old, new, tag):
    p = R / path; s = p.read_text()
    if tag in s: print(f"[skip] {path} ({tag})"); return
    assert old in s, f"khong thay doan can va trong {path}: {old[:60]!r}"
    p.write_text(s.replace(old, new, 1)); print(f"[ok] {path} ({tag})")

# --- 1. model ---------------------------------------------------------------
sub("kdel/hier.py",
    "use_mol_lm=False, mol_lm_dim=384, residual=False",
    "use_mol_lm=False, mol_lm_dim=384, residual=False, additive=False",
    "additive=False")

sub("kdel/hier.py",
    "        self.residual = residual",
    "        self.additive = additive      # PA5B: lambda_t = lambda_matrix + lambda_spec\n"
    "        assert not (residual and additive), 'residual va additive loai tru nhau'\n"
    "        self.residual = residual",
    "self.additive = additive")

sub("kdel/hier.py",
    '        out = {"log_lambda": ll, "pi_logit": self.zero(z),',
    '        sp = None\n'
    '        if self.additive:                                   # PA5B\n'
    '            sp, mx = ll[:, 0], ll[:, 1]                      # log lambda_spec, log lambda_matrix\n'
    '            ll = torch.stack([torch.logaddexp(mx, sp), mx], -1)\n'
    '        out = {"log_lambda": ll, "pi_logit": self.zero(z),',
    "if self.additive:                                   # PA5B")

sub("kdel/hier.py",
    '               "log_disp": self.log_disp, "z": z}',
    '               "log_disp": self.log_disp, "z": z}\n'
    '        if sp is not None: out["log_spec"] = sp              # PA5B: diem xep hang',
    'out["log_spec"] = sp')

# --- 2. cham diem -----------------------------------------------------------
sub("kdel/evaluate.py",
    '    smi = df["smiles"].tolist(); s_t, s_d = [], []',
    '    smi = df["smiles"].tolist(); s_t, s_d, s_s = [], [], []',
    "s_t, s_d, s_s")

sub("kdel/evaluate.py",
    "        s_t.append(lt.cpu()); s_d.append((lt - lc).cpu())\n"
    '    return {"logλ_target": torch.cat(s_t).numpy(), "logλ_t−logλ_c": torch.cat(s_d).numpy()}',
    "        s_t.append(lt.cpu()); s_d.append((lt - lc).cpu())\n"
    '        if "log_spec" in o: s_s.append(o["log_spec"].float().cpu())      # PA5B\n'
    '    out = {"logλ_target": torch.cat(s_t).numpy(), "logλ_t−logλ_c": torch.cat(s_d).numpy()}\n'
    '    if s_s: out["logλ_spec"] = torch.cat(s_s).numpy()\n'
    "    return out",
    'if s_s: out["logλ_spec"]')

# --- 3. config --------------------------------------------------------------
y = R / "configs/pa5b.yaml"
if not y.exists():
    s = (R / "configs/pa4.yaml").read_text()
    s = s.replace("name: pa4c_dmpnn_mollm", "name: pa5b_dmpnn_mollm_additive")
    s = s.replace("  mol_lm_dim: 384\n",
                  "  mol_lm_dim: 384\n  additive: true          # PA5B: lambda_t = lambda_matrix + lambda_spec\n")
    y.write_text(s); print("[ok] configs/pa5b.yaml")
else:
    print("[skip] configs/pa5b.yaml")
