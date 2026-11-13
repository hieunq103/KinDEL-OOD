"""paired.py: cho phep moi ben dung mot thuoc do rieng (--rule_pairs A:B). Idempotent."""
import pathlib
p = pathlib.Path("kdel/paired.py"); s = p.read_text()
if "rule_pairs" in s:
    print("[skip] kdel/paired.py"); raise SystemExit

def sub(old, new):
    global s
    assert old in s, f"khong thay: {old[:60]!r}"
    s = s.replace(old, new, 1)

sub('    ap.add_argument("--rules", nargs="+", default=["logλ_target", "logλ_t−logλ_c"])',
    '    ap.add_argument("--rules", nargs="+", default=["logλ_target", "logλ_t−logλ_c"])\n'
    '    ap.add_argument("--rule_pairs", nargs="+", default=None,\n'
    '                    help="moi muc \'thuocdo_A:thuocdo_B\' (A=ben a_pat, B=ben b_pat); ghi de --rules")')
sub('    rng = np.random.default_rng(a.seed); rows = []; pool = {}',
    '    pairs = [tuple(x.split(":")) for x in a.rule_pairs] if a.rule_pairs else [(r, r) for r in a.rules]\n'
    '    rng = np.random.default_rng(a.seed); rows = []; pool = {}')
sub('                for rule in a.rules:\n'
    '                    bootA = np.mean([-spearman_rows(s[rule][idx], kd[idx]) for s in SA], axis=0)\n'
    '                    bootB = np.mean([-spearman_rows(s[rule][idx], kd[idx]) for s in SB], axis=0)\n'
    '                    d = bootB - bootA\n'
    '                    pa = float(np.mean([-spearman_rows(s[rule][None], kd[None])[0] for s in SA]))\n'
    '                    pb = float(np.mean([-spearman_rows(s[rule][None], kd[None])[0] for s in SB]))',
    '                for ra, rb in pairs:\n'
    '                    rule = ra if ra == rb else f"{ra} | {rb}"\n'
    '                    bootA = np.mean([-spearman_rows(s[ra][idx], kd[idx]) for s in SA], axis=0)\n'
    '                    bootB = np.mean([-spearman_rows(s[rb][idx], kd[idx]) for s in SB], axis=0)\n'
    '                    d = bootB - bootA\n'
    '                    pa = float(np.mean([-spearman_rows(s[ra][None], kd[None])[0] for s in SA]))\n'
    '                    pb = float(np.mean([-spearman_rows(s[rb][None], kd[None])[0] for s in SB]))')
p.write_text(s); print("[ok] kdel/paired.py")
