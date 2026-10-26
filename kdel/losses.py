import torch, torch.nn.functional as F

def nb_nll(counts, log_lambda, log_disp, pre, pi_logit=None):
    """counts [B,R]; log_lambda,pi_logit [B]; pre [B]. pi_logit=None -> NB thuan."""
    r  = log_disp.clamp(-4,4).exp()
    mu = (log_lambda.clamp(-15,15) + pre.log()).exp().clamp(1e-6, 1e8).unsqueeze(1)
    nb = torch.distributions.NegativeBinomial(total_count=r, logits=(mu/r).log())
    lp = nb.log_prob(counts.float())
    if pi_logit is None:
        return -lp.mean(1)
    pi    = torch.sigmoid(pi_logit).clamp(1e-6, 1-1e-6).unsqueeze(1)
    lp_nz = (1-pi).log() + lp
    lp_z  = torch.log(pi + (1-pi)*nb.log_prob(torch.zeros_like(counts).float()).exp() + 1e-12)
    return -torch.where(counts == 0, lp_z, lp_nz).mean(1)

def total_loss(out, b, beta=1.0, zero_inflated=True):
    ll_t, ll_c = out["log_lambda"].unbind(-1)
    pi_t, pi_c = out["pi_logit"].unbind(-1)
    d_t,  d_c  = out["log_disp"].unbind(-1)
    if not zero_inflated: pi_t = pi_c = None
    l_t = nb_nll(b.target_counts,  ll_t, d_t, b.pre, pi_t).mean()
    l_c = nb_nll(b.control_counts, ll_c, d_c, b.pre, pi_c).mean()
    return l_t + beta*l_c, {"target": l_t.item(), "control": l_c.item()}


def rank_loss(score, y, w, margin=0.5):
    """Ranking logistic co trong so tin cay: voi moi cap (i,j) co y_i - y_j > margin, phat -log sigmoid(s_i - s_j).
    score,y,w: [B]. w_ij = w_i * w_j."""
    dy = y.unsqueeze(1) - y.unsqueeze(0)
    pair = (dy > margin).float() * (w.unsqueeze(1) * w.unsqueeze(0))
    l = F.softplus(-(score.unsqueeze(1) - score.unsqueeze(0)))
    return (pair * l).sum() / pair.sum().clamp_min(1e-6)
