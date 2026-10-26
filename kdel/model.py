import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

def mlp(i,h,o): return nn.Sequential(nn.Linear(i,h), nn.ReLU(), nn.Linear(h,o))

class HierGINE(nn.Module):
    def __init__(self, node_dim=28, edge_dim=6, hidden=256, n_layers=5, dropout=0.2):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden)
        self.edge_in = nn.Linear(edge_dim, hidden)
        self.convs = nn.ModuleList([GINEConv(mlp(hidden,hidden,hidden), edge_dim=hidden)
                                    for _ in range(n_layers)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)
        self.score = mlp(hidden, hidden, 2)      # log lambda: target, control
        self.zero  = mlp(hidden, hidden, 2)      # logit pi
        self.log_disp = nn.Parameter(torch.zeros(2))

    def forward(self, d):
        h = self.node_in(d.x); e = self.edge_in(d.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = self.drop(F.relu(bn(conv(h, d.edge_index, e))))
        z = global_add_pool(h, d.batch)
        return {"log_lambda": self.score(z), "pi_logit": self.zero(z),
                "log_disp": self.log_disp}
