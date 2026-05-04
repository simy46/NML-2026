import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv

from nba.graphs import build_hetero_edges


class HeteroSTGCNCell(nn.Module):
    def __init__(self, input_dim, state_dim, num_relations=5):
        super().__init__()

        self.rgcn = RGATConv(
            input_dim,
            state_dim,
            num_relations=num_relations,
        )

        self.gru = nn.GRUCell(state_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, x, h_prev, edge_index, edge_type):
        # Pass edge_type into the relational convolution
        s_feat = F.relu(self.rgcn(x, edge_index, edge_type))
        h_next = self.gru(s_feat, h_prev)

        return self.ln(h_next)


class NBAModel_HeteroSTGCN(nn.Module):
    def __init__(
        self,
        input_dim=4,
        output_dim=2,
        state_dim=64,
        context_size=8,
        horizon_size=12,
        num_entities=11,
        num_relations=5,
    ):
        super().__init__()

        self.context_size = context_size
        self.horizon_size = horizon_size
        self.num_entities = num_entities
        self.state_dim = state_dim

        # Initialize our new Heterogeneous Cell
        self.stgcn_cell = HeteroSTGCNCell(
            input_dim,
            state_dim,
            num_relations=num_relations,
        )

        self.proj = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, X):
        B, C, N, F_dim = X.shape
        T_total = self.context_size + self.horizon_size
        device = X.device

        # 1. Extract team features statically from the first timestep
        # Feature vector is [x, y, isplayer, team] -> index 3 is 'team'
        team_batch = X[:, 0, :, 3]

        # 2. Build Heterogeneous edges and types dynamically
        edge_index, edge_type = build_hetero_edges(team_batch, device)

        h = torch.zeros(B * N, self.state_dim, device=device)
        all_preds = []
        curr_coords = None

        for t in range(T_total):
            if t < self.context_size:
                x_t = X[:, t, :, :].reshape(B * N, F_dim)
            else:
                static_feats = X[:, 0, :, 2:]
                x_t = torch.cat([curr_coords, static_feats], dim=-1).reshape(
                    B * N,
                    F_dim,
                )

            # 3. Pass through Hetero cell
            h = self.stgcn_cell(x_t, h, edge_index, edge_type)

            if t >= self.context_size - 1 and t < T_total - 1:
                delta_coords = self.proj(h)

                if t == self.context_size - 1:
                    prev_coords = X[:, t, :, :2]
                else:
                    prev_coords = curr_coords

                curr_coords = prev_coords + delta_coords.reshape(B, N, 2)
                all_preds.append(curr_coords)

        out = torch.stack(all_preds, dim=0).reshape(
            self.horizon_size,
            B * N,
            2,
        )

        return out


def build_model(cfg: dict) -> NBAModel_HeteroSTGCN:
    return NBAModel_HeteroSTGCN(
        input_dim=cfg["data"]["input_dim"],
        output_dim=cfg["data"]["output_dim"],
        state_dim=cfg["model"]["state_dim"],
        context_size=cfg["data"]["context_size"],
        horizon_size=cfg["data"]["horizon_size"],
        num_entities=cfg["data"]["num_entities"],
        num_relations=cfg["model"].get("num_relations", 5),
    )