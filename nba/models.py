import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from torch import Tensor
import torch_geometric as pyg

from nba.graphs import build_hetero_edges, SpaceTimeGraph


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
    

class RGATLSTMCell(nn.Module):
    def __init__(
        self, 
        input_dim,
        state_dim,
        rgat_num_relations,
        rgat_dim_per_head,
        rgat_heads,
    ):
        super(RGATLSTMCell, self).__init__()
        self.rgat = RGATConv(
            in_channels=input_dim,
            out_channels=rgat_dim_per_head,
            num_relations=rgat_num_relations,
            heads=rgat_heads,
        )
        rgat_dim = rgat_dim_per_head * rgat_heads
        self.lstm = nn.LSTMCell(rgat_dim, state_dim)

    def forward(self, x, edge_index, edge_type, h_prev, c_prev):
        rgat_out = self.rgat(x, edge_index, edge_type)
        spatial_features = F.relu(rgat_out)
        return self.lstm(spatial_features, (h_prev, c_prev))


class RGATLSTMNBAModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        output_dim=2,
        context_size=8,
        horizon_size=12,
        num_entities=11,
        state_dim=128,
        rgat_num_relations=5,
        rgat_dim_per_head=64,
        rgat_heads=4,
        pred_head_hidden_dim=64,    
    ):
        super(RGATLSTMNBAModel, self).__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.num_entities = num_entities
        self.state_dim = state_dim

        self.rgat_lstm_cell = RGATLSTMCell(
            input_dim=input_dim,
            state_dim=state_dim,
            rgat_num_relations=rgat_num_relations,
            rgat_dim_per_head=rgat_dim_per_head,
            rgat_heads=rgat_heads,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(state_dim, pred_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_head_hidden_dim, output_dim),
        )
    
    def forward(self, X):
        batch_size, context_size, num_entity, feature_dim = X.shape
        device = X.device

        static_features = X[:, 0, :, 2:]
        team_batch = X[:, 0, :, 3]
        edge_index, edge_type = build_hetero_edges(team_batch, device)
        h = torch.zeros(
                batch_size*num_entity, self.state_dim, 
                device=device,
            ) 
        c = torch.zeros(
                batch_size*num_entity, self.state_dim, 
                device=device,
            )
        curr_coords = torch.zeros(
            batch_size, num_entity, 2, 
            device=device,
        )
        all_preds = []

        for t in range(self.window_size):
            # extract input features for time step t
            if t < self.context_size:
                x_t = X[:, t, :, :].reshape(
                    batch_size*num_entity, 
                    feature_dim
                )
            else:
                x_t = torch.cat(
                    [curr_coords, static_features], 
                    dim=-1,
                ).reshape(
                    batch_size*num_entity, 
                    feature_dim
                )
            
            # pass through cell
            h, c = self.rgat_lstm_cell(x_t, edge_index, edge_type, h, c)

            # prediction for timesteps in horizon
            if self.context_size-1 <= t and t < self.window_size-1:
                delta = self.pred_head(h)
                if t == self.context_size - 1:
                    prev_coords = X[:, t, :, :2]
                else:
                    prev_coords = curr_coords
                curr_coords = prev_coords + delta.reshape(
                    batch_size, 
                    num_entity, 
                    2
                )
                all_preds.append(curr_coords)
        
        return torch.stack(all_preds, dim=0).reshape(
            self.horizon_size,
            batch_size * num_entity,
            2,
        )
    

class GAN_GRU(torch.nn.Module):
    """ Module to perform predictions on the NBA dataset. """
    def __init__(self, input_feature_dim:int, num_nodes:int,output_dim:int,state_dim_RNN:int,state_dim_GNN:int,state_dim_MLP:int, graph_conn_radius:float, context_size:int,horizon_size:int):
        super().__init__()
        # Modules
        embedding_dim = 4
        # huge oversmoothing with GCN and fully connected graph
        self.GNN = pyg.nn.GCN(in_channels=input_feature_dim + embedding_dim, hidden_channels=state_dim_GNN, num_layers=2)
        # try with GAT

        # self.GNN = pyg.nn.GAT(
        #             in_channels=input_feature_dim + embedding_dim, 
        #             hidden_channels=state_dim_GNN, 
        #             num_layers=2, 
        #             heads=4, 
        #             concat=False
        #         )
        # self.RNN = torch.nn.GRU(input_size=state_dim_GNN, hidden_size = state_dim_RNN, bias=True)
        self.RNN = torch.nn.GRUCell(input_size=state_dim_GNN, hidden_size=state_dim_RNN, bias=True)
        self.proj = pyg.nn.MLP(in_channels=state_dim_RNN, hidden_channels=state_dim_MLP, out_channels=output_dim, num_layers=2)
        
        # Time attributes
        self.context_size = context_size
        self.horizon_size = horizon_size

        self.graph_conn_radius = graph_conn_radius # has to be normalized

        #base edges for edge index
        base_edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                base_edges.append([i, j])
        self.register_buffer('base_edge_index', torch.tensor(base_edges, dtype=torch.long).T)

        # node embeddings - they are added to the hidden state to identify the players
        self.node_emb = torch.nn.Embedding(num_nodes, embedding_dim)

    def forward(self,X:Tensor) -> Tensor:
        """ Forward pass. """
        B,_,N,F = X.shape
        h_prev = torch.zeros(size=(B*N,self.RNN.hidden_size), device=X.device)
        T = self.context_size+self.horizon_size
        batch_vector = torch.arange(B, device=X.device).repeat_interleave(N)

        # construct edge idx list - offset for the entire batch - fully connected graph
        # TODO: can it learn different weights for different edge relations now?
        # edge_index_list = []
        # for b in range(B):
        #     edge_index_list.append(self.base_edge_index + (b * N))
        
        # # dim [2, B * N * N]
        # edge_index = torch.cat(edge_index_list, dim=1)

        node_ids = torch.arange(N, device=X.device).repeat(B)
        emb = self.node_emb(node_ids)

        all_preds = []
        for t in range(T):
            # Format input
            if t < self.context_size:
                x = X[:,t,:,:].reshape(B*N,F)
                curr_pos = x[:, :2]
            else:
                # combine with static features
                x = torch.cat([curr_pos,X[:,0,:,2:].reshape(B*N,2)],dim=1)
            # edge index built dynamically with distances - to avoid oversmoothing
            edge_index = self.radius_graph(curr_pos, r=self.graph_conn_radius, batch=batch_vector)
            # add embedding to make players identifiable -> they loose the spatial awareness with that
            # node_ids = torch.arange(N, device=X.device).repeat(B)
            # x[:, 2:] = x[:, 2:] + self.node_emb(node_ids) # don't add node embedding at positions 
            # concat instead of add
            x_w_emb = torch.cat([x, emb], dim=-1)

            # Process input
            h_GNN = self.GNN(x_w_emb, edge_index)
            h = self.RNN(h_GNN,h_prev)

            delta = self.proj(h) 
            
            # The new position is: last position + predicted movement -> should evict the jump
            curr_pos = curr_pos + delta
            
            if t >= self.context_size - 1 and t < T - 1:
                all_preds.append(curr_pos)

            # Update state
            h_prev = h
        all_preds = torch.stack(all_preds,dim=0) # [T,B*N,2]
        return all_preds
    
    def radius_graph(self, pos, r, batch):
        # pos: [Batch*N, 2], r: float, batch: [Batch*N]
        # Compute pairwise distances
        dist_mat = torch.cdist(pos, pos) 
        
        adj = dist_mat <= r
        
        batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
        
        adj = adj & batch_mask
        adj.fill_diagonal_(False)
        
        # Convert to edge_index [2, E]
        return adj.nonzero(as_tuple=False).t().contiguous()



class SpaceTimeGNN(nn.Module):
    """
    SpaceTimeGNN - Same graph strategy at a given time-step, we connect each individual graph to obtain a 
    big unique graph that simultaneously models the time-series.

    Nodes are (entity, timestep) pairs.
    Spatial edges connect entities inside each timestep.
    Temporal edges connect the same entity across consecutive timesteps.

    Input:
        X: (B, C, N, F)

    Output:
        preds: (H, B * N, 2)
    
    TODO: This currently assumes a fixed context window and predicts the whole horizon in one shot.
    It would be nice to extend the model to support variable-length contexts and autoregressive next-step prediction,
    to get some LLM-style training over all prefixes of a trajectory sequence.
    """

    def __init__(self, config):
        super().__init__()

        # Data config
        data_cfg = config["data"]
        self.input_dim = data_cfg["input_dim"]
        self.output_dim = data_cfg["output_dim"]
        self.context_size = data_cfg["context_size"]
        self.horizon_size = data_cfg["horizon_size"]
        self.num_entities = data_cfg["num_entities"]

        # Model config
        model_cfg = config["model"]
        self.stgnn_num_relations = model_cfg["num_relations"]
        self.stgnn_dim_per_head = model_cfg["dim_per_head"]
        self.stgnn_heads = model_cfg["heads"]
        self.stgnn_num_layers = model_cfg["num_layers"]
        self.pred_head_hidden_dim = model_cfg["pred_head_hidden_dim"]
        self.hidden_dim = self.stgnn_dim_per_head * self.stgnn_heads

        self.device = torch.device(config["training"]["device"])

        # Graph builder
        self.graph_builder = SpaceTimeGraph(config)

        # Input projection
        self.input_proj = nn.Linear(
            self.input_dim,
            self.hidden_dim,
        )

        # Spatio-temporal RGAT layers
        self.convs = nn.ModuleList()
        for _ in range(self.stgnn_num_layers):
            self.convs.append(
                RGATConv(
                    in_channels=self.hidden_dim,
                    out_channels=self.stgnn_dim_per_head,
                    num_relations=self.stgnn_num_relations,
                    heads=self.stgnn_heads,
                )
            )

        # Normalization layers
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.stgnn_num_layers)]
        )

        # Prediction head
        # Predict H future (x,y) offsets
        self.pred_head = nn.Sequential(
            nn.Linear(
                self.hidden_dim,
                self.pred_head_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                self.pred_head_hidden_dim,
                self.horizon_size * self.output_dim,
            ),
        )

    def forward(self, X):
        """
        X: (B, C, N, F)
        """
        B, C, N, F_dim = X.shape
        assert X.device.type == self.device.type, f"Input X is on {X.device}, but model config expects {self.device}"

        assert C == self.context_size
        assert N == self.num_entities
        assert F_dim == self.input_dim

        # Static team labels, shape (B, N)
        team_batch = X[:, 0, :, 3]

        # Build one spatial graph per timestep
        timestep_graphs = []
        for t in range(C):
            edge_index_t, edge_type_t = build_hetero_edges(team_batch, self.device)
            timestep_graphs.append((edge_index_t, edge_type_t))

        # Combine into one big space-time graph
        edge_index, edge_type = self.graph_builder.build_full_graph(timestep_graphs)

        # Flatten nodes: (B, C, N, F) -> (C, B, N, F) -> (C * B * N, F)
        # This matches graph indexing: timestep blocks, each containing B*N nodes.
        x = X.permute(1, 0, 2, 3).reshape(C * B * N, F_dim)

        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_type)
            h_new = F.relu(h_new)
            h = norm(h + h_new)  # residual connection

        # Extract embeddings from last observed timestep t = C - 1
        start = (C - 1) * B * N
        end = C * B * N
        h_last = h[start:end]  # (B * N, hidden_dim)

        # Predict future displacements
        delta = self.pred_head(h_last)  # (B * N, H * 2)
        delta = delta.view(B, N, self.horizon_size, self.output_dim)

        # Convert displacements to future positions
        last_pos = X[:, -1, :, :2]  # (B, N, 2)

        # Option A: direct offsets from last observed position
        future = last_pos.unsqueeze(2) + delta  # (B, N, H, 2)

        # Return in same style as other models: (H, B*N, 2)
        future = future.permute(2, 0, 1, 3).reshape(
            self.horizon_size,
            B * N,
            self.output_dim,
        )

        return future


def build_model(cfg: dict) -> nn.Module:

    if cfg["MODEL_NAME"] == "HeteroSTGCN":
        return NBAModel_HeteroSTGCN(
            input_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
            state_dim=cfg["model"]["state_dim"],
            context_size=cfg["data"]["context_size"],
            horizon_size=cfg["data"]["horizon_size"],
            num_entities=cfg["data"]["num_entities"],
            num_relations=cfg["model"].get("num_relations", 5),
        )
    elif cfg["MODEL_NAME"] == "RGATLSTM":
        return RGATLSTMNBAModel(
            input_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
            context_size=cfg["data"]["context_size"],
            horizon_size=cfg["data"]["horizon_size"],
            num_entities=cfg["data"]["num_entities"],
            state_dim=cfg["model"]["state_dim"],
            rgat_num_relations=cfg["model"].get("rgat_num_relations", 5),
            rgat_dim_per_head=cfg["model"].get("rgat_dim_per_head", 64),
            rgat_heads=cfg["model"].get("rgat_heads", 4),
            pred_head_hidden_dim=cfg["model"].get("pred_head_hidden_dim", 64),
        )
    elif cfg["MODEL_NAME"] == "GAN_GRU":
        return GAN_GRU(
            input_feature_dim=cfg["data"]["input_dim"],
            output_dim=cfg["data"]["output_dim"],
            num_nodes=cfg["model"]["num_nodes"],
            state_dim_GNN=cfg["model"]["state_dim_GNN"],
            state_dim_RNN=cfg["model"]["state_dim_RNN"],
            state_dim_MLP=cfg["model"]["state_dim_MLP"],
            graph_conn_radius=cfg["model"]["graph_conn_radius"],
            context_size=cfg["data"]["context_size"],
            horizon_size=cfg["data"]["horizon_size"],
        )
    elif cfg["MODEL_NAME"] == "SpaceTimeGNN":
        return SpaceTimeGNN(cfg)

    else:
        raise ValueError(f"Unknown model name: {cfg['MODEL_NAME']}")