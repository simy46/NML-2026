import torch
from consts import consts


def build_hetero_edges(team_batch, device):
    """
    Vectorized construction of heterogeneous edges for a whole batch.
    team_batch: Tensor of shape (B, N) containing team IDs (-1, 0, 1).
    """
    # TODO : replace relations hardcoded values with the ones defined in consts.py

    B, N = team_batch.shape

    # Broadcast team IDs to create (B, N, N) matrices for source and destination nodes
    t_i = team_batch.unsqueeze(2)  # Source nodes
    t_j = team_batch.unsqueeze(1)  # Destination nodes

    # Initialize relation matrix
    rel = torch.zeros((B, N, N), dtype=torch.long, device=device)

    # Rule 0: Teammate (same team, not ball)
    rel[(t_i != 0) & (t_i == t_j)] = 0

    # Rule 1: Opponent (different teams, neither is ball)
    rel[(t_i != 0) & (t_j != 0) & (t_i != t_j)] = 1

    # Rule 2: Player to Ball
    rel[(t_i != 0) & (t_j == 0)] = 2

    # Rule 3: Ball to Player
    rel[(t_i == 0) & (t_j != 0)] = 3

    # Rule 4: Self-loops
    i_idx = torch.arange(N, device=device)
    rel[:, i_idx, i_idx] = 4

    # Flatten the relations into an edge_type 1D tensor
    edge_type = rel.view(-1)

    # Create batched edge_index mapping
    src = torch.arange(N, device=device).view(1, N, 1).expand(B, N, N)
    dst = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)
    batch_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)

    src = (src + batch_offset).reshape(-1)
    dst = (dst + batch_offset).reshape(-1)

    edge_index = torch.stack([src, dst], dim=0)

    return edge_index, edge_type


class SpaceTimeGraph:

    """
    Class to build the spatio-temporal graph of the 'Single-Graph approach'
    This approach consists in using only a single GNN to model both spatial and temporal dimensions.

    We use the same build_hetero_edges construction at each time step. The default approach for connexions
    across time steps is to then connect each entity with itself from each time step to the next (i.e. connect player
    i at time t to player i at t+1).
    """

    def __init__(self, config):
        self.time_connexions = config["spacetime_graph"]["temporal_connexions"]
        self.device = torch.device(config["training"]["device"])

    def get_node_id(self, node_index, t, n_max):
        """
        Get an unique spatio-temporal node ID.
            - node_index is the index of the node in a graph for a single time-step
            - t is the timestep
            - n_max be the number of entities (maximum node index)
        The id is given by 
        t * n_max + node_index
        """
        return t*n_max + node_index
    
    def build_full_graph(self, timestep_graphs):
        """
        Receives a list of timestep_graphs obtained from build_hetero_edges and connect across timesteps.

        timestep_graphs: list of tuples
        [(edge_index_t, edge_type_t), ...]

        Returns:
        full_edge_index: Tensor of shape (2, E_total)
        full_edge_type: Tensor of shape (E_total,)
        """
        T = len(timestep_graphs)

        # Check devices match
        edge_index_0, edge_type_0 = timestep_graphs[0]
        # device.type considers "mps" and "mps:0" the same instead of raising an error
        assert self.device.type == edge_index_0.device.type, f"Inconsistent devices in STGraph config ({self.device}) and time-steps graphs ({edge_index_0.device})"

        # Infer number of nodes per timestep. Add 1 because of 0-indexing.
        # If graphs are batched, this is B * N.
        num_nodes_per_timestep = int(edge_index_0.max().item()) + 1

        spatial_edge_indices = []
        spatial_edge_types = []

        # Shift each timestep graph node ids by t * n_max
        for t, (edge_index, edge_type) in enumerate(timestep_graphs):
            assert edge_index.device.type == self.device.type
            assert edge_type.device.type == self.device.type

            offset = t * num_nodes_per_timestep
            spatial_edge_indices.append(edge_index + offset)
            spatial_edge_types.append(edge_type)

        full_edge_index = torch.cat(spatial_edge_indices, dim=1)
        full_edge_type = torch.cat(spatial_edge_types, dim=0)

        if self.time_connexions == consts.STG_SELF_BASIC:
            """
            This is the default approach where we connect each entity with itself from each time step to the next
            (i.e. connect player i at time t to player i at t+1).
            """
            temporal_src = [] 
            temporal_dst = []
            temporal_types = []

            for t in range(T - 1):
                nodes_t = (
                    torch.arange(num_nodes_per_timestep, device=self.device)
                    + t * num_nodes_per_timestep
                )

                nodes_next = (
                    torch.arange(num_nodes_per_timestep, device=self.device)
                    + (t + 1) * num_nodes_per_timestep
                )

                # Forward temporal edges: t -> t+1
                temporal_src.append(nodes_t)
                temporal_dst.append(nodes_next)
                temporal_types.append(
                    torch.full(
                        (num_nodes_per_timestep,), # shape
                        consts.REL_SELF_NEXT_T, # relation type
                        dtype=torch.long,
                        device=self.device,
                    )
                )

                # Backward temporal edges: t+1 -> t
                temporal_src.append(nodes_next)
                temporal_dst.append(nodes_t)
                temporal_types.append(
                    torch.full(
                        (num_nodes_per_timestep,),
                        consts.REL_SELF_PREVIOUS_T,
                        dtype=torch.long,
                        device=self.device,
                    )
                )

            temporal_edge_index = torch.stack(
                [torch.cat(temporal_src), torch.cat(temporal_dst)],
                dim=0,
            )

            temporal_edge_type = torch.cat(temporal_types, dim=0)
            full_edge_index = torch.cat(
                [full_edge_index, temporal_edge_index],
                dim=1,
            )

            full_edge_type = torch.cat(
                [full_edge_type, temporal_edge_type],
                dim=0,
            )

        else:
            raise ValueError(
                f"Unknown temporal connection type: {self.time_connexions}"
            )

        return full_edge_index, full_edge_type
