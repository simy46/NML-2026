import torch


def build_hetero_edges(team_batch, device):
    """
    Vectorized construction of heterogeneous edges for a whole batch.
    team_batch: Tensor of shape (B, N) containing team IDs (-1, 0, 1).
    """
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