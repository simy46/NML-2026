import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.nn import RGATConv

os.environ["WANDB_MODE"] = "offline"
import wandb


class NBADataset(Dataset):
    """ Dataset to load NBA highlights tensor data. """
    def __init__(self, data_path:str, context_size:int, horizon_size:int, mu:Tensor, sigma:Tensor):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.load_data(data_path, mu, sigma)

    def load_data(self,data_path:str,mu:Tensor,sigma:Tensor):
        """ Load all sequences and normalize positions. """
        self.sequences = []
        self.max_start =[]
        for f in[os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.pt')]:
            seq = torch.load(f, weights_only=False) # [T,N,F]
            seq[:,:,[0,1]] = (seq[:,:,[0,1]].clone() - mu) / sigma
            if seq.shape[0] >= self.window_size:
                self.sequences.append(seq)
                self.max_start.append(max(0,len(seq)-self.window_size))

    def __getitem__(self, index) -> tuple:
        """ Retrieve sequence given an index in format: (seq_idx:int,start_point:int). """
        seq_idx, start = index
        T = self.window_size
        X = self.sequences[seq_idx][start:start+self.context_size]
        y = self.sequences[seq_idx][start+self.context_size:start+T]
        return X,y
    
    def __len__(self):
        return len(self.sequences)

class NBASampler(Sampler):
    def __init__(self, batch_size:int, max_start:list, seed=0, shuffle=True):
        self.batch_size = batch_size
        self.max_start = max_start
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.shuffle = shuffle
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        n = len(self)
        if self.shuffle:
            perm = torch.randperm(n, generator=self.generator).tolist()
        else:
            perm = list(range(n))
        perm_start =[(i,self.max_start[i]) for i in perm]
        for k in range(0, n, self.batch_size):
            batch = perm_start[k:k + self.batch_size]
            for idx, max_start in batch:
                start = torch.randint(0,max_start+1,size=(),generator=self.generator)
                yield idx, start

    def __len__(self):
        return len(self.max_start)


class MultiStepMSE():
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()
    
    def compute(self,pred,target) -> Tensor:
        """ Compute the average MSE over the horizon. """
        B,T,N,F = target.shape 
        target = target[:,:,:,:2].permute(1,0,2,3).reshape(T,B*N,2)
        if pred.size(1) != target.size(1):
            raise ValueError(f'Dimension mismatch: pred {pred.size(1)} vs target {target.size(1)}')
        loss = 0
        for t in range(T):
            loss += self.loss_fn(pred[t,:,:],target[t,:,:])
        return loss / T


def build_hetero_edges(team_batch, device):
    """
    Vectorized construction of heterogeneous edges for a whole batch.
    team_batch: Tensor of shape (B, N) containing team IDs (-1, 0, 1).
    """
    B, N = team_batch.shape
    
    # Broadcast team IDs to create (B, N, N) matrices for source and destination nodes
    t_i = team_batch.unsqueeze(2) # Source nodes
    t_j = team_batch.unsqueeze(1) # Destination nodes
    
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


class HeteroSTGCNCell(nn.Module):
    def __init__(self, input_dim, state_dim, num_relations=5):
        super().__init__()
        self.rgcn = RGATConv(input_dim, state_dim, num_relations=num_relations)
        self.gru = nn.GRUCell(state_dim, state_dim)
        self.ln = nn.LayerNorm(state_dim)

    def forward(self, x, h_prev, edge_index, edge_type):
        # Pass edge_type into the relational convolution
        s_feat = F.relu(self.rgcn(x, edge_index, edge_type))
        h_next = self.gru(s_feat, h_prev)
        return self.ln(h_next)

class NBAModel_HeteroSTGCN(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, state_dim=64, context_size=8, horizon_size=12, num_entities=11):
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.num_entities = num_entities
        self.state_dim = state_dim
        
        # Initialize our new Heterogeneous Cell
        self.stgcn_cell = HeteroSTGCNCell(input_dim, state_dim, num_relations=5)
        
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
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
        all_preds =[]
        curr_coords = None

        for t in range(T_total):
            if t < self.context_size:
                x_t = X[:, t, :, :].reshape(B * N, F_dim)
            else:
                static_feats = X[:, 0, :, 2:] 
                x_t = torch.cat([curr_coords, static_feats], dim=-1).reshape(B * N, F_dim)

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

        out = torch.stack(all_preds, dim=0).reshape(self.horizon_size, B * N, 2)
        return out

class NBATrainer():
    ENTITY_MAPPING = {-1: 'Team_A', 0: 'Ball', 1: 'Team_B'}
    
    def __init__(self, batch_size:int, epochs:int, train_path:str, val_path:str,
                 device:str='cuda', context_size:int=8, horizon_size:int=12, seed:int=0):
        self.batch_size = batch_size
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.epochs = epochs
        self.train_path = train_path
        self.val_path = val_path
        self.seed = seed
        
        # Load data
        mu, sigma = self.compute_normalization_statistics(self.train_path)
        self.mu, self.sigma = mu.to(self.device), sigma.to(self.device)
        self.train_dataset = NBADataset(self.train_path, context_size, horizon_size, mu, sigma)
        self.val_dataset = NBADataset(self.val_path, context_size, horizon_size, mu, sigma)

    def compute_normalization_statistics(self, data_path:str) -> tuple:
        all_pos = []
        for f in[os.path.join(data_path,f) for f in os.listdir(data_path) if f.endswith('.pt')]:
            seq = torch.load(f, weights_only=False) 
            pos = seq[:,:,[0,1]]
            all_pos.append(pos)
        all_pos = torch.cat(all_pos, dim=0) 
        mu = all_pos.mean(dim=(0,1))
        sigma = all_pos.std(dim=(0,1))
        return mu, sigma

    def set_seed(self) -> torch.Generator:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        generator = torch.Generator('cpu').manual_seed(self.seed) 
        return generator

    def train(self):
        generator = self.set_seed()
        train_sampler = NBASampler(batch_size=self.batch_size, max_start=self.train_dataset.max_start, shuffle=True)
        val_sampler = NBASampler(batch_size=self.batch_size, max_start=self.val_dataset.max_start, shuffle=False)
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, generator=generator)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, generator=generator)
        
        model = NBAModel_HeteroSTGCN(4, 2, 64, self.context_size, self.horizon_size).to(self.device)
        loss_fn = MultiStepMSE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
        
        with wandb.init(settings=wandb.Settings(_disable_stats=True, _disable_meta=True, mode="offline")):
            dataloader_dict = {'train':train_dataloader, 'val':val_dataloader}
            for epoch in range(self.epochs):
                print(f"Epoch {epoch+1}/{self.epochs}")
                for split, dataloader in dataloader_dict.items():
                    if split == 'val':
                        model.eval()
                        with torch.no_grad():
                            self.train_one_epoch(dataloader, loss_fn, model, optimizer, scheduler, split)
                    else:
                        model.train()
                        self.train_one_epoch(dataloader, loss_fn, model, optimizer, scheduler, split)
                    
                    sampler = dataloader.sampler
                    sampler.set_epoch(epoch)
                    scheduler.step()
                    
            torch.save(model.state_dict(), os.path.join(f'stgcn_model.pth'))
            print("Training Complete. Model saved as 'stgcn_model.pth'.")

    def train_one_epoch(self, dataloader, loss_fn, model, optimizer, scheduler, split):
        avg_loss = 0
        batches_processed = 0
        for batch in dataloader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)
            
            pred = model(X)
            loss = loss_fn.compute(pred, y)
            
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                
            avg_loss += loss.item()
            batches_processed += 1
            
        if batches_processed > 0:
            avg_loss = avg_loss / batches_processed
            wandb.log({f'{split}_loss': avg_loss})
            print(f"  [{split.upper()}] Loss: {avg_loss:.4f}")
        else:
            print(f"  Warning: No batches processed for {split} split.")

    def get_trajectory(self, X:Tensor) -> Tensor:
        model = NBAModel_HeteroSTGCN(4, 2, 64, 8, 12).to(self.device)
        model.load_state_dict(torch.load('stgcn_model.pth', weights_only=True))
        model.eval()
        
        with torch.no_grad():
            pred = model(X.unsqueeze(0).to(self.device)).cpu()
            
        # Denormalize
        X = X.cpu()
        mu = self.mu.cpu()
        sigma = self.sigma.cpu()
        
        X[:,:,:2] = X[:,:,:2] * sigma + mu
        pred = pred * sigma + mu
        
        # Format
        static = X[-1,:,2:].unsqueeze(0).repeat(pred.size(0), 1, 1)
        pred = torch.cat([pred, static], dim=-1) 
        pred = torch.cat([X, pred], dim=0).detach()
        return pred

    def get_kaggle_submission(self, test_dir:str, target_dir:str) -> pd.DataFrame:
        all_traj = []
        for f, f_path in[(f, os.path.join(test_dir,f)) for f in os.listdir(test_dir) if f.endswith('.pt')]:
            seq = torch.load(f_path, weights_only=False) 
            seq[:,:,[0,1]] = (seq[:,:,[0,1]].clone() - self.mu.cpu()) / self.sigma.cpu()
            
            traj = self.get_trajectory(seq)
            traj = traj[8:,:,:2]
            traj = traj.reshape(-1)
            all_traj.append([int(f.removesuffix('.pt'))] + traj.tolist())
            
        all_traj = pd.DataFrame(all_traj, columns=['id']+[f'entity_{i}_time_{t}_{axis}' for t in range(12) for i in range(11) for axis in ['x','y']]).set_index('id')
        all_traj = all_traj.sort_index()
        all_traj.to_csv(os.path.join(target_dir, 'solution.csv'))
        print(f"Submission saved to {os.path.join(target_dir, 'solution.csv')}")


if __name__ == "__main__":
    # --- IMPORTANT: Change these paths to your dataset folders ---
    TRAIN_DIR = 'data/train/train'
    TEST_DIR = 'data/test/test'
    TARGET_DIR = 'data/submission/'
    
    # Check if paths exist to prevent immediate crashing
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"Error: Could not find data directories.")
        print(f"Please ensure {TRAIN_DIR} and {TEST_DIR} exist.")
    else:
        # 1. Initialize Trainer
        trainer = NBATrainer(
            batch_size=64,
            epochs=10,
            train_path=TRAIN_DIR,
            val_path=TEST_DIR,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            seed=0
        )
        
        # 2. Train the Model
        print("Starting ST-GCN Training...")
        trainer.train()
        
        # 3. Generate Kaggle Submission
        print("Generating Kaggle Submission...")
        trainer.get_kaggle_submission(test_dir=TEST_DIR, target_dir=TARGET_DIR)
