from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv, Linear


class LearnableTables(nn.Module):
    """
    Inputs:
      - user: one shared trainable vector (uniform across users)
      - subgroup: learnable table [S, d_in]
      - question/choice: learnable table [C, d_in]
    """
    def __init__(self, num_users: int, num_subgroups: int, num_choices: int, d_in: int = 64):
        super().__init__()
        self.sub_embed = nn.Embedding(num_subgroups, d_in)
        self.cho_embed = nn.Embedding(num_choices, d_in)
        self.user_token = nn.Parameter(torch.zeros(1, d_in))

        nn.init.normal_(self.sub_embed.weight, std=0.02)
        nn.init.normal_(self.cho_embed.weight, std=0.02)
        nn.init.normal_(self.user_token, std=0.02)
        self.num_users = num_users

    def forward(self) -> Dict[str, torch.Tensor]:
        return {
            "subgroup": self.sub_embed.weight,                       # [S, d]
            "question": self.cho_embed.weight,                       # [C, d]
            "user":     self.user_token.expand(self.num_users, -1),  # [U, d] uniform
        }


class RGCNEncoder(nn.Module):
    def __init__(self, metadata: Tuple, d_h: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ntypes, self.etypes = metadata
        # Map each edge type string to a unique integer ID for RGCNConv
        self.rel2id = {rel: i for i, rel in enumerate(self.etypes)}
        n_rel = len(self.etypes)

        # Initial projection: Align different node feature dimensions to d_h
        self.in_lin = nn.ModuleDict({nt: Linear((-1), d_h) for nt in self.ntypes})
        
        # Core RGCN layers: These contain the learnable relation-specific weights
        self.layers = nn.ModuleList([
            RGCNConv(d_h, d_h, num_relations=n_rel) for _ in range(layers)
        ])
        
        # Output projection: Map hidden states to final embedding dimension
        self.out_lin = nn.ModuleDict({nt: Linear(d_h, d_h) for nt in self.ntypes})
        self.dropout = nn.Dropout(dropout)

    def _from_homo(self, x_all: torch.Tensor, slices: Dict[str, Tuple[int, int]]):
        """Split the consolidated tensor back into a dictionary of node types."""
        return {nt: x_all[a:b] for nt, (a, b) in slices.items()}

    def _to_homo(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict):
        """Consolidate heterogeneous nodes and edges into a single homogeneous graph."""
        xs, slices, off = [], {}, 0
        device = next(iter(x_dict.values())).device
        
        # Flatten node features into one large tensor
        for nt in self.ntypes:
            x = x_dict[nt]
            xs.append(x)
            slices[nt] = (off, off + x.size(0))
            off += x.size(0)
        x_all = torch.cat(xs, dim=0)

        # Globalize edge indices and create edge type labels
        e_src, e_dst, e_type = [], [], []
        for rel, ei in edge_index_dict.items():
            s_nt, _, d_nt = rel
            s0, _ = slices[s_nt]
            d0, _ = slices[d_nt]
            e_src.append(ei[0].to(device) + s0)
            e_dst.append(ei[1].to(device) + d0)
            # Assign the integer ID corresponding to this specific relation type
            e_type.append(torch.full((ei.size(1),), self.rel2id[rel], dtype=torch.long, device=device))
            
        edge_index = torch.stack([torch.cat(e_src), torch.cat(e_dst)], dim=0)
        return x_all, edge_index, torch.cat(e_type), slices

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict):
        # 1. Project initial features to hidden dimension d_h
        x_dict = {nt: F.relu(self.in_lin[nt](x)) for nt, x in x_dict.items()}
        
        # 2. Iteratively apply RGCN layers
        for conv in self.layers:
            # Convert to homogeneous format for RGCNConv compatibility
            x_all, edge_index, e_type, slices = self._to_homo(x_dict, edge_index_dict)
            
            # Use the built-in RGCNConv forward pass (handles message passing & weights)
            x_all = conv(x_all, edge_index, e_type)
            
            x_all = F.relu(x_all)
            x_all = self.dropout(x_all)
            
            # Map back to dict for the next layer or final output
            x_dict = self._from_homo(x_all, slices)
            
        # 3. Final linear transformation per node type
        return {nt: self.out_lin[nt](x) for nt, x in x_dict.items()}


class DotSoftmaxDecoder(nn.Module):
    """
    Dot(z_user, z_option) / tau, softmax over options of the SAME question.

    Supports two modes:
      - Train mode: pass gold_idx -> return (loss, acc, probs) [optional: +logits]
      - Inference mode: gold_idx=None -> return probs [optional: (probs, logits)]
    """
    def __init__(self, tau_init: float = 1.0):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))

    def forward(self,
                z_user: torch.Tensor,
                z_choice: torch.Tensor,
                users: torch.LongTensor,
                option_ids: torch.LongTensor,
                gold_idx: Optional[torch.LongTensor] = None,
                return_logits: bool = False):  # <--- New parameter
        
        tau = torch.exp(self.log_tau)
        zu = z_user[users]               # [B, d]
        zopts = z_choice[option_ids]     # [B, K, d]
        
        # scores are logits (unnormalized scores)
        scores = torch.einsum('bd,bkd->bk', zu, zopts) / tau   
        probs = F.softmax(scores, dim=-1)                      # [B, K]

        # --- 1. Inference mode ---
        if gold_idx is None:
            if return_logits:
                return probs, scores  # return tuple (probs, logits)
            else:
                return probs          # return only Tensor

        # --- 2. Train mode ---
        loss = F.cross_entropy(scores, gold_idx)
        with torch.no_grad():
            acc = (scores.argmax(dim=1) == gold_idx).float().mean().item()
        
        if return_logits:
            return loss, acc, probs, scores # return 4 values
        else:
            return loss, acc, probs         # return only Tensor


class GEMSModel(nn.Module):
    """Paper-faithful: LearnableTables + RGCNEncoder + DotSoftmaxDecoder."""
    def __init__(self, data: HeteroData, d_in: int = 64, d_h: int = 64, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # Total global nodes for each type
        U = data['user'].num_nodes
        S = data['subgroup'].num_nodes
        C = data['question'].num_nodes
        
        self.tables = LearnableTables(U, S, C, d_in=d_in)
        self.enc = RGCNEncoder(data.metadata(), d_h=d_h, layers=layers, dropout=dropout)
        self.dec = DotSoftmaxDecoder(tau_init=1.0)

    def forward(self, data_mp: HeteroData, batch: Dict[str, torch.Tensor] = None, 
                return_z: bool = False, return_logits: bool = False):
        """
        Processes the graph and decodes results. 
        Note: PyG Batching handles the node indexing; manual tiling is usually unnecessary.
        """
        # 1. Fetch base embeddings from the learnable tables
        x_in = self.tables() 

        # 2. Encode graph structure into latent embeddings z
        z = self.enc(x_in, data_mp.edge_index_dict)

        # Early return if only node embeddings are needed (e.g., for visualization)
        if return_z:
            return z

        # 3. Decode for the specific batch of tasks provided
        if batch is None:
            return None 

        gold_idx = batch.get('gold_idx', None)

        # Perform Dot-Product matching and Softmax
        return self.dec(
            z['user'], 
            z['question'],
            batch['users'],      # Indices for user nodes in z['user']
            batch['option_ids'], # Indices for question nodes in z['question']
            gold_idx=gold_idx,
            return_logits=return_logits 
        )