import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from .basis import BesselBasis, SphericalHarmonicsBasis
from .blocks import SimpleMACEBlock

class MACE(nn.Module):
    """
    Pedagogical MACE model for predicting energy and forces.
    """
    def __init__(
        self,
        num_elements: int = 120,
        r_max: float = 5.0,
        num_radial: int = 8,
        l_max: int = 2,
        num_blocks: int = 2,
        node_dim: int = 16
    ):
        super().__init__()
        self.r_max = r_max
        
        # 1. Atomic embedding (Z -> initial node features)
        self.node_embedding = nn.Embedding(num_elements, node_dim)
        
        # The node irreps starts as purely invariant scalars (L=0)
        # But after the first layer, it will contain higher l components (e.g. 16x0e + 16x1o + 16x2e)
        self.node_irreps = o3.Irreps(f"{node_dim}x0e + {node_dim}x1o + {node_dim}x2e")
        
        # Initial projection to the full irreps space (only fills the 0e part)
        self.initial_projection = o3.Linear(o3.Irreps(f"{node_dim}x0e"), self.node_irreps)
        
        # 2. Basis functions
        self.radial_basis = BesselBasis(cutoff=r_max, num_radial=num_radial)
        self.sh_basis = SphericalHarmonicsBasis(l_max=l_max)
        
        # 3. MACE Interaction Blocks
        self.blocks = nn.ModuleList([
            SimpleMACEBlock(
                node_irreps=str(self.node_irreps),
                sh_irreps=str(self.sh_basis.irreps_out),
                radial_dim=num_radial
            ) for _ in range(num_blocks)
        ])
        
        # 4. Readout Phase
        # Extract only the invariant scalars (0e) for energy prediction
        self.readout_linear = o3.Linear(self.node_irreps, o3.Irreps(f"{node_dim}x0e"))
        self.readout_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1) # Predict scalar atomic energy
        )

    def forward(self, data: Data) -> dict:
        """
        Forward pass to predict energy and forces.
        Forces are computed as the negative gradient of energy with respect to positions.
        """
        is_training = self.training
        with torch.set_grad_enabled(True):
            # Ensure pos requires grad for force computation
            pos = data.pos
            if not pos.requires_grad:
                pos = pos.clone().requires_grad_(True)
                
        z = data.z
        edge_index = data.edge_index
        row, col = edge_index
        
        # 1. Edge features
        edge_vec = pos[row] - pos[col]
        edge_len = torch.norm(edge_vec, dim=-1)
        
        radial_feat = self.radial_basis(edge_len)
        sh_feat = self.sh_basis(edge_vec)
        
        # 2. Node Embeddings
        # [num_nodes, node_dim]
        node_feat = self.node_embedding(z)
        # Project to full irreps space
        node_feat = self.initial_projection(node_feat)
        
        # 3. Message Passing Blocks
        for block in self.blocks:
            node_feat = block(node_feat, edge_index, radial_feat, sh_feat)
            
        # 4. Readout
        # Filter out non-scalar irreps
        inv_feat = self.readout_linear(node_feat) # [num_nodes, node_dim]
        # Predict atomic energies
        site_energies = self.readout_mlp(inv_feat) # [num_nodes, 1]
        
        # Sum atomic energies to get total system energy
        # If batching is used, we'd scatter according to data.batch. 
        # For pedagogy, we assume 1 graph or use a simple sum over all nodes if unbatched.
        if hasattr(data, 'batch') and data.batch is not None:
            total_energy = scatter(site_energies, data.batch, dim=0, reduce='sum')
        else:
            total_energy = site_energies.sum(dim=0, keepdim=True)
            
        # 5. Compute Forces via Autograd
        # F = - dE/dR
        forces = -torch.autograd.grad(
            outputs=total_energy,
            inputs=pos,
            grad_outputs=torch.ones_like(total_energy),
            create_graph=is_training, # Important for gradcheck and force-matching training
            retain_graph=is_training
        )[0]
        
        return {
            "energy": total_energy,
            "forces": forces
        }
