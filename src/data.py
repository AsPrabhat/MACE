import torch
from torch_geometric.data import Data
import ase
import numpy as np

def simple_radius_graph(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Computes a radius graph using pure PyTorch.
    Avoids the complex torch_cluster C++ dependency, making installation easier for students.
    O(N^2) complexity is fine for small pedagogical molecules.
    """
    if pos.size(0) > 2000:
        import warnings
        warnings.warn("Dense distance matrix computation is highly memory intensive for N > 2000.")

    # Compute pairwise distances
    dist_matrix = torch.cdist(pos, pos)
    
    # Mask: distances less than cutoff AND greater than 1e-6 (no self-loops)
    mask = (dist_matrix < cutoff) & (dist_matrix > 1e-6)
    
    # Get edge indices where mask is True
    # nonzero() returns [num_edges, 2], so we transpose to get [2, num_edges]
    edge_index = torch.nonzero(mask).t()
    
    return edge_index

def atoms_to_pyg_data(atoms: ase.Atoms, cutoff: float = 5.0) -> Data:
    """
    Converts an ASE Atoms object into a PyTorch Geometric Data object.
    Computes the neighbor list using a simple radius graph (no PBC handling for simplicity in the pedagogical version).
    
    Args:
        atoms: ase.Atoms object.
        cutoff: radial cutoff for neighbor edges.
        
    Returns:
        Data object containing:
            - z: atomic numbers (Node feature)
            - pos: 3D coordinates
            - edge_index: [2, num_edges] graph connectivity
            - edge_vec: 3D displacement vector between source and target nodes
            - edge_len: scalar distance between source and target nodes
            - y: Optional total energy
            - forces: Optional forces on atoms
    """
    # 1. Node features (atomic numbers) and positions
    z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    
    # Build neighbor graph using pure PyTorch
    # format is [source, target]. Note: in MPNNs, messages usually flow from source to target.
    # We use our custom function to exclude self-interactions and avoid torch_cluster.
    edge_index = simple_radius_graph(pos, cutoff=cutoff)
    
    # 3. Compute edge vectors and lengths
    row, col = edge_index
    # Edge vector from source (col) to target (row):
    # This direction matches convention: atom i (row) receives message from neighbor j (col)
    edge_vec = pos[row] - pos[col]
    edge_len = torch.norm(edge_vec, dim=-1)
    
    # 4. Optional target properties
    data = Data(z=z, pos=pos, edge_index=edge_index, edge_vec=edge_vec, edge_len=edge_len)
    
    if "energy" in atoms.info:
        data.y = torch.tensor([[atoms.info["energy"]]], dtype=torch.float32)
    
    try:
        forces = atoms.get_forces()
        data.forces = torch.tensor(forces, dtype=torch.float32)
    except Exception:
        pass # No forces available
        
    return data
