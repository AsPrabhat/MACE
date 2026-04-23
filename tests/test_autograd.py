import torch
import ase
from torch.autograd import gradcheck
import pytest

from src.data import atoms_to_pyg_data
from src.model import MACE

@pytest.fixture
def dummy_data():
    atoms = ase.Atoms('CO', positions=[
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0]
    ])
    return atoms_to_pyg_data(atoms, cutoff=3.0)

def test_mace_autograd(dummy_data):
    # We use float64 for gradcheck to avoid numerical precision issues with finite differences
    dummy_data.pos = dummy_data.pos.to(torch.float64)
    dummy_data.pos.requires_grad_(True)
    
    # Needs to be a very small model for gradcheck to be fast
    model = MACE(
        num_elements=10, 
        r_max=3.0, 
        num_radial=2, 
        l_max=1, 
        num_blocks=1, 
        node_dim=4
    ).to(torch.float64)
    model.train() # Must be in train mode to retain graph for gradcheck

    # The function we want to check is Energy(Positions)
    def compute_energy(pos):
        # Create a fresh copy to pass through the model to prevent graph issues
        data = dummy_data.clone()
        data.pos = pos
        
        # Recompute edges based on new pos (which is required for gradcheck)
        row, col = data.edge_index
        edge_vec = pos[row] - pos[col]
        data.edge_vec = edge_vec
        # Note: In our model, we recalculate edge_len and edge_vec inside the forward pass
        # if we pass the data object, so modifying data.pos is sufficient.
        
        preds = model(data)
        return preds["energy"]

    # gradcheck tests if analytical gradient matches numerical gradient
    # We test if d(Energy)/d(Positions) is computed correctly by PyTorch autograd engine
    test_passed = gradcheck(compute_energy, (dummy_data.pos,), eps=1e-6, atol=1e-4)
    assert test_passed, "Autograd check failed! Forces may not be the exact derivative of Energy."
