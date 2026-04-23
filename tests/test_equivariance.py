import torch
import ase
from e3nn.o3 import rand_matrix
import pytest

from src.data import atoms_to_pyg_data
from src.model import MACE

@pytest.fixture
def dummy_data():
    # Simple H2O molecule
    atoms = ase.Atoms('H2O', positions=[
        [0.0, 0.0, 0.0],
        [0.76, 0.58, 0.0],
        [-0.76, 0.58, 0.0]
    ])
    return atoms_to_pyg_data(atoms, cutoff=3.0)

def test_mace_equivariance(dummy_data):
    # 1. Instantiate model
    model = MACE(
        num_elements=10, 
        r_max=3.0, 
        num_radial=4, 
        l_max=2, 
        num_blocks=1, 
        node_dim=8
    )
    model.eval()

    # 2. Forward pass on original data
    preds_orig = model(dummy_data)
    energy_orig = preds_orig["energy"]
    forces_orig = preds_orig["forces"]

    # 3. Apply random 3D rotation
    rot = rand_matrix() # [3, 3] orthogonal rotation matrix
    
    # Rotate positions
    rotated_pos = torch.einsum("ij,nj->ni", rot, dummy_data.pos)
    
    # Create new data object with rotated positions
    rotated_data = dummy_data.clone()
    rotated_data.pos = rotated_pos
    
    # Positions are updated, model will internally recompute edge vectors and lengths

    # 4. Forward pass on rotated data
    preds_rot = model(rotated_data)
    energy_rot = preds_rot["energy"]
    forces_rot = preds_rot["forces"]

    # 5. Assert Energy Invariance: E(R*x) == E(x)
    assert torch.allclose(energy_orig, energy_rot, atol=1e-5), "Energy is not invariant to rotation!"

    # 6. Assert Force Equivariance: F(R*x) == R * F(x)
    # Rotate original forces to compare
    rotated_forces_orig = torch.einsum("ij,nj->ni", rot, forces_orig)
    assert torch.allclose(rotated_forces_orig, forces_rot, atol=1e-5), "Forces are not equivariant to rotation!"

    # 7. Assert Translation Equivariance
    translation = torch.randn(3)
    translated_data = dummy_data.clone()
    translated_data.pos = dummy_data.pos + translation
    
    preds_trans = model(translated_data)
    assert torch.allclose(energy_orig, preds_trans["energy"], atol=1e-5), "Energy is not invariant to translation!"
    assert torch.allclose(forces_orig, preds_trans["forces"], atol=1e-5), "Forces are not invariant to translation!"
