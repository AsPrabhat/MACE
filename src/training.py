import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def energy_force_loss(
    preds: dict, 
    targets: dict, 
    energy_weight: float = 1.0, 
    force_weight: float = 100.0
) -> dict:
    """
    Composite loss function for Force-Matching.
    Typically, forces are weighted much higher than energy (e.g. 100 to 1000) 
    because there are 3N force components vs 1 energy scalar per graph.
    """
    mse = nn.MSELoss()
    
    loss_e = mse(preds["energy"], targets["energy"])
    loss_f = mse(preds["forces"], targets["forces"])
    
    loss_total = energy_weight * loss_e + force_weight * loss_f
    
    return {
        "loss": loss_total,
        "loss_e": loss_e,
        "loss_f": loss_f
    }

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    energy_weight: float = 1.0,
    force_weight: float = 100.0
) -> dict:
    model.train()
    total_loss = 0.0
    total_loss_e = 0.0
    total_loss_f = 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(batch)
        
        targets = {
            "energy": batch.y,
            "forces": batch.forces
        }
        
        # Compute loss
        loss_dict = energy_force_loss(preds, targets, energy_weight, force_weight)
        
        # Backward pass
        loss = loss_dict["loss"]
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_loss_e += loss_dict["loss_e"].item()
        total_loss_f += loss_dict["loss_f"].item()
        
    num_batches = len(loader)
    return {
        "loss": total_loss / num_batches,
        "loss_e": total_loss_e / num_batches,
        "loss_f": total_loss_f / num_batches
    }

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> dict:
    model.eval()
    total_mae_e = 0.0
    total_mae_f = 0.0
    
    l1 = nn.L1Loss()
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
            
        preds = model(batch)
        
        mae_e = l1(preds["energy"], batch.y)
        mae_f = l1(preds["forces"], batch.forces)
        
        total_mae_e += mae_e.item()
        total_mae_f += mae_f.item()
        
    num_batches = len(loader)
    return {
        "mae_e": total_mae_e / num_batches,
        "mae_f": total_mae_f / num_batches
    }
