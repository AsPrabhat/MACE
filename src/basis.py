import torch
import torch.nn as nn
from e3nn import o3
import math

class PolynomialEnvelope(nn.Module):
    """
    Polynomial envelope function that smoothly goes to zero at the cutoff radius.
    Ensures that the potential and its derivatives are continuous at the cutoff.
    """
    def __init__(self, cutoff: float, p: int = 5):
        super().__init__()
        self.cutoff = cutoff
        self.p = p

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # Scale r by cutoff
        u = r / self.cutoff
        # Mask out values beyond cutoff
        u = torch.where(u < 1.0, u, torch.ones_like(u))
        # Smooth polynomial envelope: (1 - (r/rc)^p )^p or similar. 
        # A common choice in MACE/NequIP is 1 - 6u^5 + 15u^4 - 10u^3 (if p=2) 
        # Here we use a simpler standard (1-u)^p for pedagogical reasons.
        envelope = (1.0 - u)**self.p
        return envelope

class BesselBasis(nn.Module):
    """
    Radial embedding using Spherical Bessel functions multiplied by an envelope.
    """
    def __init__(self, cutoff: float, num_radial: int = 8):
        super().__init__()
        self.cutoff = cutoff
        self.num_radial = num_radial
        self.envelope = PolynomialEnvelope(cutoff=cutoff)
        
        # Frequencies for the bessel functions: pi, 2pi, 3pi, ...
        # Shape: [num_radial]
        freqs = torch.arange(1, num_radial + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: Edge lengths, shape [num_edges]
        Returns:
            Radial embeddings, shape [num_edges, num_radial]
        """
        # r shape: [num_edges, 1]
        r = r.unsqueeze(-1)
        
        # d is the normalized distance [num_edges, 1]
        d = r / self.cutoff
        
        # Bessel functions: sin(freq * d) / d
        # Using PyTorch's native sinc function (which computes sin(pi*x)/(pi*x))
        bessel = self.freqs * torch.sinc(self.freqs * d / math.pi)
        
        # Apply envelope
        env = self.envelope(r)
        
        # Final radial features: shape [num_edges, num_radial]
        return bessel * env

class SphericalHarmonicsBasis(nn.Module):
    """
    Angular embedding using e3nn spherical harmonics.
    """
    def __init__(self, l_max: int):
        super().__init__()
        self.l_max = l_max
        # Define the irreducible representations (irreps) for the spherical harmonics
        # For l_max=2, irreps will be "1x0e + 1x1o + 1x2e"
        self.irreps_out = o3.Irreps.spherical_harmonics(l_max)

    def forward(self, edge_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_vec: 3D displacement vectors, shape [num_edges, 3]
        Returns:
            Spherical harmonics embeddings, shape [num_edges, sum(2l+1)]
        """
        # Normalize the edge vectors to get unit directions
        edge_len = torch.norm(edge_vec, dim=-1, keepdim=True)
        # Avoid division by zero
        edge_dir = edge_vec / (edge_len + 1e-6)
        
        # Compute spherical harmonics. e3nn expects unnormalized or normalized vectors.
        # normalize=True ensures e3nn normalizes them internally as well.
        sh = o3.spherical_harmonics(
            self.irreps_out, 
            edge_dir, 
            normalize=True, 
            normalization='component'
        )
        return sh
