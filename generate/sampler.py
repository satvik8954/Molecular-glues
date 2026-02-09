"""
Molecule sampler for generating new molecules from trained model.
"""
from typing import List, Optional
import torch
from torch_geometric.data import Data
from tqdm import tqdm

import sys
import os

# Add project root to path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model.diffusion import MolecularDiffusion
from config import Config


class MoleculeSampler:
    """
    Wrapper class for molecule sampling with additional options.
    """
    
    def __init__(
        self,
        model: MolecularDiffusion,
        device: str = 'cpu',
    ):
        """
        Args:
            model: Trained MolecularDiffusion model
            device: Device to run sampling on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def sample(
        self,
        num_molecules: int,
        num_atoms: int = 20,
        temperature: float = 1.0,
        batch_size: int = 100,
        show_progress: bool = True,
        atom_size_range: Optional[tuple] = None,
    ) -> List[Data]:
        """
        Sample multiple molecules.
        
        Args:
            num_molecules: Total number of molecules to generate
            num_atoms: Default number of atoms (if atom_size_range is None)
            temperature: Sampling temperature (higher = more diverse)
            batch_size: Number of molecules to generate in parallel
            show_progress: Show progress bar
            atom_size_range: Optional (min, max) range for random atom counts
            
        Returns:
            List of generated molecular graphs
        """
        all_molecules = []
        
        remaining = num_molecules
        iterator = range(0, num_molecules, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating molecules")
        
        for start_idx in iterator:
            current_batch = min(batch_size, remaining)
            
            # Determine atom count for this batch
            if atom_size_range:
                min_atoms, max_atoms = atom_size_range
                # Sample different sizes
                for _ in range(current_batch):
                    n_atoms = torch.randint(min_atoms, max_atoms + 1, (1,)).item()
                    mol = self.model.sample(
                        num_molecules=1,
                        num_atoms=n_atoms,
                        temperature=temperature,
                    )[0]
                    all_molecules.append(mol)
            else:
                # Fixed size
                molecules = self.model.sample(
                    num_molecules=current_batch,
                    num_atoms=num_atoms,
                    temperature=temperature,
                )
                all_molecules.extend(molecules)
            
            remaining -= current_batch
        
        return all_molecules
    
    @torch.no_grad()
    def sample_diverse(
        self,
        num_molecules: int,
        min_atoms: int = 10,
        max_atoms: int = 30,
        temperatures: List[float] = [0.7, 1.0, 1.3],
        show_progress: bool = True,
    ) -> List[Data]:
        """
        Sample diverse molecules with varying sizes and temperatures.
        
        Args:
            num_molecules: Total number of molecules to generate
            min_atoms: Minimum atoms per molecule
            max_atoms: Maximum atoms per molecule
            temperatures: List of temperatures to use
            show_progress: Show progress bar
            
        Returns:
            List of generated molecular graphs
        """
        molecules_per_temp = num_molecules // len(temperatures)
        all_molecules = []
        
        for temp in temperatures:
            if show_progress:
                print(f"Sampling with temperature {temp}")
            
            mols = self.sample(
                num_molecules=molecules_per_temp,
                atom_size_range=(min_atoms, max_atoms),
                temperature=temp,
                show_progress=show_progress,
            )
            all_molecules.extend(mols)
        
        # Handle remainder
        remaining = num_molecules - len(all_molecules)
        if remaining > 0:
            mols = self.sample(
                num_molecules=remaining,
                atom_size_range=(min_atoms, max_atoms),
                temperature=1.0,
                show_progress=False,
            )
            all_molecules.extend(mols)
        
        return all_molecules
