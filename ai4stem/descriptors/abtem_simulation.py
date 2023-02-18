import abtem
from abtem.potentials import Potential
from abtem.waves import Probe
import numpy as np
from ase.io import read
import cv2


class abTEM_simulation():
    """
    """
    
    def __init__(self, probe_parameters=None, sampling=0.12):
        
        self.probe_parameters = probe_parameters
        self.sampling = sampling
        
        
    def calculate(self, atoms, **kwargs):
        
        # set up Probe

        probe = Probe(energy=300e3, sampling=self.sampling,
                      extent=2, semiangle_cutoff=24, 
                      focal_spread=60, defocus=0)
        
        positions = [(1, 1)]
        waves = probe.build(positions)
        probe_image = np.abs(waves.array) ** 2
        
        
        # Read simulation cell and atom positions from file
        # atoms = read(self.cell, format="vasp")

        # Determine slice thickness
        eps = 0.001 
        atoms_pos = atoms.get_positions()
        atoms_z = atoms_pos[:,2]
        slice_thickness = np.min(atoms_z[atoms_z>eps])

        # Compute the projected potential
        projpot = Potential(atoms, sampling=self.sampling,
                            slice_thickness=slice_thickness,
                            projection='finite', parametrization='kirkland')
        precalc_pot = projpot.build(pbar=True)
        precalc_pot = np.sum(precalc_pot.array, axis=0)

        # Convolve with probe
        conv_stem = cv2.filter2D(precalc_pot, -1, probe_image)
        
        return conv_stem