"""Calculations on a unit cell

Attributes:
    cu_cell (CuCell): instance of CuCell class with default lattice vector size
"""
import numpy as np
from ase import Atoms
from ase.build import bulk

try:
    from .Morse import MorsePotential
    from .util import map_func
except ImportError:
    from Morse import MorsePotential
    from util import map_func


class CuCell:

    """A class for the cu unit cell, with methods for applying hydrostatic strain

    Attributes:
        cu (Atoms): Description
    """

    def __init__(self, a: float = 3.6):
        """
        Args:
            a (float, optional): unit cell length
        """
        self.cu = bulk("Cu", "fcc", a=a, cubic=True)
        self.cu.set_calculator(MorsePotential())

        self._init_cell = np.array(self.cu.get_cell())
        self._init_vol = self.cu.get_volume()

    @property
    def init_cell(self) -> np.ndarray:
        """Get initial cell vectors

        Returns:
            np.ndarray: initial cell vectors (columns)
        """
        return self._init_cell

    @property
    def init_vol(self) -> float:
        """Get initial unit call volume

        Returns:
            float: initial volume (Å^3)
        """
        return self._init_vol

    def get_deformed_cell(self, strain: float) -> Atoms:
        """Function that returns the deformed unit cell under a given hydrostatic strain

        Args:
            strain (float): the hydrostatic strain applied

        Returns:
            Atoms: deformed unit cell
        """
        self.cu.set_cell(self.init_cell * (1 - strain), scale_atoms=True)
        return self.cu

    def get_deformed_vol(self, strain: float) -> float:
        """Calculate the new volume after applying the strain
        
        Args:
            strain (float): strain
        
        Returns:
            float: new volume (Å^3)
        """
        return self.init_vol * strain**3


cu_cell = CuCell()


def get_hydrostatic_pe(strain: float) -> float:
    """Calculate the potential energy after applying a hydrostatic strain

    Args:
        strain (float): strain

    Returns:
        float: potential energy (eV)
    """
    return cu_cell.get_deformed_cell(strain).get_potential_energy()


def get_hydrostatic_pes(arr: np.ndarray) -> np.ndarray:
    """Apply the potential energy calculation to an array of strains

    Args:
        arr (np.ndarray): array of strains

    Returns:
        np.ndarray: array of potential energies (eV)
    """
    return map_func(get_hydrostatic_pe, arr)


def get_hydrostatic_stress(strain: float) -> float:
    """Calculate the stress after applying a hydrostatic strain

    Args:
        strain (float): strain

    Returns:
        float: pressure (eV/Å^3)
    """
    return cu_cell.get_deformed_cell(strain).get_stress(voigt=False)[0, 0]


def get_hydrostatic_stresses(arr: np.ndarray) -> np.ndarray:
    """Apply the stress calculation to an array of strains

    Args:
        arr (np.ndarray): array of strains

    Returns:
        np.ndarray: array of pressures (eV/Å^3)
    """
    return map_func(get_hydrostatic_stress, arr)


def get_hydrostatic_vols(arr: np.ndarray) -> np.ndarray:
    """Apply the deformed volume calculation to an array of strains

    Args:
        arr (np.ndarray): array of strains

    Returns:
        np.ndarray: array of volumes (sÅ^3)
    """
    return map_func(cu_cell.get_deformed_vol, arr)
