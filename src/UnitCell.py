"""Calculations on a unit cell
"""
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.visualize import view

try:
    from Morse import MorsePotential
except ModuleNotFoundError:
    from .Morse import MorsePotential


class CuCell:
    """A class for the cu unit cell, with methods for applying hydrostatic strain

    Attributes:
        cu (ase.Atoms): Description
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

    def visualize(self) -> None:
        """Visualize the unit cell
        """
        view(self.cu)

    def hydrostatic_deform(self, strain: float) -> Atoms:
        """Function that returns the deformed unit cell under a given hydrostatic strain

        Args:
            strain (float): the hydrostatic strain applied

        Returns:
            ase.Atoms: deformed unit cell
        """
        self.cu.set_cell(self.init_cell * (1 - strain), scale_atoms=True)
        return self.cu

    def shear_deform(self, shear: float) -> Atoms:
        """Function that returns the deformed unit cell under a shear in Y direction

        Args:
            shear (float): shear in Y direction

        Returns:
            ase.Atoms: deformed unit cell
        """
        new_cell = self.init_cell
        new_cell[0, 1] = shear * new_cell[0, 0]
        self.cu.set_cell(new_cell)
        return self.cu
