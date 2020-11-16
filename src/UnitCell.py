"""Calculations on a unit cell
"""
from __future__ import annotations  # before 3.10 for postponed evaluation of annotations

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.visualize import view

try:
    from Morse import MorsePotential
except ModuleNotFoundError:
    from .Morse import MorsePotential


class CuCell:
    """A class for the Cu unit cell, with methods for applying strain and shear
    
    Attributes:
        cu (ase.Atoms): an object representing the atoms
        default_a (float): default unit cell size
    """
    default_a = 3.6

    def __init__(self, a: float = default_a) -> None:
        """
        Args:
            a (float, optional): unit cell length
        """
        self.cu = bulk("Cu", "fcc", a=a, cubic=True)
        self.cu.set_calculator(MorsePotential())

        self._init_cell = np.array(self.cu.get_cell())
        self._init_vol = self.cu.get_volume()

    @classmethod
    def from_default_eq_strain(cls, strain: float) -> CuCell:
        """Create the class from strain (with respect to the default unit cell size)

        The is used to create a unit cell with size where the potential energy is minimum,
        as calculated elsewhere.

        Args:
            strain (float): the strain

        Returns:
            CuCell: class created
        """
        return cls(CuCell.default_a * (1 - strain))

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

    def set_cell_size(self, a: float) -> None:
        """Set the unit cell length

        Args:
            a (float): the length of unit cell
        """
        self.cu.set_cell(a, scale_atoms=True)

    def hydrostatic_deform(self, strain: float) -> Atoms:
        """Function that returns the deformed unit cell under a given hydrostatic strain

        Args:
            strain (float): the hydrostatic strain applied

        Returns:
            ase.Atoms: deformed unit cell
        """
        self.set_cell_size(self.init_cell * (1 - strain))
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

    def strain_deform(self, strain_x: float, strain_y: float, strain_z: float) -> Atoms:
        """Function that returns the deformed unit cell under normal strains

        Args:
            strain_x (float): strain in the x direction
            strain_y (float): strain in the y direction
            strain_z (float): strain in the z direction

        Returns:
            ase.Atoms: deformed unit cell
        """
        strain = np.ones((3, 3))
        strain[np.diag_indices(3)] = np.array([strain_x, strain_y, strain_z]) + 1
        new_cell = self.init_cell * strain
        self.cu.set_cell(new_cell)
        return self.cu
