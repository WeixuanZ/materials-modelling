"""Calculations involving a pair of Cu atoms

Attributes:
    a (ase.Atoms): object representing the atoms
    calc (Morse.MorsePotential): calculator attached to the atoms
"""
from typing import Union

import numpy as np
from ase import Atoms
from ase.units import Ang

from src.Morse import MorsePotential
from src.util import map_func

calc = MorsePotential()
a = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., 1 * Ang)])
a.set_calculator(calc)


def get_pairwise_pe(d: Union[float, int]) -> float:
    """Calculate the potential energy of two atoms separated by the given distance

    Args:
        d (Union[float, int]): distance (Å)

    Returns:
        float: potential energy (eV)
    """
    a.positions[1, 2] = d * Ang
    return a.get_potential_energy()


def get_pairwise_pes(arr: np.ndarray) -> np.ndarray:
    """Apply pairwise potential energy calculation to an array of distances

    Args:
        arr (np.ndarray): array of distances (Å)

    Returns:
        np.ndarray: array of potential energies (eV)
    """
    return map_func(get_pairwise_pe, arr)


def get_pairwise_force(d: Union[float, int]) -> float:
    """Calculate the force between two atoms separated by the given distance

    Args:
        d (Union[float, int]): distance (Å)

    Returns:
        float: force (eV/Å)
    """
    a.positions[1, 2] = d * Ang
    return a.get_forces()[1, 2]


def get_pairwise_forces(arr: np.ndarray) -> np.ndarray:
    """Apply pairwise force calculation to an array of distances

    Args:
        arr (np.ndarray): array of distances (Å)

    Returns:
        np.ndarray: array of forces (eV/Å)
    """
    return map_func(get_pairwise_force, arr)


if __name__ == "__main__":
    print(get_pairwise_pe(2.5))
    print(get_pairwise_pes(np.linspace(0, 5, 10)))
