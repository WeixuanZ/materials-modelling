"""Calculations involving a pair of Cu atoms

Attributes:
    get_pair (Callable): closure of the pair of atoms
"""
from typing import Union, Callable

import numpy as np
from ase import Atoms
from ase.units import Ang

try:
    from .Morse import MorsePotential
    from .util import map_func
except ImportError:
    from Morse import MorsePotential
    from util import map_func


def build_pair(d0: Union[float, int] = 1) -> Callable:
    """Closure to store the atoms object

    Args:
        d0 (Union[float, int], optional): default unit cell length

    Returns:
        Callable: function to apply strain
    """
    calc = MorsePotential()
    a = Atoms('2Cu', positions=[(0., 0., 0.), (0., 0., d0 * Ang)])
    a.set_calculator(calc)

    def change_distance(d: Union[float, int]) -> Atoms:
        """Function that returns the deformed unit cell under a given hydrostatic strain

        Args:
            d (Union[float, int]): distance (Å)

        Returns:
            Atoms: deformed atom pair
        """
        a.positions[1, 2] = d * Ang
        return a

    return change_distance


get_pair = build_pair()  # set up the closure


def get_pairwise_pe(d: Union[float, int]) -> float:
    """Calculate the potential energy of two atoms separated by the given distance

    Args:
        d (Union[float, int]): distance (Å)

    Returns:
        float: potential energy (eV)
    """
    return get_pair(d).get_potential_energy()


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
    return get_pair(d).get_forces()[1, 2]


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
