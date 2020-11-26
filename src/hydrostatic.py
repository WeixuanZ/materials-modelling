"""Calculations related to hydrostatic loading

Attributes:
    cu_cell (CuCell): instance of CuCell class with default lattice vector size
"""
import numpy as np

try:
    from UnitCell import CuCell
    from util import map_func
except ModuleNotFoundError:
    from .UnitCell import CuCell
    from .util import map_func


cu_cell = CuCell()


def get_hydrostatic_pe(strain: float) -> float:
    """Calculate the potential energy after applying a hydrostatic strain

    Args:
        strain (float): strain

    Returns:
        float: potential energy (eV)
    """
    return cu_cell.hydrostatic_deform(strain).get_potential_energy()


def get_hydrostatic_pes(arr: np.ndarray) -> np.ndarray:
    """Apply the potential energy calculation to an array of strains

    Args:
        arr (np.ndarray): array of strains

    Returns:
        np.ndarray: array of potential energies (eV)
    """
    return map_func(get_hydrostatic_pe, arr)


def get_hydrostatic_stress(strain: float) -> np.ndarray:
    """Calculate the stress after applying a hydrostatic strain

    Args:
        strain (float): strain

    Returns:
        float: stress (eV/Å^3)
    """
    return cu_cell.hydrostatic_deform(strain).get_stress(voigt=False)


def get_hydrostatic_pressure(strain: float) -> float:
    """Calculate the pressure from the stress matrix

    Args:
        strain (np.ndarray): strain

    Returns:
        float: hydrostatic pressure (eV/Å^3)
    """
    return - 0.33 * np.trace(get_hydrostatic_stress(strain))


def get_hydrostatic_pressures(arr: np.ndarray) -> np.ndarray:
    """Apply the pressure calculation to an array of strains

    Args:
        arr (np.ndarray): array of strains

    Returns:
        np.ndarray: array of pressures (eV/Å^3)
    """
    return map_func(get_hydrostatic_pressure, arr)


def get_hydrostatic_vol(strain: float) -> float:
    """Calculate the new volume after applying the strain

    Args:
        strain (float): strain

    Returns:
        float: new volume (Å^3)
    """
    return cu_cell.init_vol * (1 + strain) ** 3


def get_hydrostatic_vols(arr: np.ndarray) -> np.ndarray:
    """Apply the deformed volume calculation to an array of strains

    Args:
        arr (np.ndarray): array of strains

    Returns:
        np.ndarray: array of volumes (sÅ^3)
    """
    return map_func(get_hydrostatic_vol, arr)
