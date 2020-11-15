"""Calculations related to shear

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


def get_shear_stress(shear: float) -> np.ndarray:
    """Calculate the stress after applying a shear in Y direction

    Args:
        shear (float): shear

    Returns:
        float: pressure (eV/Å^3)
    """
    return cu_cell.shear_deform(shear).get_stress(voigt=False)


if __name__ == "__main__":
    print(np.array(cu_cell.shear_deform(0.1).get_cell()))
    print(get_shear_stress(0.1))
    cu_cell.visualize()
