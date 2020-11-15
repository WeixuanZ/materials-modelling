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
    r"""Calculate the stress after applying a shear in :math:`y` direction

    The shear tensor is

    .. math::
       :nowrap:

        \[\left[
        \begin{array}{lll}\sigma_{x} & \tau_{x y} & \tau_{x z} \\
        \tau_{y x} & \sigma_{y} & \tau_{y z} \\
        \tau_{z x} & \tau_{z y} & \sigma_{z}\end{array}
        \right]\]

    if the stress is only applied in the :math:`y` direction, :math:`\tau_{x z}=\tau_{y z}=0`

    Args:
        shear (float): shear

    Returns:
        float: shear stress :math:`\tau_{x y}` (eV/Å^3)
    """
    return cu_cell.shear_deform(shear).get_stress(voigt=False)[0, 1]


if __name__ == "__main__":
    print(np.array(cu_cell.shear_deform(0.1).get_cell()))
    print(get_shear_stress(0.1))
    cu_cell.visualize()
