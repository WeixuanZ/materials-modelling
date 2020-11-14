"""Help functions
"""
from typing import Union, Callable

import numpy as np


def map_func(func: Callable[[Union[float, int]], float], arr: np.ndarray) -> np.ndarray:
    """Mapping a function over a Numpy array

    Args:
        func (Callable[[Union[float, int]], float]): function applied to each element of the array
        arr (np.ndarray): array to be mapped over

    Returns:
        np.ndarray: transformed array
    """
    return np.vectorize(func)(arr)


def x_of_miny(x: np.ndarray, y: np.ndarray) -> Union[int, float]:
    """Get the value of x where the y is minimum

    Args:
        x (np.ndarray): array of x
        y (np.ndarray): array of y

    Returns:
        Union[int, float]: x of the minimum point
    """
    return x[np.argmin(y)]
