import numpy as np

from src.util import map_func, x_of_miny


class TestClass:
    def test_map_func(self):
        assert ((map_func(lambda x: x ** 2, np.array([1, 2, 3])) == np.array([1, 4, 9])).all())

    def test_x_of_miny(self):
        assert (x_of_miny(np.array([1, 2, 3]), np.array([2, 1, 3])) == 2)
