import numpy as np

from src.pairwise import get_pairwise_pes, get_pairwise_forces

class TestClass:
    def test_get_pairwise_force(self):
        epsilon = 1e-2
        n = 100
        r = np.random.randint(2, 5, n)
        forces = get_pairwise_forces(r)
        approx_forces = -(get_pairwise_pes(r + epsilon) - get_pairwise_pes(r)) / epsilon
        assert(np.allclose(forces, approx_forces, rtol=0.2))

