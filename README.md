## CUED Part IB Materials Modelling Lab

Using Morse potential to investigate elastic properties of single crystal copper.
Documentation is available at https://weixuanz.github.io/materials-modelling/.

```python
import numpy as np

from src.hydrostatic import get_hydrostatic_pes
from src.util import x_of_miny

strains = np.linspace(-0.1, 0.1, 50)
E = get_hydrostatic_pes(strains) / 4

eq_strain = x_of_miny(strains, E)
```

### Bulk Modulus

```python
from ase.units import GPa

from src.hydrostatic import get_hydrostatic_pressure, get_hydrostatic_vol

delta = 0.001

K = - (
    get_hydrostatic_vol(eq_strain)
    * (get_hydrostatic_pressure(eq_strain + delta) - get_hydrostatic_pressure(eq_strain))
    / (get_hydrostatic_vol(eq_strain + delta) - get_hydrostatic_vol(eq_strain))
    )
print(K / GPa)
```

### Shear Modulus

```python
from ase.units import GPa

from src.shear import get_shear_stress

shear = 0.01

G = get_shear_stress(shear) / shear
print(G / GPa)
```

### Poisson Ratio

```python
from scipy.optimize import fsolve

from src.UnitCell import CuCell

cu_cell = CuCell.from_default_eq_strain(eq_strain)

strain_x = 0.01
strain_y_z = fsolve(lambda x: cu_cell.strain_deform(strain_x, x, x).get_stress(voigt=False)[1, 1], -strain_x)[0]

v = strain_y_z / strain_x
print(v)
```
