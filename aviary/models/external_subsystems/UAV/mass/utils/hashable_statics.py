"""Converts lists and numpy arrays into plain tuples.

The UAV mass components use options that are lists/arrays (e.g. rib materials,
rib thicknesses). OpenMDAO needs to compare these values between runs to know
if anything changed, but it can't do that with lists or arrays — it crashes
with "TypeError: unhashable type". Tuples work fine, so hashable() converts
everything to tuples before handing it to OpenMDAO. The values themselves are
unchanged.
"""

import numpy as np


def hashable(val):
    """Recursively convert a value (array, list, tuple, scalar) to a hashable form."""
    if isinstance(val, np.ndarray):
        return tuple(val.tolist())
    if isinstance(val, (list, tuple)):
        return tuple(hashable(v) for v in val)
    return val
