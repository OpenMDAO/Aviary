from aviary.subsystems.mass.mass_builder import MassBuilderBase
from aviary.subsystems.mass.simple_mass.mass_premission import SimpleMassPremission

"""
Define subsystem builder for Aviary simple mass.

Classes
--------
SimpleMassBuilderBase: the interface for the simple mass subsystem builder. 
"""

_default_name = 'simple_mass'


class SimpleMassBuilder(MassBuilderBase):
    """Base mass builder."""

    def __init__(self, name=None):
        if name is None:
            name = _default_name

        super().__init__(name=name)

    def build_pre_mission(self, aviary_inputs):
        return SimpleMassPremission()
