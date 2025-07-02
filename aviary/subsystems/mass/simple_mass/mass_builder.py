from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.mass.simple_mass.mass_premission import MassPremission


"""

Define subsystem builder for Aviary core mass.

Classes
--------------------------------------------------------------------------------------------------

MassBuilderBase: the interface for a mass subsystem builder. **Not sure how necessary this is for
                 my work right now, but wanted to include it as a just in case. I basically copied
                 it over from the mass_builder.py under the mass subsystems folder in Aviary github.

"""

_default_name = 'simple_mass'

class MassBuilderBase(SubsystemBuilderBase):
    """
    Base mass builder
    
    """

    def __init__(self, name=None):
       if name is None:
           name = _default_name
    
       super().__init__(name=name)
    
    def build_pre_mission(self, aviary_inputs):
        return MassPremission()
    
