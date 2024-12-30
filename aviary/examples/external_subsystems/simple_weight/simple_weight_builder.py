"""
Builder for a simple component that computes a new wing and horizontal tail
mass.

The SimpleWeight component is placed inside of a group so that we can promote
the variable "Tail" using the alias Aircraft.HorizontalTail.MASS
"""
import openmdao.api as om

from aviary.examples.external_subsystems.simple_weight.simple_weight import SimpleWeight
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Aircraft


class WingWeightBuilder(SubsystemBuilderBase):
    """
    Prototype of a subsystem that overrides an aviary internally computed var.

    It also provides a method to build OpenMDAO systems for the pre-mission and mission computations of the subsystem.

    Attributes
    ----------
    name : str ('wing_weight')
        object label

    Methods
    -------
    __init__(self, name='wing_weight'):
        Initializes the WingWeightBuilder object with a given name.
    build_pre_mission(self, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the pre-mission computations of the subsystem.
    """

    def __init__(self, name='wing_weight'):
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        '''
        Build an OpenMDAO system for the pre-mission computations of the subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        '''
        wing_group = om.Group()
        wing_group.add_subsystem("aerostructures", SimpleWeight(),
                                 promotes_inputs=["aircraft:*"],
                                 promotes_outputs=[Aircraft.Wing.MASS,
                                                   ('Tail', Aircraft.HorizontalTail.MASS)])
        return wing_group
