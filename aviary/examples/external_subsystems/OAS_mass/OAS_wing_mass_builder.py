import openmdao.api as om

import aviary.api as av
from aviary.examples.external_subsystems.OAS_mass.OAS_wing_mass_analysis import OAStructures


class OASWingMassBuilder(av.SubsystemBuilderBase):
    """
    Builder for an OpenAeroStruct component that computes a new wing mass.

    This also provides a method to build OpenMDAO systems for the pre-mission and mission computations of the subsystem.

    Attributes
    ----------
    name : str ('wing_mass')
        object label

    Methods
    -------
    __init__(self, name='wing_mass'):
        Initializes the OASWingmassBuilder object with a given name.
    build_pre_mission(self) -> openmdao.core.System:
        Build an OpenMDAO system for the pre-mission computations of the subsystem.
    """

    def __init__(self, name='wing_mass'):
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        """
        Build an OpenMDAO system for the pre-mission computations of the subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        """
        wing_group = om.Group()
        wing_group.add_subsystem(
            'aerostructures',
            OAStructures(
                symmetry=True,
                wing_weight_ratio=1.0,
                S_ref_type='projected',
                n_point_masses=1,
                num_twist_cp=4,
                num_box_cp=51,
            ),
            promotes_inputs=[
                ('fuel', av.Mission.Summary.FUEL_MASS),
            ],
            promotes_outputs=[('wing_mass', av.Aircraft.Wing.MASS)],
        )

        return wing_group
