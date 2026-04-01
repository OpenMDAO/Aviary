import openmdao.api as om

from aviary.subsystems.aerodynamics.flops_based.design import Design


class PreMissionFLOPSAero(om.Group):
    """design Mach number and coefficient of lift"""

    def setup(self):
        self.add_subsystem(
            'design',
            Design(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
