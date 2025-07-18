import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft


class SimpleMassSummation(om.Group):
    """

    Group to compute various design masses for this mass group.

    This group will be expanded greatly as more subsystems are created.

    """

    def setup(self):
        self.add_subsystem(
            'structure_mass', StructureMass(), promotes_inputs=['*'], promotes_outputs=['*']
        )


class StructureMass(om.ExplicitComponent):
    def setup(self):
        add_aviary_input(self, Aircraft.Wing.MASS, val=0.0, units='kg')
        add_aviary_input(self, Aircraft.Fuselage.MASS, val=0.0, units='kg')
        add_aviary_input(self, Aircraft.HorizontalTail.MASS, val=0.0, units='kg')
        add_aviary_input(self, Aircraft.VerticalTail.MASS, val=0.0, units='kg')

        # More masses can be added, i.e., tail, spars, flaps, etc. as needed

        self.add_output(Aircraft.Design.STRUCTURE_MASS, val=0.0, units='kg')

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.STRUCTURE_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        wing_mass = inputs[Aircraft.Wing.MASS]
        fuselage_mass = inputs[Aircraft.Fuselage.MASS]
        htail_mass = inputs[Aircraft.HorizontalTail.MASS]
        vtail_mass = inputs[Aircraft.VerticalTail.MASS]

        structure_mass = wing_mass + fuselage_mass + htail_mass + vtail_mass

        outputs[Aircraft.Design.STRUCTURE_MASS] = structure_mass
