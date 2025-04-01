import numpy as np

import openmdao.api as om
import openmdao.jax as omj

# Maybe add some aviary inputs at some point here

class MassSummation(om.Group):
    """

    Group to compute various design masses for this mass group.

    This group will be expanded greatly as more subsystems are created. 

    """

    def setup(self):
        self.add_subsystem(
            'structure_mass', StructureMass(),
            promotes_inputs=['*'], promotes_outputs=['*']
        )


class StructureMass(om.JaxExplicitComponent):

    def setup(self):
        # Maybe later change these to Aviary inputs?
        self.add_input('wing_mass', val=0.0, units='kg')
        self.add_input('fuse_mass', val=0.0, units='kg')
        self.add_input('tail_mass', val=0.0, units='kg')
        # More masses can be added, i.e., tail, spars, flaps, etc. as needed

        self.add_output('structure_mass', val=0.0, units='kg')

    def setup_partials(self):
        # I'm not sure what else to put here at the moment
        self.declare_partials('structure_mass', '*', val=1)

    def compute_primal(self, wing_mass, fuse_mass, tail_mass):

        structure_mass = wing_mass + fuse_mass + tail_mass

        return structure_mass