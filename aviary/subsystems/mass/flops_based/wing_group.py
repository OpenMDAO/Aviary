import numpy as np
import openmdao.api as om

from aviary.subsystems.mass.flops_based.engine_pod import EnginePodMass
from aviary.subsystems.mass.flops_based.wing_common import (
    WingBendingMass, WingMiscMass, WingShearControlMass, WingTotalMass)
from aviary.subsystems.mass.flops_based.wing_detailed import DetailedWingBendingFact
from aviary.subsystems.mass.flops_based.wing_simple import SimpleWingBendingFact
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class WingMassGroup(om.Group):
    """
    Group of components used for FLOPS-based wing mass computation.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']

        self.add_subsystem('engine_pod_mass',
                           EnginePodMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        if Aircraft.Wing.INPUT_STATION_DIST in aviary_options:
            self.add_subsystem('wing_bending_factor',
                               DetailedWingBendingFact(aviary_options=aviary_options),
                               promotes_inputs=['*'], promotes_outputs=['*'])

        else:
            self.add_subsystem('wing_bending_factor',
                               SimpleWingBendingFact(aviary_options=aviary_options),
                               promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('wing_misc',
                           WingMiscMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('wing_shear_control',
                           WingShearControlMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('wing_bending',
                           WingBendingMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('wing_total',
                           WingTotalMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])
