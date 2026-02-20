import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class InstrumentMass(om.ExplicitComponent):
    """
    Calculates mass of instrument group for transports and GA aircraft.
    The methodology is based on the GASP weight equations, modified to
    output mass instead of weight.

    ASSUMPTIONS: All engines have instrument mass that follows this equation
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Instruments.MASS_COEFFICIENT, units='unitless')

        add_aviary_output(self, Aircraft.Instruments.MASS, units='lbm')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        engine_type = self.options[Aircraft.Engine.TYPE][0]

        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        gross_mass_initial = inputs[Mission.Design.GROSS_MASS]
        mass_coefficient = inputs[Aircraft.Instruments.MASS_COEFFICIENT]
        wingspan = inputs[Aircraft.Wing.SPAN]

        num_pilots = 1
        if PAX > 9.0:
            num_pilots = 2
        if engine_type is GASPEngineType.TURBOJET and PAX > 5.0:
            num_pilots = 2
        if PAX >= 351.0:
            num_pilots = 3

        outputs[Aircraft.Instruments.MASS] = (
            mass_coefficient
            * gross_mass_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )

    def compute_partials(self, inputs, J):
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        engine_type = self.options[Aircraft.Engine.TYPE][0]

        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        gross_mass_initial = inputs[Mission.Design.GROSS_MASS]
        mass_coefficient = inputs[Aircraft.Instruments.MASS_COEFFICIENT]
        wingspan = inputs[Aircraft.Wing.SPAN]

        num_pilots = 1
        if PAX > 9.0:
            num_pilots = 2
        if engine_type is GASPEngineType.TURBOJET and PAX > 5.0:
            num_pilots = 2
        if PAX >= 351.0:
            num_pilots = 3

        dinstrument_wt_dmass_coeff_1 = (
            gross_mass_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )
        J[Aircraft.Instruments.MASS, Aircraft.Instruments.MASS_COEFFICIENT] = (
            dinstrument_wt_dmass_coeff_1
        )

        dinstrument_wt_dgross_wt_initial = (
            0.386
            * mass_coefficient
            * gross_mass_initial ** (0.386 - 1)
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan**0.696
        )
        J[Aircraft.Instruments.MASS, Mission.Design.GROSS_MASS] = dinstrument_wt_dgross_wt_initial

        dinstrument_wt_dfus_len = (
            0.05
            * mass_coefficient
            * gross_mass_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len ** (0.05 - 1)
            * wingspan**0.696
        )
        J[Aircraft.Instruments.MASS, Aircraft.Fuselage.LENGTH] = dinstrument_wt_dfus_len

        dinstrument_wt_dwingspan = (
            0.696
            * mass_coefficient
            * gross_mass_initial**0.386
            * num_engines**0.687
            * num_pilots**0.31
            * fus_len**0.05
            * wingspan ** (0.696 - 1)
        )
        J[Aircraft.Instruments.MASS, Aircraft.Wing.SPAN] = dinstrument_wt_dwingspan
