import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import distributed_engine_count_factor
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class TransportInstrumentMass(om.ExplicitComponent):
    """
    Calculates mass of instrument group for transports and GA aircraft.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    ASSUMPTIONS: All engines have instrument mass that follows this equation
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)
        add_aviary_option(self, Mission.Constraints.MAX_MACH)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Instruments.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Instruments.MASS, units='lbm')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        num_crew = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        num_wing_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES]
        num_fuse_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES]
        num_wing_eng_fact = distributed_engine_count_factor(num_wing_eng)
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)

        fuse_area = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        max_mach = self.options[Mission.Constraints.MAX_MACH]
        mass_scaler = inputs[Aircraft.Instruments.MASS_SCALER]

        instrument_weight = (
            0.48
            * fuse_area**0.57
            * max_mach**0.5
            * (10.0 + 2.5 * num_crew + num_wing_eng_fact + 1.5 * num_fuse_eng_fact)
        )

        outputs[Aircraft.Instruments.MASS] = instrument_weight * mass_scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        num_crew = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        num_wing_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES]
        num_fuse_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES]
        num_wing_eng_fact = distributed_engine_count_factor(num_wing_eng)
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)

        fuse_area = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        max_mach = self.options[Mission.Constraints.MAX_MACH]
        mass_scaler = inputs[Aircraft.Instruments.MASS_SCALER]

        fact = 10.0 + 2.5 * num_crew + num_wing_eng_fact + 1.5 * num_fuse_eng_fact
        area_fact = fuse_area**0.57
        mach_fact = max_mach**0.5

        J[Aircraft.Instruments.MASS, Aircraft.Fuselage.PLANFORM_AREA] = (
            0.2736 * fuse_area**-0.43 * mach_fact * fact * mass_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Instruments.MASS, Aircraft.Instruments.MASS_SCALER] = (
            0.48 * area_fact * mach_fact * fact / GRAV_ENGLISH_LBM
        )
