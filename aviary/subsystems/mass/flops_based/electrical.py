import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import distributed_engine_count_factor
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class ElectricalMass(om.ExplicitComponent):
    """
    Computes the mass of the electrical subsystems. The methodology is based on
    the FLOPS weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Fuselage.NUM_FUSELAGES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Electrical.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Electrical.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        nfuse = self.options[Aircraft.Fuselage.NUM_FUSELAGES]
        ncrew = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        npass = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        length = inputs[Aircraft.Fuselage.LENGTH]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines_factor = distributed_engine_count_factor(num_eng)
        mass_scaler = inputs[Aircraft.Electrical.MASS_SCALER]

        outputs[Aircraft.Electrical.MASS] = (
            92.0
            * length**0.4
            * width**0.14
            * nfuse**0.27
            * num_engines_factor**0.69
            * (1.0 + 0.044 * ncrew + 0.0015 * npass)
            * mass_scaler
            / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        nfuse = self.options[Aircraft.Fuselage.NUM_FUSELAGES]
        ncrew = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        npass = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        length = inputs[Aircraft.Fuselage.LENGTH]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_engines_factor = distributed_engine_count_factor(num_eng)
        mass_scaler = inputs[Aircraft.Electrical.MASS_SCALER]

        fact = 92.0 * nfuse**0.27 * (1.0 + 0.044 * ncrew + 0.0015 * npass)
        length_fact = length**0.4
        width_fact = width**0.14
        ecf_fact = num_engines_factor**0.69

        J[Aircraft.Electrical.MASS, Aircraft.Fuselage.LENGTH] = (
            0.4 * length**-0.6 * fact * width_fact * ecf_fact * mass_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Electrical.MASS, Aircraft.Fuselage.MAX_WIDTH] = (
            0.14 * width**-0.86 * fact * length_fact * ecf_fact * mass_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Electrical.MASS, Aircraft.Electrical.MASS_SCALER] = (
            fact * length_fact * ecf_fact * width_fact / GRAV_ENGLISH_LBM
        )


class AltElectricalMass(om.ExplicitComponent):
    """Computes the mass of the electrical subsystems using the alternate method."""

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.Electrical.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Electrical.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        npass = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        mass_scaler = inputs[Aircraft.Electrical.MASS_SCALER]

        outputs[Aircraft.Electrical.MASS] = 16.3 * npass * mass_scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        npass = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        J[Aircraft.Electrical.MASS, Aircraft.Electrical.MASS_SCALER] = (
            16.3 * npass / GRAV_ENGLISH_LBM
        )
