import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class ElectricalMass(om.ExplicitComponent):
    """
    Computes the mass of the electrical subsystems. The methodology is based on
    the GASP weight equations, modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, units='lbm')

        add_aviary_output(self, Aircraft.Electrical.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        elec_mass_coeff = inputs[Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER] * GRAV_ENGLISH_LBM

        if PAX <= 12:
            electrical_wt = 0.03217 * gross_wt_initial - 20.0
        else:
            if num_engines == 1:
                electrical_wt = 0.00778 * gross_wt_initial + 33.0
            else:
                electrical_wt = elec_mass_coeff * PAX + 170.0

        outputs[Aircraft.Electrical.MASS] = electrical_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]

        if PAX <= 12.0:
            delectrical_wt_dgross_wt_initial = 0.03217
            delectrical_wt_delec_mass_coeff = 0.0
        else:
            if num_engines == 1:
                delectrical_wt_dgross_wt_initial = 0.00778
                delectrical_wt_delec_mass_coeff = 0.0
            else:
                delectrical_wt_dgross_wt_initial = 0.0
                delectrical_wt_delec_mass_coeff = PAX * GRAV_ENGLISH_LBM

        J[Aircraft.Electrical.MASS, Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER] = (
            delectrical_wt_delec_mass_coeff
        ) / GRAV_ENGLISH_LBM

        J[Aircraft.Electrical.MASS, Mission.Design.GROSS_MASS] = (
            delectrical_wt_dgross_wt_initial / GRAV_ENGLISH_LBM
        )
