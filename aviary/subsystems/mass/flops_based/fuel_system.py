import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import distributed_engine_count_factor
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class TransportFuelSystemMass(om.ExplicitComponent):
    """
    Component to calculate fuel system mass of Transports. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Mission.Constraints.MAX_MACH)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, units='lbm')

        add_aviary_output(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        scaler = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]
        capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        outputs[Aircraft.Fuel.FUEL_SYSTEM_MASS] = (
            1.07 * capacity**0.58 * num_eng_fact**0.43 * max_mach**0.34 * scaler
        ) / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        scaler = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]
        capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_mach = self.options[Mission.Constraints.MAX_MACH]

        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER] = (
            1.07 * capacity**0.58 * num_eng_fact**0.43 * max_mach**0.34 / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.TOTAL_CAPACITY] = (
            1.07
            * 0.58
            * capacity**-0.42
            * num_eng**0.43
            * max_mach**0.34
            * scaler
            / GRAV_ENGLISH_LBM
        )


class AltFuelSystemMass(om.ExplicitComponent):
    """
    Component for an alternate way to calculate fuel system mass. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuel.NUM_TANKS)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        number_of_fuel_tanks = self.options[Aircraft.Fuel.NUM_TANKS]
        total_fuel_capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        scaler = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]

        fuel_sys_weight = (
            978.6 * (number_of_fuel_tanks / 13.0)
            + 2283.4 * (total_fuel_capacity / 208100.0) ** (2.0 / 3.0)
            + 350.0
            + 0.00029 * total_fuel_capacity
        ) * scaler

        outputs[Aircraft.Fuel.FUEL_SYSTEM_MASS] = fuel_sys_weight / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        number_of_fuel_tanks = self.options[Aircraft.Fuel.NUM_TANKS]
        total_fuel_capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        scaler = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER]

        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.TOTAL_CAPACITY] = (
            (
                2283.4
                * (2.0 / 3.0)
                * (1 / 208100.0) ** (2.0 / 3.0)
                * total_fuel_capacity ** (-1.0 / 3.0)
                + 0.00029
            )
            * scaler
            / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuel.FUEL_SYSTEM_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER] = (
            0.3 * 3262.0 * (number_of_fuel_tanks / 13.0)
            + 0.7 * 3262.0 * (total_fuel_capacity / 208100.0) ** (2.0 / 3.0)
            + 350.0
            + 0.00029 * total_fuel_capacity
        ) / GRAV_ENGLISH_LBM
