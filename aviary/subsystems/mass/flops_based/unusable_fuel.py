import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import (
    distributed_engine_count_factor,
    distributed_thrust_factor,
)
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportUnusableFuelMass(om.ExplicitComponent):
    """
    Calculates the mass of unusable fuel using the default method.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    Assumptions
    -----------.
    All engines use fuel, single engine model on aircraft (both engine level thrust
    (scaled by num_engines) and propulsion-level count factor is being used)
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuel.NUM_TANKS)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units='lbm/galUS')
        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_output(self, Aircraft.Fuel.TOTAL_VOLUME, units='galUS')

        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuel.TOTAL_VOLUME, [Aircraft.Fuel.TOTAL_CAPACITY, Aircraft.Fuel.DENSITY]
        )

        self.declare_partials(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS,
            [
                Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER,
                Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
                Aircraft.Wing.AREA,
                Aircraft.Fuel.TOTAL_CAPACITY,
                Aircraft.Fuel.DENSITY,
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        tank_count = self.options[Aircraft.Fuel.NUM_TANKS]
        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER]
        # Calculate fuel density ratio relative to Jet A: 6.7 lbm/galUS = 50.11948 lbm/ft**3
        density_ratio = inputs[Aircraft.Fuel.DENSITY] / 6.7
        total_capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_eng)
        wing_area = inputs[Aircraft.Wing.AREA]

        # TODO: check if this is really a volume
        # outputs[Aircraft.Fuel.TOTAL_VOLUME] = total_capacity / density_ratio

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS] = (
            (
                (
                    11.5 * num_eng_fact * thrust_factor**0.2
                    + 0.07 * wing_area
                    + 1.6 * tank_count * total_capacity**0.28
                )
                * density_ratio
            )
            * scaler
            / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        tank_count = self.options[Aircraft.Fuel.NUM_TANKS]
        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER]
        density_ratio = inputs[Aircraft.Fuel.DENSITY] / 6.7
        total_capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        num_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        num_eng_fact = distributed_engine_count_factor(num_eng)
        max_sls_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        thrust_factor = distributed_thrust_factor(max_sls_thrust, num_eng)
        wing_area = inputs[Aircraft.Wing.AREA]

        term1 = thrust_factor**0.2
        term2 = total_capacity**0.28

        # J[Aircraft.Fuel.TOTAL_VOLUME, Aircraft.Fuel.TOTAL_CAPACITY] = (
        #     1.0 / density_ratio)

        # J[Aircraft.Fuel.TOTAL_VOLUME, Aircraft.Fuel.DENSITY_RATIO] = (
        #     -total_capacity / (density_ratio * density_ratio))

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER] = (
            (11.5 * num_eng_fact * term1 + 0.07 * wing_area + 1.6 * tank_count * term2)
            * density_ratio
            / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = (
            2.3 * thrust_factor**-0.8 * density_ratio * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Wing.AREA] = (
            0.07 * density_ratio * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.TOTAL_CAPACITY] = (
            0.448 * tank_count * total_capacity**-0.72 * density_ratio * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.DENSITY] = (
            ((11.5 * num_eng_fact * term1 + 0.07 * wing_area + 1.6 * tank_count * term2) / 6.7)
            * scaler
            / GRAV_ENGLISH_LBM
        )


class AltUnusableFuelMass(om.ExplicitComponent):
    """
    Calculates the mass of unusable fuel using the alternate method.
    The methodology is based on the FLOPS weight equations, modified
    to output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER, units='unitless')

        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, units='lbm')

        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        total_fuel_capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER]

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS] = 0.0084 * total_fuel_capacity * scaler

    def compute_partials(self, inputs, J):
        total_fuel_capacity = inputs[Aircraft.Fuel.TOTAL_CAPACITY]
        scaler = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER]

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_SCALER] = (
            0.0084 * total_fuel_capacity
        )

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.TOTAL_CAPACITY] = 0.0084 * scaler
