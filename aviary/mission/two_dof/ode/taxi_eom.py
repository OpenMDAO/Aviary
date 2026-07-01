import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Dynamic, Mission


class TaxiFuelComponent(om.ExplicitComponent):
    """Compute the fuel consumed during taxi and update the mass after taxi in a 2DOF mission."""

    def initialize(self):
        add_aviary_option(self, Mission.Taxi.DURATION, units='s')

    def setup(self):
        add_aviary_input(
            self,
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
            units='lbm/s',
        )
        add_aviary_input(self, Mission.GROSS_MASS)

        self.add_output(
            'taxi_fuel_consumed',
            val=1.0,
            units='lbm',
            desc='taxi_fuel_consumed',
        )
        add_aviary_output(
            self,
            Dynamic.Vehicle.MASS,
            units='lbm',
            desc='mass after taxi',
        )

    def setup_partials(self):
        self.declare_partials(
            'taxi_fuel_consumed',
            [Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL],
        )
        self.declare_partials(
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
        )
        self.declare_partials(Dynamic.Vehicle.MASS, Mission.GROSS_MASS, val=1)

    def compute(self, inputs, outputs):
        fuelflow, takeoff_mass = inputs.values()
        dt_taxi, _ = self.options[Mission.Taxi.DURATION]
        outputs['taxi_fuel_consumed'] = -fuelflow * dt_taxi
        outputs[Dynamic.Vehicle.MASS] = takeoff_mass - outputs['taxi_fuel_consumed']

    def compute_partials(self, inputs, J):
        dt_taxi, _ = self.options[Mission.Taxi.DURATION]

        J[
            'taxi_fuel_consumed',
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
        ] = -dt_taxi

        J[
            Dynamic.Vehicle.MASS,
            Dynamic.Vehicle.Propulsion.FUEL_MASS_FLOW_RATE_NEGATIVE_TOTAL,
        ] = dt_taxi
