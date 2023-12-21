from dymos.models.atmosphere.atmos_1976 import USatm1976Comp
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import add_opts2vals, create_opts2vals

from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.phases.taxi_component import TaxiFuelComponent
from aviary.subsystems.propulsion.propulsion_mission import \
    PropulsionMission
from aviary.variable_info.variables import Dynamic, Mission


class TaxiSegment(BaseODE):
    def setup(self):
        options: AviaryValues = self.options['aviary_options']
        self.add_subsystem("params", ParamPort(), promotes=["*"])
        self.add_subsystem(
            "USatm",
            USatm1976Comp(num_nodes=1),
            promotes_inputs=[("h", Mission.Takeoff.AIRPORT_ALTITUDE)],
            promotes_outputs=[("temp", Dynamic.Mission.TEMPERATURE),
                              ("pres", Dynamic.Mission.STATIC_PRESSURE)],
        )

        add_opts2vals(self, create_opts2vals(
            [Mission.Taxi.MACH]), options)

        self.add_subsystem(
            name='propulsion',
            subsys=PropulsionMission(
                num_nodes=1,
                aviary_options=options,
            ),
            promotes_inputs=['*', (Dynamic.Mission.ALTITUDE, Mission.Takeoff.AIRPORT_ALTITUDE),
                             (Dynamic.Mission.MACH, Mission.Taxi.MACH)],
            promotes_outputs=['*'])

        self.add_subsystem("taxifuel", TaxiFuelComponent(
            aviary_options=options), promotes=["*"])

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Mission.Taxi.MACH, 0)

        # Throttle Idle
        self.set_input_defaults('throttle', 0.0)
