from aviary.mission.gasp_based.ode.base_ode import BaseODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.phases.taxi_component import TaxiFuelComponent
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilderBase
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import add_opts2vals, create_opts2vals
from aviary.variable_info.variables import Dynamic, Mission


class TaxiSegment(BaseODE):
    """ODE for taxi phase of a 2DOF mission"""

    def setup(self):
        options: AviaryValues = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        self.add_subsystem("params", ParamPort(), promotes=["*"])

        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=1),
            promotes=[
                '*',
                (Dynamic.Mission.ALTITUDE, Mission.Takeoff.AIRPORT_ALTITUDE),
            ],
        )

        add_opts2vals(self, create_opts2vals(
            [Mission.Taxi.MACH]), options)

        for subsystem in core_subsystems:
            if isinstance(subsystem, PropulsionBuilderBase):
                system = subsystem.build_mission(num_nodes=1, aviary_inputs=options)

                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=['*', (Dynamic.Mission.ALTITUDE, Mission.Takeoff.AIRPORT_ALTITUDE),
                                                    (Dynamic.Mission.MACH, Mission.Taxi.MACH)],
                                   promotes_outputs=['*'])

        self.add_subsystem("taxifuel", TaxiFuelComponent(
            aviary_options=options), promotes=["*"])

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Mission.Taxi.MACH, 0)

        # Throttle Idle
        self.set_input_defaults('throttle', 0.0)
