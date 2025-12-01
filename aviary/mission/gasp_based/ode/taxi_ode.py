import numpy as np
import openmdao.api as om

from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.taxi_eom import TaxiFuelComponent
from aviary.mission.gasp_based.ode.two_dof_ode import TwoDOFODE
from aviary.subsystems.propulsion.propulsion_builder import PropulsionBuilderBase
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import add_opts2vals, create_opts2vals
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TaxiSegment(TwoDOFODE):
    """ODE for taxi phase of a 2DOF mission."""

    def setup(self):
        options: AviaryValues = self.options['aviary_options']
        core_subsystems = self.options['core_subsystems']

        self.add_subsystem('params', ParamPort(), promotes=['*'])

        add_opts2vals(self, create_opts2vals([Mission.Taxi.MACH]), options)

        alias_comp = om.ExecComp(
            'alt=airport_alt',
            alt={
                'val': np.zeros(1),
                'units': 'ft',
            },
            airport_alt={'val': np.zeros(1), 'units': 'ft'},
            has_diag_partials=True,
        )

        alias_comp.add_expr(
            'mach=taxi_mach',
            mach={'val': np.zeros(1), 'units': 'unitless'},
            taxi_mach={'val': np.zeros(1), 'units': 'unitless'},
        )

        self.add_subsystem(
            'alias_taxi_phase',
            alias_comp,
            promotes_inputs=[
                ('airport_alt', Mission.Takeoff.AIRPORT_ALTITUDE),
                ('taxi_mach', Mission.Taxi.MACH),
            ],
            promotes_outputs=[
                ('alt', Dynamic.Mission.ALTITUDE),
                ('mach', Dynamic.Atmosphere.MACH),
            ],
        )

        self.add_atmosphere(input_speed_type=SpeedType.MACH)

        for subsystem in core_subsystems:
            if isinstance(subsystem, PropulsionBuilderBase):
                system = subsystem.build_mission(num_nodes=1, aviary_inputs=options)

                self.add_subsystem(
                    subsystem.name,
                    system,
                    promotes_inputs=[
                        '*',
                        (Dynamic.Mission.ALTITUDE, Mission.Takeoff.AIRPORT_ALTITUDE),
                        (Dynamic.Atmosphere.MACH, Mission.Taxi.MACH),
                    ],
                    promotes_outputs=['*'],
                )

        self.add_external_subsystems()

        self.add_subsystem('taxifuel', TaxiFuelComponent(), promotes=['*'])

        ParamPort.set_default_vals(self)
        self.set_input_defaults(Mission.Taxi.MACH, 0)

        # Throttle Idle
        num_engine_types = len(options.get_val(Aircraft.Engine.NUM_ENGINES))
        self.set_input_defaults(
            Dynamic.Vehicle.Propulsion.THROTTLE, np.zeros((1, num_engine_types))
        )
