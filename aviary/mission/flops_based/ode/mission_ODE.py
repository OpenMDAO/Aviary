import numpy as np

import openmdao.api as om
from aviary.subsystems.atmosphere.atmosphere import Atmosphere

from aviary.mission.flops_based.ode.mission_EOM import MissionEOM
from aviary.mission.gasp_based.ode.time_integration_base_classes import add_SGM_required_inputs, add_SGM_required_outputs
from aviary.subsystems.propulsion.throttle_allocation import ThrottleAllocator
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.enums import AnalysisScheme, ThrottleAllocation, SpeedType
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class ExternalSubsystemGroup(om.Group):
    """
    For external subsystem group, promote relevant aircraft and mission variables.
    """

    def configure(self):
        promote_aircraft_and_mission_vars(self)


class MissionODE(om.Group):
    """Define the ODE of motion"""

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int,
            desc='Number of nodes to be evaluated in the RHS')
        self.options.declare(
            'subsystem_options', types=dict, default={},
            desc='dictionary of parameters to be passed to the subsystem builders')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')
        self.options.declare(
            'core_subsystems',
            desc='list of core subsystem builder instances to be added to the ODE'
        )
        self.options.declare(
            'external_subsystems', default=[],
            desc='list of external subsystem builder instances to be added to the ODE')
        self.options.declare(
            'meta_data', default=_MetaData,
            desc='metadata associated with the variables to be passed into the ODE')
        self.options.declare(
            'use_actual_takeoff_mass', default=False,
            desc='flag to use actual takeoff mass in the climb phase, otherwise assume 100 kg fuel burn')
        self.options.declare(
            'throttle_enforcement', default='path_constraint',
            values=['path_constraint', 'boundary_constraint', 'bounded', None],
            desc='flag to enforce throttle constraints on the path or at the segment boundaries or using solver bounds'
        )
        self.options.declare(
            'throttle_allocation', default=ThrottleAllocation.FIXED,
            types=ThrottleAllocation,
            desc='Flag that determines how to handle throttles for multiple engines.'
        )
        self.options.declare(
            "analysis_scheme",
            default=AnalysisScheme.COLLOCATION,
            types=AnalysisScheme,
            desc="The analysis method that will be used to close the trajectory; for example collocation or time integration",
        )

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        analysis_scheme = options['analysis_scheme']
        aviary_options = options['aviary_options']
        core_subsystems = options['core_subsystems']
        subsystem_options = options['subsystem_options']
        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))

        if analysis_scheme is AnalysisScheme.SHOOTING:
            SGM_required_inputs = {
                't_curr': {'units': 's'},
                Dynamic.Mission.DISTANCE: {'units': 'm'},
            }
            add_SGM_required_inputs(self, SGM_required_inputs)

        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=nn, input_speed_type=SpeedType.MACH),
            promotes=['*'],
        )

        # add execcomp to compute velocity_rate based off mach_rate and sos
        self.add_subsystem(
            name='velocity_rate_comp',
            subsys=om.ExecComp(
                'velocity_rate = mach_rate * sos',
                mach_rate={'units': 'unitless/s', 'shape': (nn,)},
                sos={'units': 'm/s', 'shape': (nn,)},
                velocity_rate={'units': 'm/s**2', 'shape': (nn,)},
                has_diag_partials=True,
            ),
            promotes_inputs=[('mach_rate', Dynamic.Mission.MACH_RATE),
                             ('sos', Dynamic.Mission.SPEED_OF_SOUND)],
            promotes_outputs=[('velocity_rate', Dynamic.Mission.VELOCITY_RATE)],
        )

        base_options = {'num_nodes': nn, 'aviary_inputs': aviary_options}

        for subsystem in core_subsystems:
            # check if subsystem_options has entry for a subsystem of this name
            if subsystem.name in subsystem_options:
                kwargs = subsystem_options[subsystem.name]
            else:
                kwargs = {}

            kwargs.update(base_options)
            system = subsystem.build_mission(**kwargs)

            if system is not None:
                self.add_subsystem(subsystem.name,
                                   system,
                                   promotes_inputs=subsystem.mission_inputs(**kwargs),
                                   promotes_outputs=subsystem.mission_outputs(**kwargs))

        # Create a lightly modified version of an OM group to add external subsystems
        # to the ODE with a special configure() method that promotes
        # all aircraft:* and mission:* variables to the ODE.
        external_subsystem_group = ExternalSubsystemGroup()
        add_subsystem_group = False

        for subsystem in self.options['external_subsystems']:
            subsystem_mission = subsystem.build_mission(
                num_nodes=nn, aviary_inputs=aviary_options)
            if subsystem_mission is not None:
                add_subsystem_group = True
                external_subsystem_group.add_subsystem(subsystem.name,
                                                       subsystem_mission)

        # Only add the external subsystem group if it has at least one subsystem.
        # Without this logic there'd be an empty OM group added to the ODE.
        if add_subsystem_group:
            self.add_subsystem(
                name='external_subsystems',
                subsys=external_subsystem_group,
                promotes_inputs=['*'],
                promotes_outputs=['*'])

        self.add_subsystem(
            name='mission_EOM',
            subsys=MissionEOM(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS,
                Dynamic.Mission.THRUST_MAX_TOTAL,
                Dynamic.Mission.DRAG, Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE],
            promotes_outputs=[
                Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                Dynamic.Mission.ALTITUDE_RATE_MAX,
                Dynamic.Mission.DISTANCE_RATE,
                'thrust_required',
            ])

        # THROTTLE Section
        # TODO: Split this out into a function that can be used by the other ODEs.
        if num_engine_type > 1:

            # Multi Engine

            self.add_subsystem(name='throttle_balance',
                               subsys=om.BalanceComp(name="aggregate_throttle",
                                                     units="unitless",
                                                     val=np.ones((nn, )),
                                                     lhs_name='thrust_required',
                                                     rhs_name=Dynamic.Mission.THRUST_TOTAL,
                                                     eq_units="lbf",
                                                     normalize=False,
                                                     res_ref=1.0e6,
                                                     ),
                               promotes_inputs=['*'],
                               promotes_outputs=['*'])

            self.add_subsystem(
                "throttle_allocator",
                ThrottleAllocator(
                    num_nodes=nn,
                    aviary_options=aviary_options,
                    throttle_allocation=self.options['throttle_allocation']
                ),
                promotes_inputs=['*'],
                promotes_outputs=['*']
            )

        else:

            # Single Engine

            # Add a balance comp to compute throttle based on the required thrust.
            self.add_subsystem(name='throttle_balance',
                               subsys=om.BalanceComp(name=Dynamic.Mission.THROTTLE,
                                                     units="unitless",
                                                     val=np.ones((nn, )),
                                                     lhs_name='thrust_required',
                                                     rhs_name=Dynamic.Mission.THRUST_TOTAL,
                                                     eq_units="lbf",
                                                     normalize=False,
                                                     lower=0.0
                                                     if options['throttle_enforcement'] == 'bounded'
                                                     else None,
                                                     upper=1.0
                                                     if options['throttle_enforcement'] == 'bounded'
                                                     else None,
                                                     res_ref=1.0e6,
                                                     ),
                               promotes_inputs=['*'],
                               promotes_outputs=['*'])

            self.set_input_defaults(Dynamic.Mission.THROTTLE, val=1.0, units='unitless')

        self.set_input_defaults(Dynamic.Mission.MACH, val=np.ones(nn), units='unitless')
        self.set_input_defaults(Dynamic.Mission.MASS, val=np.ones(nn), units='kg')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.ones(nn), units='m/s')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='m')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE_RATE,
                                val=np.ones(nn), units='m/s')

        if options['use_actual_takeoff_mass']:
            exec_comp_string = 'initial_mass_residual = initial_mass - mass[0]'
            initial_mass_string = Mission.Takeoff.FINAL_MASS
        else:
            exec_comp_string = 'initial_mass_residual = initial_mass - mass[0] - 100.'
            initial_mass_string = Mission.Summary.GROSS_MASS

        # Experimental: Add a component to constrain the initial mass to be equal to design gross weight.
        initial_mass_residual_constraint = om.ExecComp(
            exec_comp_string,
            initial_mass={'units': 'kg'},
            mass={'units': 'kg', 'shape': (nn,)},
            initial_mass_residual={'units': 'kg', 'res_ref': 1.0e5},
        )

        self.add_subsystem('initial_mass_residual_constraint', initial_mass_residual_constraint,
                           promotes_inputs=[
                               ('initial_mass', initial_mass_string),
                               ('mass', Dynamic.Mission.MASS)
                           ],
                           promotes_outputs=['initial_mass_residual'])

        if analysis_scheme is AnalysisScheme.SHOOTING:
            SGM_required_outputs = {
                Dynamic.Mission.ALTITUDE_RATE: {'units': 'm/s'},
            }
            add_SGM_required_outputs(self, SGM_required_outputs)

        print_level = 0 if analysis_scheme is AnalysisScheme.SHOOTING else 2

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True,
                                                atol=1.0e-10,
                                                rtol=1.0e-10,
                                                )
        self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self.linear_solver = om.DirectSolver(assemble_jac=True)
        self.nonlinear_solver.options['err_on_non_converge'] = True
        self.nonlinear_solver.options['iprint'] = print_level
