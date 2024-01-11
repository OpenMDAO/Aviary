import numpy as np
import openmdao.api as om
from dymos.models.atmosphere import USatm1976Comp

from aviary.mission.flops_based.ode.mission_EOM import MissionEOM
from aviary.subsystems.aerodynamics.flops_based.mach_number import MachNumber

from aviary.mission.gasp_based.ode.time_integration_base_classes import add_SGM_required_inputs
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic, Mission, Aircraft
from aviary.variable_info.variables_in import VariablesIn
from aviary.variable_info.enums import AnalysisScheme


class ExternalSubsystemGroup(om.Group):
    def configure(self):
        promote_aircraft_and_mission_vars(self)


class MissionODE(om.Group):
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
        engine_count = len(aviary_options.get_val('engine_models'))

        if analysis_scheme is AnalysisScheme.SHOOTING:
            SGM_required_inputs = {
                't_curr': {'units': 's'},
                Dynamic.Mission.RANGE: {'units': 'm'},
            }
            add_SGM_required_inputs(self, SGM_required_inputs)

        self.add_subsystem(
            'input_port',
            VariablesIn(aviary_options=aviary_options,
                        meta_data=self.options['meta_data'],
                        context='mission'),
            promotes_inputs=['*'],
            promotes_outputs=['*'])
        self.add_subsystem(
            name='atmosphere',
            subsys=USatm1976Comp(num_nodes=nn),
            promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                ('sos', Dynamic.Mission.SPEED_OF_SOUND), ('rho', Dynamic.Mission.DENSITY),
                ('temp', Dynamic.Mission.TEMPERATURE), ('pres', Dynamic.Mission.STATIC_PRESSURE)])
        self.add_subsystem(
            name=Dynamic.Mission.MACH,
            subsys=MachNumber(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.VELOCITY, Dynamic.Mission.SPEED_OF_SOUND],
            promotes_outputs=[Dynamic.Mission.MACH])

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
                external_subsystem_group.add_subsystem(subsystem.name, subsystem_mission)

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
                Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.THRUST_MAX_TOTAL,
                Dynamic.Mission.DRAG, Dynamic.Mission.VELOCITY_RATE],
            promotes_outputs=[
                Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.ALTITUDE_RATE_MAX,
                Dynamic.Mission.RANGE_RATE])

        self.set_input_defaults(Mission.Design.GROSS_MASS, val=1, units='kg')
        self.set_input_defaults(
            Aircraft.Fuselage.CHARACTERISTIC_LENGTH, val=1, units='ft')
        self.set_input_defaults(Aircraft.Fuselage.FINENESS, val=1, units='unitless')
        self.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, val=1, units='ft**2')
        self.set_input_defaults(
            Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, val=1, units='ft')
        self.set_input_defaults(Aircraft.VerticalTail.FINENESS, val=1, units='unitless')
        self.set_input_defaults(Aircraft.VerticalTail.WETTED_AREA, val=1, units='ft**2')
        self.set_input_defaults(
            Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, val=1, units='ft')
        self.set_input_defaults(Aircraft.HorizontalTail.FINENESS,
                                val=1, units='unitless')
        self.set_input_defaults(
            Aircraft.HorizontalTail.WETTED_AREA, val=1, units='ft**2')
        self.set_input_defaults(Aircraft.Wing.CHARACTERISTIC_LENGTH, val=1, units='ft')
        self.set_input_defaults(Aircraft.Wing.FINENESS, val=1, units='unitless')
        self.set_input_defaults(Aircraft.Wing.WETTED_AREA, val=1, units='ft**2')
        self.set_input_defaults(
            Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, val=1, units='unitless')
        self.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=1, units='unitless')
        self.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD,
                                val=1, units='unitless')
        self.set_input_defaults(Aircraft.Wing.SWEEP, val=1, units='deg')
        self.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=1, units='unitless')
        self.set_input_defaults(
            Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, val=1, units='unitless')
        self.set_input_defaults(
            Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, val=1, units='unitless')

        self.set_input_defaults(Dynamic.Mission.MASS, val=np.ones(nn), units='kg')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.ones(nn), units='m/s')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='m')
        self.set_input_defaults(
            Dynamic.Mission.THROTTLE, val=np.ones((nn, engine_count)),
            units='unitless')

        if options['use_actual_takeoff_mass']:
            exec_comp_string = 'initial_mass_residual = initial_mass - mass[0]'
            initial_mass_string = Mission.Takeoff.FINAL_MASS
        else:
            exec_comp_string = 'initial_mass_residual = initial_mass - mass[0] - 100.'
            initial_mass_string = Mission.Design.GROSS_MASS

        # Experimental: Add a component to constrain the initial mass to be equal to design gross weight.
        initial_mass_residual_constraint = om.ExecComp(
            exec_comp_string,
            initial_mass={'units': 'kg'},
            mass={'units': 'kg', 'shape': (nn,)},
            initial_mass_residual={'units': 'kg'},
        )

        self.add_subsystem('initial_mass_residual_constraint', initial_mass_residual_constraint,
                           promotes_inputs=[
                               ('initial_mass', initial_mass_string),
                               ('mass', Dynamic.Mission.MASS)
                           ],
                           promotes_outputs=['initial_mass_residual'])

        if analysis_scheme is AnalysisScheme.SHOOTING and False:
            from aviary.utils.functions import create_printcomp
            dummy_comp = create_printcomp(
                all_inputs=[
                    't_curr',
                    Mission.Design.RESERVE_FUEL,
                    Dynamic.Mission.MASS,
                    Dynamic.Mission.RANGE,
                    Dynamic.Mission.ALTITUDE,
                    Dynamic.Mission.FLIGHT_PATH_ANGLE,
                ],
                input_units={
                    't_curr': 's',
                    Dynamic.Mission.FLIGHT_PATH_ANGLE: 'deg',
                    Dynamic.Mission.RANGE: 'NM',
                })
            self.add_subsystem(
                "dummy_comp",
                dummy_comp(),
                promotes_inputs=["*"],)
            self.set_input_defaults(
                Dynamic.Mission.RANGE, val=0, units='NM')
            self.set_input_defaults('t_curr', val=0, units='s')
