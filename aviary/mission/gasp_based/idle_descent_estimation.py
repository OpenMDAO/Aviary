import openmdao.api as om

from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.utils.functions import promote_aircraft_and_mission_vars
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Aircraft, Dynamic


def add_descent_estimation_as_submodel(
    main_prob: om.Problem,
    subsys_name='idle_descent_estimation',
    phases=None,
    ode_args=None,
    initial_mass=None,
    cruise_alt=None,
    cruise_mach=None,
    reserve_fuel=None,
    all_subsystems=None,
    verbosity=Verbosity.QUIET,
):
    """
    This creates a sub model that contains a copy of the descent portion of the mission's trajectory. This is used to calculate an estimation of the fuel burn and distance required for the descent, so that they can be used as triggers for the cruise phase. The sub model is then added to the main problem.
    The user can specify certain initial conditions or requirements such as cruise Mach number, reserve fuel required, etc.
    """
    if phases is None:
        from aviary.interface.default_phase_info.two_dof_fiti import add_default_sgm_args
        from aviary.interface.default_phase_info.two_dof_fiti import descent_phases as phases

        add_default_sgm_args(phases, ode_args)

    if all_subsystems is None:
        all_subsystems = []

    traj = FlexibleTraj(
        Phases=phases,
        traj_initial_state_input=[
            Dynamic.Vehicle.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
        traj_final_state_output=[
            Dynamic.Vehicle.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
        promote_all_auto_ivc=True,
    )

    model = om.Group()

    if isinstance(initial_mass, str):
        model.add_subsystem(
            'top_of_descent_mass',
            om.ExecComp(
                'mass_initial = top_of_descent_mass',
                mass_initial={'units': 'lbm'},
                top_of_descent_mass={'units': 'lbm'},
            ),
            promotes_inputs=['top_of_descent_mass'],
            promotes_outputs=['mass_initial'],
        )
    else:
        model.add_subsystem(
            'top_of_descent_mass',
            om.ExecComp(
                'mass_initial = operating_mass + payload_mass + reserve_fuel + descent_fuel_estimate',
                mass_initial={'units': 'lbm'},
                operating_mass={'units': 'lbm'},
                payload_mass={'units': 'lbm'},
                reserve_fuel={'units': 'lbm', 'val': 0},
                descent_fuel_estimate={'units': 'lbm', 'val': 0},
            ),
            promotes_inputs=[
                ('operating_mass', Aircraft.Design.OPERATING_MASS),
                ('payload_mass', Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS),
                'reserve_fuel',
                # ('reserve_fuel', Mission.Design.RESERVE_FUEL),
                ('descent_fuel_estimate', 'descent_fuel'),
            ],
            promotes_outputs=['mass_initial'],
        )

    all_bus_vars = set()
    for subsystem in all_subsystems:
        bus_vars = subsystem.get_pre_mission_bus_variables()
        for var, data in bus_vars.items():
            mission_variable_name = data['mission_name']
            if not isinstance(mission_variable_name, list):
                mission_variable_name = [mission_variable_name]
            for mission_var_name in mission_variable_name:
                all_bus_vars.add(mission_var_name)

    model.add_subsystem(
        'descent_traj',
        traj,
        promotes_inputs=['altitude_initial', 'mass_initial', 'aircraft:*']
        + [(var, 'parameters:' + var) for var in all_bus_vars],
        promotes_outputs=['mass_final', 'distance_final'],
    )

    model.add_subsystem(
        'actual_fuel_burn',
        om.ExecComp(
            'actual_fuel_burn = mass_initial - mass_final',
            actual_fuel_burn={'units': 'lbm'},
            mass_initial={'units': 'lbm'},
            mass_final={'units': 'lbm'},
        ),
        promotes_inputs=[
            'mass_initial',
            'mass_final',
        ],
        promotes_outputs=[('actual_fuel_burn', 'descent_fuel')],
    )

    if verbosity >= Verbosity.BRIEF:
        from aviary.utils.functions import create_printcomp

        dummy_comp = create_printcomp(
            all_inputs=[
                Aircraft.Design.OPERATING_MASS,
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
                'descent_fuel',
                'reserve_fuel',
                'mass_initial',
                'distance_final',
            ],
            input_units={
                'descent_fuel': 'lbm',
                'reserve_fuel': 'lbm',
                'mass_initial': 'lbm',
                'distance_final': 'nmi',
            },
        )
        model.add_subsystem(
            'dummy_comp',
            dummy_comp(),
            promotes_inputs=['*'],
        )
        model.set_input_defaults('reserve_fuel', 0, 'lbm')
        model.set_input_defaults('mass_initial', 0, 'lbm')

    model.add_objective('descent_fuel', ref=1e4)

    model.linear_solver = om.DirectSolver(assemble_jac=True)
    model.nonlinear_solver = om.NonlinearBlockGS(iprint=3, rtol=1e-2, maxiter=5)

    input_aliases = []
    if isinstance(initial_mass, str):
        input_aliases.append(('top_of_descent_mass', initial_mass))
    elif isinstance(initial_mass, (int, float)):
        model.set_input_defaults('mass_initial', initial_mass)

    if isinstance(cruise_alt, str):
        input_aliases.append(('altitude_initial', cruise_alt))
    elif isinstance(cruise_alt, (int, float)):
        model.set_input_defaults('altitude_initial', cruise_alt)

    if isinstance(reserve_fuel, str):
        input_aliases.append(('reserve_fuel', reserve_fuel))
    elif isinstance(reserve_fuel, (int, float)):
        model.set_input_defaults('reserve_fuel', reserve_fuel)

    model.set_input_defaults(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 0)
    model.set_input_defaults(Aircraft.Design.OPERATING_MASS, val=0, units='lbm')
    model.set_input_defaults('descent_traj.' + Dynamic.Vehicle.Propulsion.THROTTLE, 0)

    promote_aircraft_and_mission_vars(model)

    subprob = om.Problem(model=model)
    subcomp = om.SubmodelComp(
        problem=subprob,
        inputs=[
            'aircraft:*',
        ],
        outputs=['distance_final', 'descent_fuel', 'mass_initial'],
        do_coloring=False,
    )

    main_prob.model.add_subsystem(
        subsys_name,
        subcomp,
        promotes_inputs=[
            'aircraft:*',
        ]
        + input_aliases,
        promotes_outputs=[
            ('distance_final', 'descent_range'),
            'descent_fuel',
            ('mass_initial', 'start_of_descent_mass'),
        ],
    )
