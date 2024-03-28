import warnings

import openmdao.api as om

from aviary.interface.default_phase_info.two_dof_fiti import create_2dof_based_descent_phases
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.variable_info.variables import Aircraft, Mission, Dynamic
from aviary.utils.functions import promote_aircraft_and_mission_vars, set_aviary_initial_values
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables_in import VariablesIn


def descent_range_and_fuel(
    phases=None,
    ode_args=None,
    initial_mass=154e3,
    cruise_alt=35e3,
    cruise_mach=.8,
    empty_weight=85e3,
    payload_weight=30800,
    reserve_fuel=4998,
):

    prob = om.Problem()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = 'IPOPT'
    prob.driver.opt_settings['tol'] = 1.0E-6
    prob.driver.opt_settings['mu_init'] = 1e-5
    prob.driver.opt_settings['max_iter'] = 50
    prob.driver.opt_settings['print_level'] = 5

    if phases is None:
        phases = create_2dof_based_descent_phases(
            ode_args,
            cruise_mach=cruise_mach,
        )

    traj = FlexibleTraj(
        Phases=phases,
        traj_final_state_output=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
        traj_initial_state_input=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
    )
    prob.model = om.Group()
    prob.model.add_subsystem('traj', traj)

    prob.model.add_subsystem(
        "fuel_obj",
        om.ExecComp(
            "reg_objective = overall_fuel/10000",
            reg_objective={"val": 0.0, "units": "unitless"},
            overall_fuel={"units": "lbm"},
        ),
        promotes_inputs=[
            ("overall_fuel", Mission.Summary.TOTAL_FUEL_MASS),
        ],
        promotes_outputs=[("reg_objective", Mission.Objectives.FUEL)],
    )

    prob.model.add_objective(Mission.Objectives.FUEL, ref=1e4)

    prob.setup()
    prob.set_val("traj.altitude_initial", val=cruise_alt, units="ft")
    prob.set_val("traj.mass_initial", val=initial_mass, units="lbm")
    prob.set_val("traj.distance_initial", val=0, units="NM")

    # prevent UserWarning that is displayed when an event is triggered
    warnings.filterwarnings('ignore', category=UserWarning)
    prob.run_model()
    warnings.filterwarnings('default', category=UserWarning)

    final_range = prob.get_val('traj.distance_final', units='NM')[0]
    final_mass = prob.get_val('traj.mass_final', units='lbm')[0]
    descent_fuel = initial_mass - final_mass
    print('final range: ', final_range)
    print('fuel burned: ', descent_fuel)

    initial_mass2 = empty_weight + payload_weight + reserve_fuel + descent_fuel
    prob.set_val("traj.mass_initial", val=initial_mass2, units="lbm")

    # prevent UserWarning that is displayed when an event is triggered
    warnings.filterwarnings('ignore', category=UserWarning)
    prob.run_model()
    warnings.filterwarnings('default', category=UserWarning)

    final_range2 = prob.get_val('traj.distance_final', units='NM')[0]
    final_mass2 = prob.get_val('traj.mass_final', units='lbm')[0]
    descent_fuel2 = initial_mass2 - final_mass2
    print('initial mass: ', initial_mass2)
    print('final range: ', final_range2)
    print('fuel burned: ', initial_mass2-final_mass2)

    results = {
        'initial_guess': {
            'distance_flown': final_range,
            'fuel_burned': descent_fuel
        },
        'refined_guess': {
            'distance_flown': final_range2,
            'fuel_burned': descent_fuel2
        }
    }

    return results


def add_descent_estimation_as_submodel(
        main_prob: om.Problem,
        phases=None,
        ode_args=None,
        initial_mass=None,
        cruise_alt=None,
        cruise_mach=None,
):

    if phases is None:
        phases = create_2dof_based_descent_phases(
            ode_args,
            cruise_mach=cruise_mach,
        )

    traj = FlexibleTraj(
        Phases=phases,
        traj_initial_state_input=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
        traj_final_state_output=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
        promote_all_auto_ivc=True,
    )

    model = om.Group()

    model.add_subsystem(
        'top_of_descent_mass',
        om.ExecComp(
            'mass_initial = empty_mass + payload_mass + 0 + descent_fuel_estimate',
            mass_initial={'units': 'lbm'},
            empty_mass={'units': 'lbm'},
            payload_mass={'units': 'lbm'},
            # reserve_fuel = {'units':'lbm'},
            descent_fuel_estimate={'units': 'lbm', 'val': 0},
        ),
        promotes_inputs=[
            ('empty_mass', Aircraft.Design.EMPTY_MASS),
            ('payload_mass', Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS),
            # ('reserve_fuel', Aircraft.Design.EMPTY_MASS),
            ('descent_fuel_estimate', 'descent_fuel'),
        ],
        promotes_outputs=['mass_initial'])

    model.add_subsystem(
        'traj', traj,
        promotes_inputs=['altitude_initial', 'mass_initial', 'aircraft:*'],
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
        promotes_outputs=[('actual_fuel_burn', 'descent_fuel')])

    model.add_subsystem(
        "start_of_descent_mass",
        om.ExecComp(
            "start_of_descent_mass = mass_initial",
            start_of_descent_mass={"units": "lbm"},
            mass_initial={"units": "lbm"},
        ),
        promotes_inputs=["mass_initial"],
        promotes_outputs=["start_of_descent_mass"],
    )

    model.add_subsystem(
        "fuel_obj",
        om.ExecComp(
            "reg_objective = overall_fuel/10000",
            reg_objective={"val": 0.0, "units": "unitless"},
            overall_fuel={"units": "lbm"},
        ),
        promotes_inputs=[
            ("overall_fuel", 'descent_fuel'),
        ],
        promotes_outputs=["reg_objective"],
    )

    model.add_objective("reg_objective", ref=1e4)

    model.linear_solver = om.DirectSolver(assemble_jac=True)
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=0, iprint=0)

    input_aliases = []
    # if isinstance(initial_mass, str):
    #     input_aliases.append(('mass_initial',initial_mass))
    if isinstance(initial_mass, (int, float)):
        model.set_input_defaults('mass_initial', initial_mass)

    if isinstance(cruise_alt, str):
        input_aliases.append(('altitude_initial', cruise_alt))
    elif isinstance(cruise_alt, (int, float)):
        model.set_input_defaults('altitude_initial', cruise_alt)

    aviary_inputs = main_prob.aviary_inputs

    model.add_subsystem(
        'input_sink',
        VariablesIn(aviary_options=aviary_inputs,
                    meta_data=BaseMetaData),
        promotes_inputs=['*'],
        promotes_outputs=['*'])

    promote_aircraft_and_mission_vars(model)
    set_aviary_initial_values(model, aviary_inputs, BaseMetaData)

    subprob = om.Problem(model=model)
    subcomp = om.SubmodelComp(
        problem=subprob,
        inputs=[
            # Aircraft.Design.EMPTY_MASS,
            # Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
            'aircraft:*'
        ],
        outputs=['distance_final', 'descent_fuel'])

    main_prob.model.add_subsystem(
        'idle_descent_estimation',
        subcomp,
        promotes_inputs=['aircraft:*']+input_aliases,
        promotes_outputs=[
            ('distance_final', 'descent_range'),
            'descent_fuel',
        ],

    )
    set_aviary_initial_values(main_prob.model, aviary_inputs, BaseMetaData)
