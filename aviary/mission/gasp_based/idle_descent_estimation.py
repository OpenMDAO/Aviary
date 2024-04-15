import warnings

import openmdao.api as om

from aviary.interface.default_phase_info.two_dof_fiti import create_2dof_based_descent_phases
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.variable_info.variables import Aircraft, Mission, Dynamic
from aviary.utils.functions import set_aviary_initial_values, promote_aircraft_and_mission_vars
from aviary.variable_info.variables_in import VariablesIn
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData


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

    from aviary.utils.functions import create_printcomp
    dummy_comp1 = create_printcomp(
        all_inputs=[
            Aircraft.Design.OPERATING_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
            'descent_fuel',
            'mass_initial',
        ],
        input_units={
            'descent_fuel': 'lbm',
            'mass_initial': 'lbm',
        })
    model.add_subsystem(
        "dummy_comp_pre",
        dummy_comp1(),
        promotes_inputs=["*"],
    )

    model.add_subsystem(
        'top_of_descent_mass',
        # om.ExecComp(
        #     'mass_initial = operating_mass + payload_mass + 0 + descent_fuel_estimate',
        #     mass_initial={'units': 'lbm'},
        #     operating_mass={'units': 'lbm'},
        #     payload_mass={'units': 'lbm'},
        #     # reserve_fuel = {'units':'lbm'},
        #     descent_fuel_estimate={'units': 'lbm', 'val': 0},
        # ),
        initial_mass_comp(),
        promotes_inputs=[
            'aircraft:*',
            # ('operating_mass', Aircraft.Design.OPERATING_MASS),
            # ('payload_mass', Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS),
            # ('reserve_fuel', Aircraft.Design.EMPTY_MASS),
            ('descent_fuel_estimate', 'descent_fuel'),
        ],
        promotes_outputs=['mass_initial']
    )

    dummy_comp2 = create_printcomp(
        all_inputs=[
            Aircraft.Design.OPERATING_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
            'descent_fuel',
            'mass_initial',
        ],
        input_units={
            'descent_fuel': 'lbm',
            'mass_initial': 'lbm',
        })
    model.add_subsystem(
        "dummy_comp_post",
        dummy_comp2(),
        promotes_inputs=["*"],
    )
    model.set_input_defaults(
        Aircraft.Design.OPERATING_MASS, val=0, units='lbm')

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

    model.set_input_defaults(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, 0)

    # aviary_inputs = main_prob.aviary_inputs

    # model.add_subsystem(
    #     'input_sink',
    #     VariablesIn(aviary_options=aviary_inputs,
    #                 meta_data=BaseMetaData),
    #     promotes_inputs=['*'],
    #     promotes_outputs=['*'])

    promote_aircraft_and_mission_vars(model)
    # set_aviary_initial_values(model, aviary_inputs, BaseMetaData)

    subprob = om.Problem(model=model)
    subcomp = om.SubmodelComp(
        problem=subprob,
        inputs=[
            # 'aircraft:*'
            '*'
        ],
        outputs=['distance_final', 'descent_fuel', 'mass_initial'],
        do_coloring=False
    )

    main_prob.model.add_subsystem(
        'idle_descent_estimation',
        subcomp,
        promotes_inputs=[
            # 'aircraft:*'
            '*'
        ] + input_aliases,
        promotes_outputs=[
            ('distance_final', 'descent_range'),
            'descent_fuel',
        ],

    )
    subprob.setup()
    om.n2(subprob, 'subprob_n2.html', show_browser=False)
    # exit()
    # set_aviary_initial_values(main_prob.model, aviary_inputs, BaseMetaData)


input_list = [
    # 'traj:EAS',
    # 'traj:alt_trigger',
    # 'traj:altitude',
    # 'traj:distance',
    # 'traj:distance_initial',
    # 'traj:mach',
    # 'traj:mass',
    # 'traj:mission:design:gross_mass',
    # 'traj:mission:design:lift_coefficient_max_flaps_up',
    # 'traj:mission:design:reserve_fuel',
    # 'traj:mission:landing:airport_altitude',
    # 'traj:mission:landing:drag_coefficient_flap_increment',
    # 'traj:mission:landing:lift_coefficient_flap_increment',
    # 'traj:mission:landing:lift_coefficient_max',
    # 'traj:mission:summary:fuel_flow_scaler',
    # 'traj:mission:summary:gross_mass',
    # 'traj:mission:takeoff:airport_altitude',
    # 'traj:mission:takeoff:drag_coefficient_flap_increment',
    # 'traj:mission:takeoff:lift_coefficient_flap_increment',
    # 'traj:mission:takeoff:lift_coefficient_max',
    # 'traj:operating_mass',
    # 'traj:speed_trigger',
    # 'traj:t_curr',
    # 'traj:throttle',
    # 'traj:throttle_max',
    # 'traj:velocity_rate',
    Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
    Aircraft.Design.CG_DELTA,
    Aircraft.Design.DRAG_COEFFICIENT_INCREMENT,
    Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE,
    Aircraft.Design.OPERATING_MASS,
    Aircraft.Design.STATIC_MARGIN,
    Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT,
    Aircraft.Engine.SCALE_FACTOR,
    Aircraft.Fuselage.AVG_DIAMETER,
    Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT,
    Aircraft.Fuselage.FORM_FACTOR,
    Aircraft.Fuselage.LENGTH,
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.HorizontalTail.AREA,
    Aircraft.HorizontalTail.AVERAGE_CHORD,
    Aircraft.HorizontalTail.FORM_FACTOR,
    Aircraft.HorizontalTail.MOMENT_RATIO,
    Aircraft.HorizontalTail.SPAN,
    Aircraft.HorizontalTail.SWEEP,
    Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
    Aircraft.Nacelle.AVG_LENGTH,
    Aircraft.Nacelle.FORM_FACTOR,
    Aircraft.Nacelle.SURFACE_AREA,
    Aircraft.Strut.AREA_RATIO,
    Aircraft.Strut.CHORD,
    Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR,
    Aircraft.VerticalTail.AREA,
    Aircraft.VerticalTail.AVERAGE_CHORD,
    Aircraft.VerticalTail.FORM_FACTOR,
    Aircraft.VerticalTail.SPAN,
    Aircraft.Wing.AREA,
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.AVERAGE_CHORD,
    Aircraft.Wing.CENTER_DISTANCE,
    Aircraft.Wing.FLAP_CHORD_RATIO,
    Aircraft.Wing.FLAP_DEFLECTION_LANDING,
    Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF,
    Aircraft.Wing.FORM_FACTOR,
    Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR,
    Aircraft.Wing.HEIGHT,
    Aircraft.Wing.INCIDENCE,
    Aircraft.Wing.MAX_THICKNESS_LOCATION,
    Aircraft.Wing.MIN_PRESSURE_LOCATION,
    Aircraft.Wing.MOUNTING_TYPE,
    Aircraft.Wing.SPAN,
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
    Aircraft.Wing.ZERO_LIFT_ANGLE,
    'altitude_initial',
]


class initial_mass_comp(om.ExplicitComponent):

    def setup(self):
        self.add_input(Aircraft.Design.OPERATING_MASS, units='lbm')
        self.add_input(Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, units='lbm')
        self.add_input('descent_fuel_estimate', units='lbm')
        self.add_output('mass_initial', units='lbm')

    def compute(self, inputs, outputs):
        oem = inputs[Aircraft.Design.OPERATING_MASS]
        payload = inputs[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS]
        fuel = inputs['descent_fuel_estimate']
        print('oem', 'payload', 'fuel')
        print(oem, payload, fuel)
        outputs['mass_initial'] = oem + payload + fuel
        print('mass_initial')
        print(outputs['mass_initial'])
