import aviary.api as av
import warnings
import unittest
import dymos as dm
import openmdao.api as om
from aviary.api import Dynamic, Mission
from aviary.core.aviary_group import AviaryGroup
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import inputs
from aviary.utils.preprocessors import preprocess_options


class TestTakeoffToEngineFailureTest(unittest.TestCase):
    """Test takeoff phase builder."""

    def test_case1(self):
        aviary_options = inputs.deepcopy()

        # This builder can be used for both takeoff and landing phases
        aero_builder = av.CoreAerodynamicsBuilder(
            name='low_speed_aero', code_origin=av.LegacyCode.FLOPS
        )

        # fmt: off
        takeoff_subsystem_options = {
            'low_speed_aero': {
                'method': 'low_speed',
                'ground_altitude': 0.0,  # units='m'
                'angles_of_attack': [
                    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                ],  # units='deg'
                'lift_coefficients': [
                    0.5178, 0.6, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,
                    1.35, 1.5, 1.6, 1.7, 1.8, 1.85, 1.9, 1.95,
                ],
                'drag_coefficients': [
                    0.0674, 0.065, 0.065, 0.07, 0.072, 0.076, 0.084, 0.09,
                    0.10, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.20,
                ],
                'lift_coefficient_factor': 1.0,
                'drag_coefficient_factor': 1.0,
            }
        }
        # fmt: off

        # when using spoilers, add a few more options
        takeoff_spoiler_subsystem_options = {
            'low_speed_aero': {
                **takeoff_subsystem_options['low_speed_aero'],
                'use_spoilers': True,
                'spoiler_drag_coefficient': inputs.get_val(Mission.Takeoff.SPOILER_DRAG_COEFFICIENT),
                'spoiler_lift_coefficient': inputs.get_val(Mission.Takeoff.SPOILER_LIFT_COEFFICIENT),
            }
        }

        # We also need propulsion analysis for takeoff and landing. No additional configuration
        # is needed for this builder
        engines = [av.build_engine_deck(aviary_options)]
        preprocess_options(aviary_options, engine_models=engines)
        prop_builder = av.CorePropulsionBuilder(engine_models=engines)

        # BRAKE RELEASE TO DECISION SPEED
        takeoff_brake_release_user_options = av.AviaryValues()

        takeoff_brake_release_user_options.set_val('max_duration', val=60.0, units='s')
        takeoff_brake_release_user_options.set_val('time_duration_ref', val=60.0, units='s')
        takeoff_brake_release_user_options.set_val('distance_max', val=7500.0, units='ft')
        takeoff_brake_release_user_options.set_val('max_velocity', val=167.85, units='kn')
        takeoff_brake_release_user_options.set_val('terminal_condition', val='V1')

        tobl_nl_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6)
        tobl_nl_solver.linesearch = om.BoundsEnforceLS()

        takeoff_brake_release_user_options.set_val('nonlinear_solver', val=tobl_nl_solver)
        takeoff_brake_release_user_options.set_val('linear_solver', val=om.DirectSolver())

        takeoff_v1_to_vr_initial_guesses = av.AviaryValues()

        takeoff_v1_to_vr_initial_guesses.set_val('time', [0.0, 30.0], 's')
        takeoff_v1_to_vr_initial_guesses.set_val('distance', [0.0, 4100.0], 'ft')
        takeoff_v1_to_vr_initial_guesses.set_val('velocity', [0.01, 150.0], 'kn')

        gross_mass_units = 'lbm'
        gross_mass = inputs.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        takeoff_v1_to_vr_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        takeoff_v1_to_vr_initial_guesses.set_val('throttle', 1.0)
        takeoff_v1_to_vr_initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        takeoff_brake_release_to_decision_speed_builder = av.DetailedTakeoffPhaseBuilder(
            'takeoff_brake_release_to_decision_speed',
            core_subsystems=[aero_builder, prop_builder],
            subsystem_options=takeoff_subsystem_options,
            user_options=takeoff_brake_release_user_options,
            initial_guesses=takeoff_v1_to_vr_initial_guesses,
        )

        # DECISION SPEED TO ROTATION

        takeoff_v1_to_vr_user_options = av.AviaryValues()

        takeoff_v1_to_vr_user_options.set_val('max_duration', val=90.0, units='s')
        takeoff_v1_to_vr_user_options.set_val('time_duration_ref', val=60.0, units='s')
        takeoff_v1_to_vr_user_options.set_val('distance_max', val=15000.0, units='ft')
        takeoff_v1_to_vr_user_options.set_val('max_velocity', val=167.85, units='kn')
        takeoff_v1_to_vr_user_options.set_val('terminal_condition', val='VR')

        nl_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6)
        nl_solver.linesearch = om.BoundsEnforceLS()

        takeoff_v1_to_vr_user_options.set_val('nonlinear_solver', val=nl_solver)
        takeoff_v1_to_vr_user_options.set_val('linear_solver', val=om.DirectSolver())

        takeoff_v1_to_vr_initial_guesses = av.AviaryValues()

        takeoff_v1_to_vr_initial_guesses.set_val('time', [30.0, 1.0], 's')
        takeoff_v1_to_vr_initial_guesses.set_val('distance', [4100.0, 4800.0], 'ft')
        takeoff_v1_to_vr_initial_guesses.set_val('velocity', [70., 150.0], 'kn')

        gross_mass_units = 'lbm'
        gross_mass = inputs.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        takeoff_v1_to_vr_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        takeoff_v1_to_vr_initial_guesses.set_val('throttle', 1.0)
        takeoff_v1_to_vr_initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        takeoff_decision_speed_to_rotate_builder = av.DetailedTakeoffPhaseBuilder(
            'takeoff_decision_speed_to_rotate',
            core_subsystems=[aero_builder, prop_builder],
            subsystem_options=takeoff_subsystem_options,
            user_options=takeoff_v1_to_vr_user_options,
            initial_guesses=takeoff_v1_to_vr_initial_guesses,
        )

        # ROTATION TO LIFTOFF

        vr_to_liftoff_user_options = av.AviaryValues()

        vr_to_liftoff_user_options.set_val('max_duration', val=90.0, units='s')
        vr_to_liftoff_user_options.set_val('time_duration_ref', val=60.0, units='s')
        vr_to_liftoff_user_options.set_val('distance_max', val=15000.0, units='ft')
        vr_to_liftoff_user_options.set_val('max_velocity', val=167.85, units='kn')
        vr_to_liftoff_user_options.set_val('pitch_control', val='alpha_rate_fixed', units='unitless')
        vr_to_liftoff_user_options.set_val('terminal_condition', val='LIFTOFF')

        nl_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6)
        nl_solver.linesearch = om.BoundsEnforceLS()

        vr_to_liftoff_user_options.set_val('nonlinear_solver', val=nl_solver)
        vr_to_liftoff_user_options.set_val('linear_solver', val=om.DirectSolver())

        vr_to_liftoff_initial_guesses = av.AviaryValues()

        vr_to_liftoff_initial_guesses.set_val('time', [31.0, 5.0], 's')
        vr_to_liftoff_initial_guesses.set_val('distance', [4800.0, 5500.0], 'ft')
        vr_to_liftoff_initial_guesses.set_val('velocity', [100., 120.0], 'kn')
        vr_to_liftoff_initial_guesses.set_val('angle_of_attack', [0.0, 5.0], 'deg')

        gross_mass_units = 'lbm'
        gross_mass = inputs.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        vr_to_liftoff_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        vr_to_liftoff_initial_guesses.set_val('throttle', 1.0)
        vr_to_liftoff_initial_guesses.set_val('angle_of_attack_rate', 2.0, units='deg/s')

        vr_to_liftoff_builder = av.DetailedTakeoffPhaseBuilder(
            'takeoff_rotate_to_liftoff',
            core_subsystems=[aero_builder, prop_builder],
            subsystem_options=takeoff_subsystem_options,
            user_options=vr_to_liftoff_user_options,
            initial_guesses=vr_to_liftoff_initial_guesses,
        )

        # LIFTOFF TO CLIMB GRADIENT

        liftoff_to_climb_gradient_user_options = av.AviaryValues()

        liftoff_to_climb_gradient_user_options.set_val('max_duration', val=90.0, units='s')
        liftoff_to_climb_gradient_user_options.set_val('time_duration_ref', val=60.0, units='s')
        liftoff_to_climb_gradient_user_options.set_val('distance_max', val=15000.0, units='ft')
        liftoff_to_climb_gradient_user_options.set_val('max_velocity', val=167.85, units='kn')
        liftoff_to_climb_gradient_user_options.set_val('pitch_control', val='alpha_rate_fixed', units='unitless')
        liftoff_to_climb_gradient_user_options.set_val('climbing', val=True, units='unitless')
        liftoff_to_climb_gradient_user_options.set_val('terminal_condition', val='CLIMB_GRADIENT')

        nl_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6)
        nl_solver.linesearch = om.BoundsEnforceLS()

        liftoff_to_climb_gradient_user_options.set_val('nonlinear_solver', val=nl_solver)
        liftoff_to_climb_gradient_user_options.set_val('linear_solver', val=om.DirectSolver())

        liftoff_to_climb_gradient_initial_guesses = av.AviaryValues()

        liftoff_to_climb_gradient_initial_guesses.set_val('time', [35.0, 1.0], 's')
        liftoff_to_climb_gradient_initial_guesses.set_val('distance', [5000.0, 6000.0], 'ft')
        liftoff_to_climb_gradient_initial_guesses.set_val('velocity', [120., 100.0], 'kn')
        liftoff_to_climb_gradient_initial_guesses.set_val('angle_of_attack', [5.0, 10.0], 'deg')
        liftoff_to_climb_gradient_initial_guesses.set_val('flight_path_angle', [0.0, 2.0], 'deg')

        gross_mass_units = 'lbm'
        gross_mass = inputs.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        liftoff_to_climb_gradient_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        liftoff_to_climb_gradient_initial_guesses.set_val('throttle', 1.0)
        liftoff_to_climb_gradient_initial_guesses.set_val('angle_of_attack_rate', 2.0, units='deg/s')

        liftoff_to_climb_gradient_builder = av.DetailedTakeoffPhaseBuilder(
            'takeoff_liftoff_to_climb_gradient',
            core_subsystems=[aero_builder, prop_builder],
            subsystem_options=takeoff_subsystem_options,
            user_options=liftoff_to_climb_gradient_user_options,
            initial_guesses=liftoff_to_climb_gradient_initial_guesses,
        )

        # CLIMB GRADIENT TO OBSTACLE

        climb_gradient_to_obstacle_user_options = av.AviaryValues()

        climb_gradient_to_obstacle_user_options.set_val('max_duration', val=90.0, units='s')
        climb_gradient_to_obstacle_user_options.set_val('time_duration_ref', val=60.0, units='s')
        climb_gradient_to_obstacle_user_options.set_val('distance_max', val=15000.0, units='ft')
        climb_gradient_to_obstacle_user_options.set_val('max_velocity', val=200., units='kn')
        climb_gradient_to_obstacle_user_options.set_val('pitch_control', val='gamma_fixed', units='unitless')
        climb_gradient_to_obstacle_user_options.set_val('climbing', val=True, units='unitless')
        climb_gradient_to_obstacle_user_options.set_val('terminal_condition', val='OBSTACLE')

        nl_solver = om.NewtonSolver(solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6, debug_print=False)
        nl_solver.linesearch = om.BoundsEnforceLS()

        climb_gradient_to_obstacle_user_options.set_val('nonlinear_solver', val=nl_solver)
        climb_gradient_to_obstacle_user_options.set_val('linear_solver', val=om.DirectSolver())

        climb_gradient_to_obstacle_initial_guesses = av.AviaryValues()

        climb_gradient_to_obstacle_initial_guesses.set_val('time', [35.0, 3.0], 's')
        climb_gradient_to_obstacle_initial_guesses.set_val('distance', [5500.0, 5800.0], 'ft')
        climb_gradient_to_obstacle_initial_guesses.set_val('velocity', [120., 120.0], 'kn')
        climb_gradient_to_obstacle_initial_guesses.set_val('flight_path_angle', [2.0, 2.0], 'deg')

        gross_mass_units = 'lbm'
        gross_mass = inputs.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        climb_gradient_to_obstacle_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        climb_gradient_to_obstacle_initial_guesses.set_val('throttle', 1.0)

        climb_gradient_to_obstacle_builder = av.DetailedTakeoffPhaseBuilder(
            'takeoff_climb_gradient_to_obstacle',
            core_subsystems=[aero_builder, prop_builder],
            subsystem_options=takeoff_subsystem_options,
            user_options=climb_gradient_to_obstacle_user_options,
            initial_guesses=climb_gradient_to_obstacle_initial_guesses,
        )

        # from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import (
        #     takeoff_decision_speed_builder,
        #     takeoff_engine_cutback_builder,
        #     takeoff_engine_cutback_to_mic_p1_builder,
        #     takeoff_liftoff_builder,
        #     takeoff_liftoff_user_options,
        #     takeoff_mic_p1_to_climb_builder,
        #     takeoff_mic_p2_builder,
        #     takeoff_mic_p2_to_engine_cutback_builder,
        #     takeoff_rotate_builder,
        # )
        from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
        from aviary.variable_info.functions import setup_model_options

        takeoff_trajectory_builder = av.DetailedTakeoffTrajectoryBuilder('detailed_takeoff')

        takeoff_trajectory_builder.set_brake_release_to_decision_speed(takeoff_brake_release_to_decision_speed_builder)

        takeoff_trajectory_builder.set_decision_speed_to_rotate(takeoff_decision_speed_to_rotate_builder)

        takeoff_trajectory_builder.set_rotate_to_liftoff(vr_to_liftoff_builder)

        takeoff_trajectory_builder.set_liftoff_to_climb_gradient(liftoff_to_climb_gradient_builder)

        takeoff_trajectory_builder.set_climb_gradient_to_obstacle(climb_gradient_to_obstacle_builder)

        # takeoff_trajectory_builder.set_liftoff_to_obstacle(takeoff_liftoff_builder)

        # takeoff_trajectory_builder.set_obstacle_to_mic_p2(takeoff_mic_p2_builder)

        # takeoff_trajectory_builder.set_mic_p2_to_engine_cutback(takeoff_mic_p2_to_engine_cutback_builder)

        # takeoff_trajectory_builder.set_engine_cutback(takeoff_engine_cutback_builder)

        # takeoff_trajectory_builder.set_engine_cutback_to_mic_p1(takeoff_engine_cutback_to_mic_p1_builder)

        # takeoff_trajectory_builder.set_mic_p1_to_climb(takeoff_mic_p1_to_climb_builder)

        takeoff = om.Problem() # model=AviaryGroup(aviary_options=aviary_options, aviary_metadata=BaseMetaData))

        # default subsystems
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

        # Upstream pre-mission analysis for aero
        takeoff.model.add_subsystem(
            'core_subsystems',
            av.CorePreMission(
                aviary_options=aviary_options,
                subsystems=default_premission_subsystems,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        # Instantiate the trajectory and add the phases
        traj = dm.Trajectory()
        takeoff.model.add_subsystem('traj', traj)

        takeoff_trajectory_builder.build_trajectory(
            aviary_options=aviary_options, model=takeoff.model, traj=traj
        )

        # distance_max, units = takeoff_liftoff_user_options.get_item('distance_max')
        # liftoff = takeoff_trajectory_builder.get_phase('takeoff_liftoff')

        # liftoff.add_objective(Dynamic.Mission.DISTANCE, loc='final', ref=distance_max, units=units)

        # Insert a constraint for a fake decision speed, until abort is added.
        # takeoff.model.add_constraint(
        #     'traj.takeoff_brake_release.states:velocity', equals=149.47, units='kn', ref=150.0, indices=[-1]
        # )

        # takeoff.model.add_constraint(
        #     'traj.takeoff_decision_speed.states:velocity',
        #     equals=155.36,
        #     units='kn',
        #     ref=159.0,
        #     indices=[-1],
        # )

        varnames = [
            av.Aircraft.Wing.AREA,
            av.Aircraft.Wing.ASPECT_RATIO,
            av.Aircraft.Wing.SPAN,
        ]
        av.set_aviary_input_defaults(takeoff.model, varnames, aviary_options)

        setup_model_options(takeoff, aviary_options)

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            takeoff.setup(check=False)

        av.set_aviary_initial_values(takeoff, aviary_options)

        takeoff_trajectory_builder.apply_initial_guesses(takeoff, 'traj')

        takeoff.final_setup()

        # takeoff.model.run_apply_nonlinear()

        # takeoff.model.list_vars(print_arrays=True)

        takeoff.run_model()

        takeoff.model.traj.phases.takeoff_climb_gradient_to_obstacle.ode_iter_group.segment_prop_group.ode_all.takeoff_eom.list_vars(print_arrays=True)

        # vars = takeoff.model.list_vars(print_arrays=True, units=True, prom_name=True, return_format='dict', out_stream=None)

        # vars = {meta['prom_name']: meta for meta in vars.values()}

        # systems = set([path.rsplit('.', 1)[0] for path in vars.keys()])

        # print('\n'.join(vars.keys()))

        # for sys in takeoff.model.system_iter(include_self=True, recurse=True):
        #     sys.list_vars(prom_name=True, units=True)


        # from textual.app import App, ComposeResult
        # from textual.widgets import DataTable, Collapsible


        # class TableApp(App):

        #     def __init__(self, systems):
        #         self._systems = systems

        #     def compose(self) -> ComposeResult:
        #         for sys, vars in self._systems.items():
        #             with Collapsible(title=sys.pathname):
        #                 yield DataTable()

        #     def on_mount(self) -> None:
        #         table = self.query_one(DataTable)
        #         table.add_columns('promoted name', 'units')
        #         print(table)
        #         # table.add_rows(ROWS[1:])

        # systems = {}

        # for sys in takeoff.model.system_iter(include_self=True, recurse=True):
        #     vars = sys.list_vars(prom_name=True, units=True, out_stream=None)
        #     systems[sys] = vars

        # app = TableApp(systems)
        # app.run()

        # takeoff.check_partials(compact_print=True)
        # takeoff.model.run_apply_nonlinear()
        # takeoff.model.list_vars(print_arrays=True)
        # om.n2(takeoff.model)


if __name__ == '__main__':
    unittest.main()
