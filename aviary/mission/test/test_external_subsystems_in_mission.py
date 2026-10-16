"""Test for some features when using an external subsystem in the mission."""

import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info as energy_phase_info
from aviary.models.missions.two_dof_default import phase_info as two_dof_phase_info
from aviary.subsystems.aerodynamics.aerodynamics_builder import (
    AERO_2DOF_INPUTS,
    AERO_LS_2DOF_INPUTS,
    AERO_CLEAN_2DOF_INPUTS,
)
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import CruiseAero, LowSpeedAero
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.variable_meta_data import CoreMetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


energy_phase_info = deepcopy(energy_phase_info)

energy_phase_info['pre_mission']['include_takeoff'] = False
energy_phase_info['post_mission']['include_landing'] = False
energy_phase_info.pop('climb')
energy_phase_info.pop('descent')


@use_tempdirs
class TestExternalSubsystems(unittest.TestCase):
    def test_mission_solver_energy(self):
        local_phase_info = deepcopy(energy_phase_info)

        prob = AviaryProblem()

        prob.load_inputs(
            'validation_cases/validation_data/test_models/high_wing_single_aisle.csv',
            local_phase_info,
        )
        prob.load_external_subsystems([SolverBuilder(name='solve_me')])
        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs_all.solver_sub,
                'solve_me',
            )
        )

    def test_no_mission_solver_energy(self):
        local_phase_info = deepcopy(energy_phase_info)

        prob = AviaryProblem()

        prob.load_inputs(
            'validation_cases/validation_data/test_models/high_wing_single_aisle.csv',
            local_phase_info,
        )

        prob.load_external_subsystems([NoSolverBuilder(name='do_not_solve_me')])

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs_all,
                'do_not_solve_me',
            )
        )

    def test_mission_solver_2DOF(self):
        local_phase_info = deepcopy(two_dof_phase_info)

        prob = AviaryProblem()

        prob.load_inputs(
            'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.csv',
            local_phase_info,
        )
        prob.load_external_subsystems([SolverBuilder(name='solve_me')])

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs_all.solver_sub,
                'solve_me',
            )
        )

    def test_no_mission_solver_2DOF(self):
        local_phase_info = deepcopy(two_dof_phase_info)

        prob = AviaryProblem()

        prob.load_inputs(
            'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.csv',
            local_phase_info,
        )
        prob.load_external_subsystems([NoSolverBuilder(name='do_not_solve_me')])

        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.setup()

        prob.run_model()

        self.assertTrue(
            hasattr(
                prob.model.traj.phases.cruise.rhs_all,
                'do_not_solve_me',
            )
        )

    def test_2dof_external_aero(self):
        # Test using a builder that mimics the actual 2dof aero builder so that any balances in
        # the ODE will be able to solve.

        class SelectiveAeroBuilder(SubsystemBuilder):
            def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
                if subsystem_options.get('enable_low_speed') is True:
                    aero_opts = subsystem_options.copy()
                    aero_opts.pop('enable_low_speed')
                    obj = LowSpeedAero(num_nodes=num_nodes, **aero_opts)
                elif subsystem_options.get('enable_cruise') is True:
                    obj = CruiseAero(num_nodes=num_nodes)
                else:
                    obj = None
                return obj

            def needs_mission_solver(self, aviary_inputs, user_options, subsystem_options):
                return True

            def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
                if subsystem_options.get('enable_low_speed') is True:
                    promotes = [
                        '*',
                        ('airport_alt', Mission.Takeoff.AIRPORT_ALTITUDE),
                        ('CL_max_flaps', Mission.Takeoff.LIFT_COEFFICIENT_MAX),
                        (
                            'dCL_flaps_model',
                            Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
                        ),
                        (
                            'dCD_flaps_model',
                            Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
                        ),
                        ('flap_defl', Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF),
                    ]
                else:
                    promotes = ['*']

                return promotes

            def mission_outputs(
                self, aviary_inputs=None, user_options=None, subsystem_options=None
            ):
                if subsystem_options.get('enable_low_speed') is True:
                    promotes = [
                        Dynamic.Vehicle.DRAG,
                        Dynamic.Vehicle.LIFT,
                        Dynamic.Vehicle.DRAG_COEFFICIENT,
                        Dynamic.Vehicle.LIFT_COEFFICIENT,
                        'flap_factor',
                        'gear_factor',
                    ]
                elif subsystem_options.get('enable_cruise') is True:
                    promotes = [
                        Dynamic.Vehicle.DRAG,
                        Dynamic.Vehicle.LIFT,
                        Dynamic.Vehicle.DRAG_COEFFICIENT,
                        Dynamic.Vehicle.LIFT_COEFFICIENT,
                    ]
                else:
                    promotes = ['*']
                return promotes

            def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None):
                params = {}
                if subsystem_options.get('enable_low_speed') is True:
                    all_vars = set()
                    all_vars = set(AERO_2DOF_INPUTS + AERO_LS_2DOF_INPUTS)

                elif subsystem_options.get('enable_cruise') is True:
                    all_vars = set(AERO_2DOF_INPUTS + AERO_CLEAN_2DOF_INPUTS)

                else:
                    return params

                for var in all_vars:
                    meta = CoreMetaData[var]

                    val = meta['default_value']
                    if val is None:
                        val = 0.0  # _unspecified
                    units = meta['units']

                    if var in aviary_inputs:
                        try:
                            val = aviary_inputs.get_val(var, units)
                        except TypeError:
                            val = aviary_inputs.get_val(var)

                    params[var] = {'val': val, 'units': units, 'static_target': True}

                return params

            def get_pre_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
                kwargs = {}

                if mission_info is not None:
                    phases = []

                    for name, phase in mission_info.items():
                        subsystem_options = phase.get('subsystem_options')

                        if subsystem_options is None:
                            continue

                        aero_options = subsystem_options.get('aerodynamics')

                        if aero_options is None:
                            continue

                        if aero_options.get('enable_low_speed') or aero_options.get(
                            'enable_cruise'
                        ):
                            phases.append(name)

                    if len(phases) == 0:
                        return {}

                    kwargs = {'phases': phases}

                return {
                    'interference_independent_of_shielded_area': {
                        'mission_name': ['interference_independent_of_shielded_area'],
                        # "post_mission_name": ['interference_independent_of_shielded_area'],
                        'units': 'unitless',
                        **kwargs,
                    },
                    'drag_loss_due_to_shielded_wing_area': {
                        'mission_name': ['drag_loss_due_to_shielded_wing_area'],
                        # "post_mission_name": ['drag_loss_due_to_shielded_wing_area'],
                        'units': 'unitless',
                        **kwargs,
                    },
                }

        phase_info = deepcopy(two_dof_phase_info)

        # Disable internal aero
        # Also, include any settings in our new aero.
        for phase in phase_info:
            if phase in ['groundroll', 'rotation', 'ascent', 'accel']:
                phase_info[phase]['subsystem_options']['aerodynamics'] = {
                    'method': 'external',
                }
                phase_info[phase]['subsystem_options']['new_aero'] = {
                    'enable_low_speed': True,
                }
            elif phase in ['climb1', 'cruise']:
                phase_info[phase]['subsystem_options']['aerodynamics'] = {
                    'method': 'external',
                }
                phase_info[phase]['subsystem_options']['new_aero'] = {
                    'enable_cruise': True,
                }


        prob = AviaryProblem(verbosity=0)

        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_GwGm.csv', phase_info
        )
        subs = [SelectiveAeroBuilder(name='new_aero')]
        prob.load_external_subsystems(subs)
        prob.check_and_preprocess_inputs()

        prob.build_model()

        prob.add_driver('SLSQP')

        prob.add_design_variables()
        prob.add_objective()

        prob.setup()

        # If everything is working correctly, we can setup and run without any errors.
        prob.run_model()


class ExternNoSolve(om.ExplicitComponent):
    """This component should not have a solver above it."""

    def setup(self):
        self.add_input(Aircraft.Wing.AREA, 1.0, units='ft**2')
        self.add_output('stuff', 1.0, units='ft**2')

    def compute(self, inputs, outputs):
        pass


class NoSolverBuilder(SubsystemBuilder):
    """Mission only. No solver."""

    def needs_mission_solver(self, aviary_inputs, user_options, subsystem_options):
        return False

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return ExternNoSolve()


class SolverBuilder(SubsystemBuilder):
    """Mission only. Solver."""

    def needs_mission_solver(self, aviary_inputs, user_options, subsystem_options):
        return True

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return ExternNoSolve()


if __name__ == '__main__':
    unittest.main()
    # test = TestExternalSubsystems()
    # test.test_no_mission_solver_2DOF()
