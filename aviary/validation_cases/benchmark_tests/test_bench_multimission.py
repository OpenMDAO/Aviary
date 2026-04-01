import copy as copy
import unittest

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

import aviary.api as av
from aviary.models.missions.energy_state_default import phase_info
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.variables import Aircraft, Mission, Settings


def multi_mission_example():
    # fly the same mission twice with two different passenger loads
    phase_info_mission1 = copy.deepcopy(phase_info)
    phase_info_mission2 = copy.deepcopy(phase_info)

    # get large single aisle values
    aviary_inputs_mission1 = get_flops_inputs('LargeSingleAisle2FLOPS')
    # loading a CSV - a new ' create vehicle function'

    aviary_inputs_mission2 = copy.deepcopy(aviary_inputs_mission1)

    # Due to current limitations in Aviary's ability to detect user input vs. default values,
    # the only way to set an aircraft to zero passengers is by setting
    # TOTAL_PAYLOAD_MASS = X CARGO_MASS + 0 PASSENGER_PAYLOAD_MASS.
    # This zeros out passenger and baggage mass.
    # Due to issue #610, setting PASSENGER_PAYLOAD_MASS = 0 will not work yet.
    # aviary_inputs_deadhead.set_val(Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, 4077, 'lbm')

    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, 1, 'unitless')
    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_ECONOMY_CLASS, 1, 'unitless')
    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_BUSINESS_CLASS, 0, 'unitless')
    aviary_inputs_mission2.set_val(Aircraft.CrewPayload.NUM_FIRST_CLASS, 0, 'unitless')

    # merged_meta_data = av.merge_hierarchies(meta_data1, meta_data2)
    # merge all metata data hierarchies here

    prob = av.AviaryProblem(problem_type=ProblemType.MULTI_MISSION)
    # set constraints in the background. Currently works with every objective type except Range.
    # can accept meta_data = merged_meta_data

    prob.add_aviary_group(
        'mission1', aircraft=aviary_inputs_mission1, phase_info=phase_info_mission1
    )
    # This method calls load_inputs(), check_and_preprocess_inputs(), and combines meta data.
    # This can only accept an AviaryValues, .csv are not accepted. You can pass problem_configurator
    # as an input.

    prob.add_aviary_group(
        'mission2', aircraft=aviary_inputs_mission2, phase_info=phase_info_mission2
    )
    # Load aircraft in second configuration for same mission

    prob.build_model()
    # combines four basic commands
    # prob.add_pre_mission_systems()
    # prob.add_phases()
    # prob.add_post_mission_systems()
    # prob.link_phases()

    prob.promote_inputs(
        ['mission1', 'mission2'],
        [
            (Aircraft.Design.GROSS_MASS, 'Aircraft1:GROSS_MASS'),
            (Aircraft.Design.RANGE, 'Aircraft1:RANGE'),
            (Aircraft.Wing.SWEEP, 'Aircraft1:SWEEP'),
        ],
    )
    # Links key design variables to ensure both aircraft are modelled the same:
    # Design gross mass sizes things like the landing gear
    # Design range sizing things like the avionics system

    prob.add_design_var_default(
        'Aircraft1:GROSS_MASS',
        lower=10.0,
        upper=900e3,
        units='lbm',
        default_val=100000,
    )
    prob.add_design_var_default(
        'Aircraft1:SWEEP',
        lower=23.0,
        upper=27.0,
        units='deg',
        default_val=25,
    )
    # This both adds the design variable AND sets the default value. This value can be over-written after-setup using set_val.

    prob.add_composite_objective(
        ('mission1', Mission.FUEL, 2),
        ('mission2', Mission.FUEL, 1),
        ref=1,
    )
    # Adds an objective where mission 1 is flown 2x more times than mission2
    # Alternative way that users could specify the same objective:
    # prob.add_composite_objective_adv(missions=['mission1', 'mission2'], mission_weights=[2,1], outputs=[Mission.FUEL],  ref=1)
    # TODO: MULTI_MISSION cannot handle RANGE objectives correctly at the moment.

    # optimizer and iteration limit are optional provided here
    # Note: IPOPT needs more iters than SNOPT.
    prob.add_driver('IPOPT', max_iter=70)
    prob.add_design_variables()

    prob.setup()
    # combines 2 basic commands:
    # prob.setup()
    # prob.set_initial_guesses()

    # set_val on OpenMDAO design variables can be placed here and will over-write all other defaults

    prob.set_design_range(('mission1', 'mission2'), range='Aircraft1:RANGE')
    # Determines the maximum design_range from both missions and sets that as the design range for both missions
    # to ensure that the avionics system is designed similarly for both aircraft

    # TODO: how to handle "aircraft that the user says are the same but are not the same i.e. wing design is different"

    prob.run_aviary_problem()

    return prob


@use_tempdirs
class MultiMissionTestcase(unittest.TestCase):
    """Test the different throttle allocation methods for models with multiple, unique EngineModels."""

    def setUp(self):
        om.clear_reports()
        _clear_problem_names()  # need to reset these to simulate separate runs

    @require_pyoptsparse(optimizer='IPOPT')
    def test_multimission(self):
        prob = multi_mission_example()

        objective = prob.get_val('composite_objective', units=None)
        objective_expected_value = 25517.15

        mission1_fuel = prob.get_val('mission1.mission:fuel', units='lbm')
        mission1_fuel_expected_value = 26877.7 # includes takeoff

        mission2_fuel = prob.get_val('mission2.mission:fuel', units='lbm')
        mission2_fuel_expected_value = 22795.9 # includes takeoff

        mission1_cargo = prob.get_val(
            'mission1.aircraft:crew_and_payload:total_payload_mass', units='lbm'
        )
        mission1_cargo_expected_value = 36477.0

        mission2_cargo = prob.get_val(
            'mission2.aircraft:crew_and_payload:total_payload_mass', units='lbm'
        )
        mission2_cargo_expected_value = 4277.0

        # alloc_climb = prob.get_val('traj.climb.parameter_vals:throttle_allocations', units='')

        self.assertTrue(prob.result.success)

        expected_values = {
            'objective': (objective_expected_value, objective),
            'mission1_fuel': (mission1_fuel_expected_value, mission1_fuel),
            'mission2_fuel': (mission2_fuel_expected_value, mission2_fuel),
            'mission1_cargo': (mission1_cargo_expected_value, mission1_cargo),
            'mission2_cargo': (mission2_cargo_expected_value, mission2_cargo),
        }

        for var_name, (expected, actual) in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(expected, actual, tolerance=1e-3)


if __name__ == '__main__':
    unittest.main()
