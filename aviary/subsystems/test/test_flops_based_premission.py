import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    get_flops_outputs,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings


class PreMissionGroupTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)
        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES),
        )
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

        prob = self.prob

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        # prob.model.set_input_defaults(
        #     Aircraft.Engine.SCALE_FACTOR,
        #     flops_inputs.get_val(
        #         Aircraft.Engine.SCALE_FACTOR))

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(2)

        # Initial guess for gross weight.
        # We set it to an unconverged value to test convergence.
        prob.set_val(Mission.Design.GROSS_MASS, val=1000.0)

        set_aviary_initial_values(prob, flops_inputs)

        if case_name in ['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPSdw']:
            # We set these so that their derivatives are defined.
            # The ref values are not set in our test models.
            prob[Aircraft.Wing.ASPECT_RATIO_REF] = prob[Aircraft.Wing.ASPECT_RATIO]
            prob[Aircraft.Wing.THICKNESS_TO_CHORD_REF] = prob[Aircraft.Wing.THICKNESS_TO_CHORD]

        prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm'
        )
        prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, units='lbm'
        )

        flops_validation_test(
            prob,
            case_name,
            input_keys=[],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS,
                Aircraft.Design.OPERATING_MASS,
                Aircraft.Design.ZERO_FUEL_MASS,
                Mission.Design.FUEL_MASS,
            ],
            step=1.01e-40,
            atol=1e-8,
            rtol=1e-10,
        )

    def test_diff_configuration_mass(self):
        # This standalone test provides coverage for some features unique to this
        # model.

        prob = om.Problem()

        flops_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')
        flops_outputs = get_flops_outputs('LargeSingleAisle2FLOPS')
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        prob.setup(check=False)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            prob,
            'LargeSingleAisle2FLOPS',
            input_keys=[],
            output_keys=[Mission.Design.FUEL_MASS],
            atol=1e-4,
            rtol=1e-4,
            check_partials=False,
            flops_inputs=flops_inputs,
            flops_outputs=flops_outputs,
        )


if __name__ == '__main__':
    unittest.main()
    # test = PreMissionGroupTest()
    # test.setUp()
    # test.test_diff_configuration_mass()
