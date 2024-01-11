import unittest
from parameterized import parameterized

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from aviary.subsystems.premission import CorePreMission
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.aviary_values import get_items
from aviary.validation_cases.validation_tests import (
    flops_validation_test, get_flops_inputs, get_flops_outputs, get_flops_case_names, print_case
)
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.variables_in import VariablesIn
from aviary.utils.functions import set_aviary_initial_values
from aviary.interface.default_phase_info.height_energy import default_premission_subsystems
from aviary.utils.preprocessors import preprocess_crewpayload


class PreMissionGroupTest(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)
        flops_inputs.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
                             flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES))

        preprocess_crewpayload(flops_inputs)

        prob = self.prob

        prob.model.add_subsystem(
            "pre_mission",
            CorePreMission(aviary_options=flops_inputs,
                           subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.add_subsystem(
            'input_sink',
            VariablesIn(aviary_options=flops_inputs),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

        set_aviary_initial_values(prob.model, flops_inputs)
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(2)

        # Initial guess for gross weight.
        # We set it to an unconverged value to test convergence.
        prob.set_val(Mission.Design.GROSS_MASS, val=1000.0)

        # Set inital values for all variables.
        for (key, (val, units)) in get_items(flops_inputs):
            try:
                prob.set_val(key, val, units)

            except KeyError:
                # This is an option, not a variable
                continue

        if case_name in ['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPSdw']:
            # We set these so that their derivatives are defined.
            # The ref values are not set in our test models.
            prob[Aircraft.Wing.ASPECT_RATIO_REF] = prob[Aircraft.Wing.ASPECT_RATIO]
            prob[Aircraft.Wing.THICKNESS_TO_CHORD_REF] = prob[Aircraft.Wing.THICKNESS_TO_CHORD]

        prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm')
        prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, units='lbm')

        flops_validation_test(
            prob,
            case_name,
            input_keys=[],
            output_keys=[Aircraft.Design.STRUCTURE_MASS,
                         Aircraft.Propulsion.MASS,
                         Aircraft.Design.SYSTEMS_EQUIP_MASS,
                         Aircraft.Design.EMPTY_MASS,
                         Aircraft.Design.OPERATING_MASS,
                         Aircraft.Design.ZERO_FUEL_MASS,
                         Mission.Design.FUEL_MASS],
            step=1.01e-40,
            atol=1e-8,
            rtol=1e-10)

    def test_diff_configuration_mass(self):
        # This standalone test provides coverage for some features unique to this
        # model.
        from aviary.models.large_single_aisle_2.large_single_aisle_2_FLOPS_data import LargeSingleAisle2FLOPS

        prob = om.Problem()

        flops_inputs: AviaryValues = LargeSingleAisle2FLOPS['inputs']
        flops_outputs: AviaryValues = LargeSingleAisle2FLOPS['outputs']

        prob.model.add_subsystem(
            "pre_mission",
            CorePreMission(aviary_options=flops_inputs,
                           subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.add_subsystem(
            'input_sink',
            VariablesIn(aviary_options=flops_inputs),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

        set_aviary_initial_values(prob.model, flops_inputs)
        prob.setup(check=False)

        # Set inital values for all variables.
        for (key, (val, units)) in get_items(flops_inputs):
            try:
                prob.set_val(key, val, units)

            except KeyError:
                # This is an option, not a variable
                continue

        prob.run_model()

        flops_validation_test(
            prob,
            "LargeSingleAisle2FLOPS",
            input_keys=[],
            output_keys=[Mission.Design.FUEL_MASS],
            atol=1e-4,
            rtol=1e-4,
            check_partials=False,
            flops_inputs=flops_inputs,
            flops_outputs=flops_outputs)


if __name__ == "__main__":
    unittest.main()
