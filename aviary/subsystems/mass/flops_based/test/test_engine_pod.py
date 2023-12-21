import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.engine_pod import EnginePodMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft


class EnginePodMassTest(unittest.TestCase):
    '''
    Tests the engine pod mass needed for the detailed wing calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    # Only cases that use detailed wing weight.
    @parameterized.expand(get_flops_case_names(omit=['LargeSingleAisle2FLOPS', 'LargeSingleAisle2FLOPSalt']),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'engine_pod',
            EnginePodMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        # Tol not that tight, but it is unclear where the pod mass values in files come from,
        # since they aren't printed in the FLOPS output.
        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Electrical.MASS,
                        Aircraft.Fuel.FUEL_SYSTEM_MASS,
                        Aircraft.Hydraulics.MASS,
                        Aircraft.Instruments.MASS,
                        Aircraft.Nacelle.MASS,
                        Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                        Aircraft.Engine.MASS,
                        Aircraft.Propulsion.TOTAL_STARTER_MASS,
                        Aircraft.Engine.THRUST_REVERSERS_MASS,
                        Aircraft.Engine.SCALED_SLS_THRUST,
                        Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST],
            output_keys=Aircraft.Engine.POD_MASS,
            tol=3e-3)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
