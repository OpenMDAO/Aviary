import numpy as np
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import (assert_check_partials,
                                         assert_near_equal)
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
            EnginePodMass(aviary_options=get_flops_inputs(case_name, preprocess=True)),
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

    def test_case_multiengine(self):
        # test with multiple engine types
        prob = self.prob

        aviary_options = get_flops_inputs('LargeSingleAisle1FLOPS')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 2, 3]))
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 7)

        prob.model.add_subsystem(
            'engine_pod',
            EnginePodMass(aviary_options=aviary_options),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        prob.set_val(Aircraft.Nacelle.MASS, val=np.array([100, 30, 95]))
        prob.set_val(Aircraft.Engine.MASS, val=np.array([410, 45, 315]))
        prob.set_val(Aircraft.Engine.THRUST_REVERSERS_MASS, val=np.array([130, 0, 60]))
        prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST,
                     val=np.array([36000, 22500, 18000]))
        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=189000.0)

        prob.run_model()

        mass = prob.get_val(Aircraft.Engine.POD_MASS)
        expected_mass = np.array([525., 60., 362.14285714])

        assert_near_equal(mass, expected_mass, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
    # test = EnginePodMassTest()
    # test.setUp()
    # test.test_case_multiengine()
