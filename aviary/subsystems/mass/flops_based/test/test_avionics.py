import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.avionics import TransportAvionicsMass
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission

_fuse_count = 'TBD:option:fuselage_count'
_fuse_total_length = 'aircraft:fuselage:dimensions:total_length'
_fuse_max_depth = 'aircraft:fuselage:dimensions:max_depth'
_max_mach = 'aircraft:design:dimensions:max_mach'
# _carrier_based = 'aircraft:landing_gear:control:carrier_based'
_avionics_group_mass = 'TBD:avionics'


class TransportAvionicsMassTest(unittest.TestCase):
    '''
    Tests transport/GA avionics mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "avionics",
            TransportAvionicsMass(aviary_options=get_flops_inputs(
                case_name, preprocess=True)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Avionics.MASS_SCALER,
                        Aircraft.Fuselage.PLANFORM_AREA,
                        Mission.Design.RANGE],
            output_keys=Aircraft.Avionics.MASS,
            aviary_option_keys=[Aircraft.CrewPayload.NUM_FLIGHT_CREW],
            tol=2.0e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
