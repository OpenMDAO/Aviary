import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.hydraulics import (AltHydraulicsGroupMass,
                                                           TransportHydraulicsGroupMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission

_sys_press = 'aircraft:hydraulics:dimensions:sys_press'
_wing_ref_area = 'aircraft:wing:dimensions:area'
_fuse_planform_area = 'TBD:fuselage_planform_area'
_horiz_tail_wetted_area = 'aircraft:horizontal_tail:dimensions:wetted_area'
_horiz_tail_thick_chord = 'aircraft:horizontal_tail:dimensions:thickness_to_chord_ratio'
_vert_tail_area = 'aircraft:vertical_tails:dimensions:area'
_wing_engine_count_factor = 'aircraft:propulsion:control:wing_engine_count_factor'
_fuse_engine_count_factor = 'aircraft:propulsion:control:fuselage_engine_count_factor'
_var_sweep_mass_penalty = 'aircraft:wing:mass:var_sweep_mass_penalty'
_max_mach = 'aircraft:design:dimensions:max_mach'
_hydraulics_group_mass = 'TBD:hydraulics_mass'


class TransportHydraulicsGroupMassTest(unittest.TestCase):
    '''
    Tests transport/GA hydraulics mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'hydraulics',
            TransportHydraulicsGroupMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Fuselage.PLANFORM_AREA,
                        Aircraft.Hydraulics.SYSTEM_PRESSURE,
                        Aircraft.Hydraulics.MASS_SCALER,
                        Aircraft.Wing.AREA,
                        Aircraft.Wing.VAR_SWEEP_MASS_PENALTY],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.TRANSPORT,
            tol=4.0e-4)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltHydraulicsGroupMassTest(unittest.TestCase):
    '''
    Tests alternate hydraulics mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'hydraulics',
            AltHydraulicsGroupMass(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Wing.AREA,
                        Aircraft.HorizontalTail.WETTED_AREA,
                        Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                        Aircraft.VerticalTail.AREA,
                        Aircraft.Hydraulics.MASS_SCALER],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
