import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.hydraulics import (
    AltHydraulicsGroupMass,
    TransportHydraulicsGroupMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.variables import Aircraft, Mission


class TransportHydraulicsGroupMassTest(unittest.TestCase):
    """Tests transport/GA hydraulics mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'hydraulics',
            TransportHydraulicsGroupMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Hydraulics.SYSTEM_PRESSURE,
                Aircraft.Hydraulics.MASS_SCALER,
                Aircraft.Wing.AREA,
                Aircraft.Wing.VAR_SWEEP_MASS_PENALTY,
            ],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.TRANSPORT,
            tol=4.0e-4,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class TransportHydraulicsGroupMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.hydraulics as hydraulics

        hydraulics.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.hydraulics as hydraulics

        hydraulics.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        inputs = get_flops_inputs('AdvancedSingleAisle', preprocess=True)

        options = {
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES
            ),
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: inputs.get_val(
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
            ),
            Mission.Constraints.MAX_MACH: inputs.get_val(Mission.Constraints.MAX_MACH),
        }

        prob.model.add_subsystem(
            'hydraulics',
            TransportHydraulicsGroupMass(**options),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.PLANFORM_AREA, 1500.0, 'ft**2')
        prob.set_val(Aircraft.Hydraulics.SYSTEM_PRESSURE, 5000.0, 'psi')
        prob.set_val(Aircraft.Wing.AREA, 1000.0, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class AltHydraulicsGroupMassTest(unittest.TestCase):
    """Tests alternate hydraulics mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'hydraulics', AltHydraulicsGroupMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Wing.AREA,
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD,
                Aircraft.VerticalTail.AREA,
                Aircraft.Hydraulics.MASS_SCALER,
            ],
            output_keys=Aircraft.Hydraulics.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltHydraulicsGroupMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.hydraulics as hydraulics

        hydraulics.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.hydraulics as hydraulics

        hydraulics.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'hydraulics', AltHydraulicsGroupMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.AREA, 10.0, 'ft**2')
        prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, 10.0, 'ft**2')
        prob.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1, 'unitless')
        prob.set_val(Aircraft.VerticalTail.AREA, 10.0, 'ft**2')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
