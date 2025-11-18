import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fuselage import (
    AltFuselageMass,
    BWBAftBodyMass,
    BWBFuselageMass,
    TransportFuselageMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_options,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS']


@use_tempdirs
class FuselageMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'fuselage',
            TransportFuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Fuselage.MASS_SCALER,
            ],
            output_keys=Aircraft.Fuselage.MASS,
            version=Version.TRANSPORT,
            atol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class FuselageMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.fuselage as fuselage

        fuselage.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.fuselage as fuselage

        fuselage.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'fuselage',
            TransportFuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options('AdvancedSingleAisle', preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.LENGTH, 100.0, 'ft')
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 10.0, 'ft')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class AltFuselageMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'fuselage',
            AltFuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.MASS_SCALER,
                Aircraft.Fuselage.WETTED_AREA,
                Aircraft.Fuselage.MAX_HEIGHT,
                Aircraft.Fuselage.MAX_WIDTH,
            ],
            output_keys=Aircraft.Fuselage.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltFuselageMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.fuselage as fuselage

        fuselage.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.fuselage as fuselage

        fuselage.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'fuselage',
            AltFuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        prob.setup(check=False, force_alloc_complex=True)
        prob.set_val(Aircraft.Fuselage.WETTED_AREA, 4000.0, 'ft**2')
        prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, 15.0, 'ft')
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, 15.0, 'ft')

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class BWBFuselageMassTest(unittest.TestCase):
    """Tests fuselage mass calculation for BWB."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case1(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'fuselage',
            BWBFuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        self.prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Mission.Design.GROSS_MASS,
                Aircraft.Fuselage.CABIN_AREA,
            ],
            output_keys=Aircraft.Fuselage.MASS,
            version=Version.BWB,
            atol=1e-10,
        )


@use_tempdirs
class BWBAftBodyMassTest(unittest.TestCase):
    """Tests aft body mass calculation for BWB."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case1(self, case_name):
        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 3, units='unitless')
        prob = self.prob

        prob.model.add_subsystem(
            'aftbody',
            BWBAftBodyMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        setup_model_options(self.prob, aviary_options)
        self.prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Mission.Design.GROSS_MASS,
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Wing.COMPOSITE_FRACTION,
            ],
            output_keys=[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Wing.BWB_AFTBODY_MASS],
            version=Version.BWB,
        )


if __name__ == '__main__':
    unittest.main()
