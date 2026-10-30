import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fuselage import (
    AltFuselageMass,
    BWBAftBodyMass,
    BWBFuselageMass,
    TransportFuselageMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    get_flops_options,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS', 'BWB300FLOPS']


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
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.REF_DIAMETER,
                Aircraft.Fuselage.MASS_SCALER,
            ],
            output_keys=Aircraft.Fuselage.MASS,
            version=Version.TRANSPORT,
            atol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


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
            self,
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
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Design.GROSS_MASS,
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
        flops_inputs = get_flops_inputs(case_name)
        prob = self.prob

        prob.model.add_subsystem(
            'aftbody',
            BWBAftBodyMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        setup_model_options(self.prob, flops_inputs)
        self.prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Design.GROSS_MASS,
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
