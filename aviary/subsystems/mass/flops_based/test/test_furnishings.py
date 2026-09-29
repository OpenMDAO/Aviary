import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.furnishings import (
    AltFurnishingsGroupMass,
    AltFurnishingsGroupMassBase,
    BWBFurnishingsGroupMass,
    TransportFurnishingsGroupMass,
)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    get_flops_options,
    print_case,
    Version,
)
from aviary.variable_info.variables import Aircraft

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS', 'BWB300FLOPS']


@use_tempdirs
class TransportFurnishingsGroupMassTest(unittest.TestCase):
    """Tests transport/GA furnishings mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'furnishings',
            TransportFurnishingsGroupMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Furnishings.MASS_SCALER,
                Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Fuselage.MAX_HEIGHT,
            ],
            output_keys=Aircraft.Furnishings.MASS,
            version=Version.TRANSPORT,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBFurnishingsGroupMassTest(unittest.TestCase):
    """Tests BWB furnishings mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'furnishings',
            BWBFurnishingsGroupMass(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Furnishings.MASS_SCALER,
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Fuselage.MAX_HEIGHT,
                Aircraft.BWB.NUM_BAYS,
            ],
            output_keys=Aircraft.Furnishings.MASS,
            version=Version.BWB,
        )


@use_tempdirs
class AltFurnishingsGroupMassBaseTest(unittest.TestCase):
    """Tests alternate base furnishings mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'furnishings',
            AltFurnishingsGroupMassBase(),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=Aircraft.Furnishings.MASS_SCALER,
            output_keys=Aircraft.Furnishings.MASS_BASE,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class AltFurnishingsGroupMassTest(unittest.TestCase):
    """Tests alternate furnishings mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'furnishings', AltFurnishingsGroupMass(), promotes_outputs=['*'], promotes_inputs=['*']
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[
                Aircraft.Furnishings.MASS_BASE,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS_BASE,
            ],
            output_keys=Aircraft.Furnishings.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
