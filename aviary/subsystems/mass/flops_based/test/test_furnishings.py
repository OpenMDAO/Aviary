import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.furnishings import (
    AltFurnishingsGroupMass, AltFurnishingsGroupMassBase,
    BWBFurnishingsGroupMass, TransportFurnishingsGroupMass)
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class TransportFurnishingsGroupMassTest(unittest.TestCase):
    '''
    Tests transport/GA furnishings mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'furnishings',
            TransportFurnishingsGroupMass(
                aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Furnishings.MASS_SCALER,
                        Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH,
                        Aircraft.Fuselage.MAX_WIDTH,
                        Aircraft.Fuselage.MAX_HEIGHT],
            output_keys=Aircraft.Furnishings.MASS,
            version=Version.TRANSPORT)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class BWBFurnishingsGroupMassTest(unittest.TestCase):
    '''
    Tests BWB furnishings mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob
        flops_inputs = get_flops_inputs(case_name, preprocess=True)
        flops_inputs.update({
            Aircraft.Fuselage.MILITARY_CARGO_FLOOR: (False, 'unitless'),
            Aircraft.BWB.NUM_BAYS: (5, 'unitless')
        })

        prob.model.add_subsystem(
            'furnishings',
            BWBFurnishingsGroupMass(aviary_options=flops_inputs),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        # TODO: add FLOPS tests cases with BWB furnishings mass calculations

        # These inputs aren't in the FLOPS input data so we'll give dummy values here,
        # instead of trying to transfer them from the FLOPS input data. The test
        # case will only check the partials.
        prob.set_val(Aircraft.BWB.CABIN_AREA, 1000.0, units='ft**2')
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 30.0, units='deg')

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Furnishings.MASS_SCALER,
                        # Aircraft.BWB.CABIN_AREA,
                        # Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                        Aircraft.Fuselage.MAX_WIDTH,
                        Aircraft.Fuselage.MAX_HEIGHT],
            output_keys=Aircraft.AirConditioning.MASS,
            version=Version.BWB,
            tol=1.0e-3,
            atol=1e-11)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltFurnishingsGroupMassBaseTest(unittest.TestCase):
    '''
    Tests alternate base furnishings mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'furnishings',
            AltFurnishingsGroupMassBase(
                aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=Aircraft.Furnishings.MASS_SCALER,
            output_keys=Aircraft.Furnishings.MASS_BASE,
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltFurnishingsGroupMassTest(unittest.TestCase):
    '''
    Tests alternate furnishings mass calculation.
    '''

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            'furnishings',
            AltFurnishingsGroupMass(
                aviary_options=get_flops_inputs(case_name, preprocess=True)),
            promotes_outputs=['*'],
            promotes_inputs=['*']
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.Furnishings.MASS_BASE,
                        Aircraft.Design.STRUCTURE_MASS,
                        Aircraft.Propulsion.MASS,
                        Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE],
            output_keys=Aircraft.Furnishings.MASS,
            version=Version.ALTERNATE)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
