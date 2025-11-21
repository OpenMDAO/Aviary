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
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    get_flops_options,
    print_case,
)
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class TransportFurnishingsGroupMassTest(unittest.TestCase):
    """Tests transport/GA furnishings mass calculation."""

    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
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

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob
        flops_inputs = get_flops_inputs(case_name, preprocess=True)

        opts = {
            Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS: flops_inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
            ),
            Aircraft.CrewPayload.NUM_FLIGHT_CREW: flops_inputs.get_val(
                Aircraft.CrewPayload.NUM_FLIGHT_CREW
            ),
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS: flops_inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_FIRST_CLASS
            ),
            Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS: flops_inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
            ),
            Aircraft.Fuselage.MILITARY_CARGO_FLOOR: False,
        }

        prob.model.add_subsystem(
            'furnishings',
            BWBFurnishingsGroupMass(**opts),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.model_options['*'] = get_flops_options(case_name, preprocess=True)

        prob.setup(check=False, force_alloc_complex=True)

        # TODO: add FLOPS tests cases with BWB furnishings mass calculations

        # These inputs aren't in the FLOPS input data so we'll give dummy values here,
        # instead of trying to transfer them from the FLOPS input data. The test
        # case will only check the partials.
        prob.set_val(Aircraft.Fuselage.CABIN_AREA, 1000.0, units='ft**2')
        prob.set_val(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 30.0, units='deg')
        prob.set_val(Aircraft.BWB.NUM_BAYS, 5.0, units='unitless')

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Furnishings.MASS_SCALER,
                # Aircraft.Fuselage.CABIN_AREA,
                # Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Fuselage.MAX_HEIGHT,
            ],
            output_keys=Aircraft.AirConditioning.MASS,
            version=Version.BWB,
            tol=1.0e-3,
            atol=1e-11,
            check_values=False,  # Currently no BWB validation data.
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


@use_tempdirs
class BWBFurnishingsGroupMassTest2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.flops_based.furnishings as furnishings

        furnishings.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.flops_based.furnishings as furnishings

        furnishings.GRAV_ENGLISH_LBM = 1.0

    def test_case(self):
        prob = om.Problem()

        flops_inputs = get_flops_inputs('AdvancedSingleAisle', preprocess=True)

        opts = {
            Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS: flops_inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS
            ),
            Aircraft.CrewPayload.NUM_FLIGHT_CREW: flops_inputs.get_val(
                Aircraft.CrewPayload.NUM_FLIGHT_CREW
            ),
            Aircraft.CrewPayload.Design.NUM_FIRST_CLASS: flops_inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_FIRST_CLASS
            ),
            Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS: flops_inputs.get_val(
                Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS
            ),
            Aircraft.Fuselage.MILITARY_CARGO_FLOOR: False,
        }

        prob.model.add_subsystem(
            'furnishings',
            BWBFurnishingsGroupMass(**opts),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=100.0, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.MAX_WIDTH, val=30.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.MAX_HEIGHT, val=15.0, units='ft')
        prob.model.set_input_defaults(Aircraft.BWB.NUM_BAYS, 5.0, units='unitless')
        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


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
            prob,
            case_name,
            input_keys=[
                Aircraft.Furnishings.MASS_BASE,
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE,
            ],
            output_keys=Aircraft.Furnishings.MASS,
            version=Version.ALTERNATE,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
