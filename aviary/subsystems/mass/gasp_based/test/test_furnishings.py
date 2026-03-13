import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.furnishings import FurnishingMass, BWBFurnishingMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class FurnishingMassTestCase1(unittest.TestCase):
    """Created based on EquipMassTestCase1"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # large_single_aisle_1_GASP.csv
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft'
        )  # unknown
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # unknown
        self.prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless'
        )  # generic_BWB_GASP.csv
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2'
        )  # unknown

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 13266.56, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase2(unittest.TestCase):
    """Test mass-weight conversion"""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless'
        )  # large_single_aisle_1_GASP.csv
        options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless'
        )  # arbitrary

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # large_single_aisle_1_GASP.csv
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless'
        )  # generic BWB GASP.csv
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase3(unittest.TestCase):
    """
    Created based on GASP BWB model where SWF is DHYDRAL
    NUM_PASSENGERS < 50
    """

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=49, units='unitless')
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # large_single_aisle_1_GASP.csv
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=19.365, units='ft'
        )  # Unknown origin
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, 71.5245514, units='ft'
        )  # unknown origin
        self.prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless'
        )  # generic_BWB_GASP.csv
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, val=1283.5249, units='ft**2'
        )

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        SMOOTH_MASS_DISCONTINUITIES = False
        """
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 3348.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_case2(self):
        """
        SMOOTH_MASS_DISCONTINUITIES = True
        """
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()
        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 3348.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class BWBFurnishingMassTestCase1(unittest.TestCase):
    """
    Created based on GASP BWB model
    """

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless'
        )

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'furnishing',
            BWBFurnishingMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 150000, units='lbm'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.365, units='ft'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, 71.5245514, units='ft'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2'
        )  # generic_BWB_GASP.csv

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        USE_EMPIRICAL_EQUATION = True
        SMOOTH_MASS_DISCONTINUITIES = False
        """
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 11269.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_case2(self):
        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=False, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 18839.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        # assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 11269.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=False, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 18839.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class BWBFurnishingMassTestCase2(unittest.TestCase):
    """
    Created based on GASP BWB model
    GROSS_MASS < 10000
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'furnishing',
            BWBFurnishingMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 9999.0, units='lbm')  # arbitrary
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.365, units='ft'
        )  # arbitrary
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')  # arbitrary
        prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2'
        )  # arbitrary

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 590.935, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
