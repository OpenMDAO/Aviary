import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.air_conditioning import ACMass, BWBACMass

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class ACMassTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'ac',
            ACMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )  # large_single_aisle_1_GASP
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # large_single_aisle_1_GASP
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # dont know where from
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )  # large_single_aisle_1_GASP
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft'
        )  # dont know where from

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1324.0561, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class ACMassTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.air_conditioning as ac

        ac.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.air_conditioning as ac

        ac.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'ac',
            ACMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1203.68740336, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class BWBACMassTestCase1(unittest.TestCase):
    """
    Created based on GASP BWB model
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'ac',
            BWBACMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT,
            1.155,
            units='unitless',  # generic_BWB_GASP.csv
        )
        prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 150000.0, units='lbm'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, 71.52455, units='ft'
        )  # dont know where from
        prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, 10.0, units='psi'
        )  # generic_BWB_GASP.csv
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.365, units='ft'
        )  # dont know where from

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1301.56663237, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
