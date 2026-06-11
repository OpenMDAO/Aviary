import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.landing import LandingGearMassGroup
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class FixedMassGroupTestCase1(unittest.TestCase):
    """the large single aisle 1 V3 test case"""

    def setUp(self):
        options = {
            Aircraft.Engine.NUM_ENGINES: [2],
        }

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem('landing', LandingGearMassGroup(), promotes=['*'])
        prob.model_options['*'] = options

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, val=1.0)
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.0)
        prob.model.set_input_defaults(Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2)
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, val=7.35, units='ft')
        prob.model.set_input_defaults(Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04)
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1.0)
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, val=0.85)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        expected_values = {
            Aircraft.Design.TOUCHDOWN_MASS_MAX: 175400.0,
            Aircraft.LandingGear.TOTAL_MASS: 7510.885838,
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6384.2529623,
            Aircraft.LandingGear.NOSE_GEAR_MASS: 1126.6328757,
        }
        tol = 1e-8
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class FixedMassGroupTestCase2(unittest.TestCase):
    """Gravity Modification"""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.landing as landing

        landing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.landing as landing

        landing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = {
            Aircraft.Engine.NUM_ENGINES: [2],
        }

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem('landing', LandingGearMassGroup(), promotes=['*'])
        prob.model_options['*'] = options

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, val=1.0)
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.0)
        prob.model.set_input_defaults(Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2)
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, val=7.35, units='ft')
        prob.model.set_input_defaults(Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04)
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1.0)
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, val=0.85)

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
