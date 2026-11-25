import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.fuselage import BWBFuselageMass, FuselageMass

from aviary.variable_info.variables import Aircraft


# this is the large single aisle 1 V3 test case
class FuselageMassTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('fuselage', FuselageMass(), promotes=['*'])

        self.prob.model.set_input_defaults('fus_mass_full', val=102270, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, val=4000, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.TailBoom.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18763, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-12, rtol=1e-12)


class BWBFuselageMassTestCase(unittest.TestCase):
    """GASP data."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('fuselage', BWBFuselageMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_COEFFICIENT, 0.889, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, 4573.8833, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL, 0.2, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA, 5.0, units='lbm/ft**2'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 27159.69841266, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-11, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
