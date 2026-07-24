import unittest
import numpy as np
import openmdao.api as om

from aviary.subsystems.mass.UAV_mass.fuselage import FuselageMass
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

class TestFuselageMass(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        fm = FuselageMass()

        self.prob.model.add_subsystem(
            'fuselage', fm, promotes_inputs=['*'], promotes_outputs=['*']
        )

        ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])
        bulkhead_materials = np.where(ribs != 0, 'Ply', 'Balsa').tolist()
        rib_thicks = np.where(ribs == 2, 0.0024, 0.0016)

        fm.options[Aircraft.Fuselage.STRINGER_THICKNESS] = (0.0032, 'm')
        fm.options[Aircraft.Fuselage.BULKHEAD_MATERIALS] = bulkhead_materials
        fm.options[Aircraft.Fuselage.BULKHEAD_THICKNESS] = (rib_thicks, 'm')
        fm.options[Aircraft.Fuselage.BULKHEAD_LIGHTENING_FACTOR] = 0.18
        fm.options[Aircraft.Fuselage.NUM_SPARS] = 1.0
        fm.options[Aircraft.Fuselage.SPAR_OUTER_DIAMETER] = (0.008,'m')
        fm.options[Aircraft.Fuselage.SPAR_WALL_THICKNESS] = (0.0005, 'm')
        fm.options[Aircraft.Fuselage.SPAR_DENSITY] = (1500.0, 'kg/m**3')
        fm.options[Aircraft.Fuselage.AREAL_SKIN_DENSITY] = (0.08, 'kg/m**2')
        fm.options[Aircraft.Fuselage.GLUE_FACTOR] = 0.08
        fm.options[Aircraft.Fuselage.STRINGER_DENSITY] = (160.0, 'kg/m**3')
        fm.options[Aircraft.Fuselage.SHEETING_THICKNESS] = (0.002, 'm')
        fm.options[Aircraft.Fuselage.SHEETING_DENSITY] = (160.0, 'kg/m**3')
        fm.options[Aircraft.Fuselage.SHEETING_COVERAGE] = 0.7
        fm.options[Aircraft.Fuselage.SHEETING_LIGHTENING_FACTOR] = 0.3
        fm.options[Aircraft.Fuselage.FLOOR_LENGTH] = (1.0, 'm')
        fm.options[Aircraft.Fuselage.FLOOR_DENSITY] = (340.0, 'kg/m**3')
        fm.options[Aircraft.Fuselage.FLOOR_THICKNESS] = (0.004, 'm')
        fm.options[Aircraft.Fuselage.MISC_MASS] = (0.0, 'kg')

        self.prob.setup(force_alloc_complex=True)

        #inputs
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 1.33, units='m')
        self.prob.set_val(Aircraft.Fuselage.AVG_HEIGHT, 0.07, units='m')
        self.prob.set_val(Aircraft.Fuselage.AVG_WIDTH, 0.05, units='m')
        self.prob.set_val(Aircraft.Fuselage.WETTED_AREA, 0.583, units='m**2')

    def test_mass_output(self):
        self.prob.run_model()

        actual_mass = self.prob.get_val(Aircraft.Fuselage.MASS, units='kg')
        print('Computed Mass:', actual_mass)

        expected_mass = 0.20813777
        tol = 1e-6

        assert_near_equal(actual_mass, expected_mass, tolerance=tol)

    def test_partials(self):
        self.prob.run_model()
        partials_data = self.prob.check_partials(compact_print=True, method='cs')
        assert_check_partials(partials_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()