import unittest
import numpy as np
import os
import openmdao.api as om

from aviary.subsystems.mass.UAV_mass.horizontaltail import (
    HorizontalTailMass,
)
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

class TestHorizontalTailMass(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        hm = HorizontalTailMass()

        self.prob.model.add_subsystem(
            'horizontal_tail', hm, promotes_inputs=['*'], promotes_outputs=['*']
        )

        # Set required options
        ribs = np.array([0] * 15 + [1] * 5)
        rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
        rib_thicks = np.array([0.0032] * 20)
        
        hm.options[Aircraft.HorizontalTail.STRINGER_THICKNESS] = (0.0032, 'm')
        hm.options[Aircraft.HorizontalTail.RIB_MATERIALS] = rib_materials
        hm.options[Aircraft.HorizontalTail.RIB_THICKNESS] = (rib_thicks, 'm')
        hm.options[Aircraft.HorizontalTail.RIB_LIGHTENING_FACTOR] = 2/3
        hm.options[Aircraft.HorizontalTail.NUM_SPARS] = 1.0
        hm.options[Aircraft.HorizontalTail.SPAR_OUTER_DIAMETER] = (0.015,'m')
        hm.options[Aircraft.HorizontalTail.SPAR_WALL_THICKNESS] = (0.003, 'm')
        hm.options[Aircraft.HorizontalTail.SPAR_DENSITY] = (1500.0, 'kg/m**3')
        hm.options[Aircraft.HorizontalTail.AREAL_SKIN_DENSITY] = (0.08, 'kg/m**2')
        hm.options[Aircraft.HorizontalTail.GLUE_FACTOR] = 0.15
        hm.options[Aircraft.HorizontalTail.STRINGER_DENSITY] = (160.0, 'kg/m**3')
        hm.options[Aircraft.HorizontalTail.NUM_STRINGERS] = 2.0
        hm.options[Aircraft.HorizontalTail.SHEETING_THICKNESS] = (0.0016, 'm')
        hm.options[Aircraft.HorizontalTail.SHEETING_DENSITY] = (160.0, 'kg/m**3')
        hm.options[Aircraft.HorizontalTail.SHEETING_COVERAGE] = 0.4
        hm.options[Aircraft.HorizontalTail.SHEETING_LIGHTENING_FACTOR] = 1.0
        airfoil = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'utils', 'mh84-il.csv')
        )        
        hm.options[Aircraft.HorizontalTail.AIRFOIL_PATH] = airfoil
        hm.options[Aircraft.HorizontalTail.MISC_MASS] = (0.0, 'kg')

        self.prob.setup(force_alloc_complex=True)

        # Inputs to the component (defined via add_aviary_input)
        self.prob.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.508, units='m')
        self.prob.set_val(Aircraft.HorizontalTail.SPAN, 1.4225, units='m')
        self.prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, 0.85, units='m**2')

    def test_mass_output(self):
        self.prob.run_model()

        actual_mass = self.prob.get_val(Aircraft.HorizontalTail.MASS, units='kg')
        print('Computed Mass:', actual_mass)

        expected_mass = 0.70565484  # <<< Update to match any new output once they are verified
        tol = 1e-6

        assert_near_equal(actual_mass, expected_mass, tolerance=tol)

    def test_partials(self):
        self.prob.run_model()
        partials_data = self.prob.check_partials(compact_print=True, method='cs')
        assert_check_partials(partials_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
