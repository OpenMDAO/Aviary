import unittest
import numpy as np
import os
import openmdao.api as om

from aviary.subsystems.mass.UAV_mass.verticaltail import VerticalTailMass
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

class TestVerticalTailMass(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        vm = VerticalTailMass()

        self.prob.model.add_subsystem(
            'vtail', vm, promotes_inputs=['*'], promotes_outputs=['*']
        )

        # Set required options
        ribs = np.array([0] * 15 + [1] * 5)
        rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
        rib_thicks = np.array([0.0032] * 20)

        vm.options[Aircraft.VerticalTail.STRINGER_THICKNESS] = (0.0032, 'm')
        vm.options[Aircraft.VerticalTail.RIB_MATERIALS] = rib_materials
        vm.options[Aircraft.VerticalTail.RIB_THICKNESS] = (rib_thicks, 'm')
        vm.options[Aircraft.VerticalTail.RIB_LIGHTENING_FACTOR] = 2/3
        vm.options[Aircraft.VerticalTail.NUM_SPARS] = 1.0
        vm.options[Aircraft.VerticalTail.SPAR_OUTER_DIAMETER] = (0.015,'m')
        vm.options[Aircraft.VerticalTail.SPAR_WALL_THICKNESS] = (0.003, 'm')
        vm.options[Aircraft.VerticalTail.SPAR_DENSITY] = (1500, 'kg/m**3')
        vm.options[Aircraft.VerticalTail.AREAL_SKIN_DENSITY] = (0.08, 'kg/m**2')
        vm.options[Aircraft.VerticalTail.GLUE_FACTOR] = 0.15
        vm.options[Aircraft.VerticalTail.STRINGER_DENSITY] = (160, 'kg/m**3')
        vm.options[Aircraft.VerticalTail.NUM_STRINGERS] = 2.0
        vm.options[Aircraft.VerticalTail.SHEETING_THICKNESS] = (0.0016, 'm')
        vm.options[Aircraft.VerticalTail.SHEETING_DENSITY] = (160.0, 'kg/m**3')
        vm.options[Aircraft.VerticalTail.SHEETING_COVERAGE] = 0.4
        vm.options[Aircraft.VerticalTail.SHEETING_LIGHTENING_FACTOR] = 1.0
        airfoil = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'utils', 'mh84-il.csv')
        )        
        vm.options[Aircraft.VerticalTail.AIRFOIL_PATH] = airfoil
        vm.options[Aircraft.VerticalTail.MISC_MASS] = (0.0, 'kg')

        self.prob.setup(force_alloc_complex=True)

        # Inputs to the component (defined via add_aviary_input)
        self.prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.508, units='m')
        self.prob.set_val(Aircraft.VerticalTail.SPAN, 1.4225, units='m')
        self.prob.set_val(Aircraft.VerticalTail.WETTED_AREA, 0.85, units='m**2')

    def test_mass_output(self):
        self.prob.run_model()

        actual_mass = self.prob.get_val(Aircraft.VerticalTail.MASS, units='kg')
        print('Computed Mass:', actual_mass)

        expected_mass = 0.70565484 # <<< Update to match new output once verified
        tol = 1e-6

        assert_near_equal(actual_mass, expected_mass, tolerance=tol)

    def test_partials(self):
        self.prob.run_model()
        partials_data = self.prob.check_partials(compact_print=True, method='cs')
        assert_check_partials(partials_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
