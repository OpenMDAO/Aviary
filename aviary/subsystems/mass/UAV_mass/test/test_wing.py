import unittest
import numpy as np
import os
import openmdao.api as om

from aviary.subsystems.mass.UAV_mass.variable_info.enums import WingType
from aviary.subsystems.mass.UAV_mass.wing import WingMass
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft

class TestWingMass(unittest.TestCase):
    #Creates a separate problem for tests for each wing design
    def build_problem(self, wing_type):
        prob = om.Problem()
        wm = WingMass()
        prob.model.add_subsystem('wing', wm, promotes_inputs=['*'], promotes_outputs=['*'])

        #Rib definitions
        ribs = np.array([0] * 15 + [1] * 5)
        rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
        rib_thicks = np.array([0.0032] * 20)

        wm.options[Aircraft.Wing.RIB_MATERIALS] = rib_materials
        wm.options[Aircraft.Wing.RIB_THICKNESS] = (rib_thicks, 'inch')

        #Airfoil path
        airfoil = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils', 'mh84-il.csv'))

        #Tests for the simple wing design
        if wing_type == WingType.SIMPLE:
            wm.options[Aircraft.Wing.TYPE] = WingType.SIMPLE
            wm.options[Aircraft.Wing.FOAM_DENSITY] = (32.0, 'kg/m**3')
            wm.options[Aircraft.Wing.ROD_DENSITY] = (1500.0, 'kg/m**3')
            wm.options[Aircraft.Wing.ROD_RADIUS] = (0.003, 'm')
            wm.options[Aircraft.Wing.ROD_THICKNESS] = (0.0005, 'm')
            wm.options[Aircraft.Wing.AIRFOIL_PATH] = airfoil

        #Tests for the medium wing design
        elif wing_type == WingType.MEDIUM:
            wm.options[Aircraft.Wing.TYPE] = WingType.MEDIUM
            wm.options[Aircraft.Wing.RIB_LIGHTENING_FACTOR] = 2/3
            wm.options[Aircraft.Wing.NUM_SPARS] = 1.0
            wm.options[Aircraft.Wing.SPAR_OUTER_DIAMETER] = (0.015, 'm')       
            wm.options[Aircraft.Wing.SPAR_WALL_THICKNESS] = (0.003, 'm')            
            wm.options[Aircraft.Wing.SPAR_DENSITY] = (1500.0, 'kg/m**3')          
            wm.options[Aircraft.Wing.AREAL_SKIN_DENSITY] = (0.08, 'kg/m**2')          
            wm.options[Aircraft.Wing.GLUE_FACTOR] = 0.15
            wm.options[Aircraft.Wing.STRINGER_THICKNESS] = (0.0032, 'm')           
            wm.options[Aircraft.Wing.STRINGER_DENSITY] = (160.0, 'kg/m**3')            
            wm.options[Aircraft.Wing.NUM_STRINGERS] = 2.0
            wm.options[Aircraft.Wing.SHEETING_THICKNESS] = (0.0016, 'm')
            wm.options[Aircraft.Wing.SHEETING_DENSITY] = (160, 'kg/m**3')            
            wm.options[Aircraft.Wing.SHEETING_COVERAGE] = 0.4
            wm.options[Aircraft.Wing.SHEETING_LIGHTENING_FACTOR] = 1.0
            wm.options[Aircraft.Wing.AIRFOIL_PATH] = airfoil
            wm.options[Aircraft.Wing.MISC_MASS] = (0.0, 'kg')

        # Inputs
        prob.setup(force_alloc_complex=True)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')
        prob.set_val(Aircraft.Wing.SPAN, 1.4225, units='m')
        prob.set_val(Aircraft.Wing.WETTED_AREA, 0.85, units='m**2')

        return prob

    #Simple wing test
    def test_simple_mass(self):
        prob = self.build_problem(WingType.SIMPLE)
        prob.run_model()

        actual = prob.get_val(Aircraft.Wing.MASS, units='kg')
        expected = 0.97041034
        assert_near_equal(actual, expected, tolerance=1e-6)

    #Medium wing test
    def test_medium_mass(self):
        prob = self.build_problem(WingType.MEDIUM)
        prob.run_model()

        actual = prob.get_val(Aircraft.Wing.MASS, units='kg')
        expected = 0.46738585

        assert_near_equal(actual, expected, tolerance=1e-6)

    #Partials test
    def test_partials_1(self):
        prob = self.build_problem(WingType.MEDIUM)
        prob.run_model()
        partials = prob.check_partials(compact_print=False, method='cs')
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)

    def test_partials_2(self):
        prob = self.build_problem(WingType.SIMPLE)
        prob.run_model()
        partials = prob.check_partials(compact_print=False, method='cs')
        assert_check_partials(partials, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()