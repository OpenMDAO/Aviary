import unittest
import numpy as np
import openmdao.api as om
import os

from aviary.models.external_subsystems.UAV.mass.model.mass_premission import MassPremission
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft
from aviary.models.external_subsystems.UAV.mass.utils.enums import WingType


class TestMassPremission(unittest.TestCase):
    def set_defaults(self, comp, Aircraft):
        # Spars
        comp.options[Aircraft.NUM_SPARS] = 1.0
        comp.options[Aircraft.SPAR_OUTER_DIAMETER] = (0.02, 'm')
        comp.options[Aircraft.SPAR_WALL_THICKNESS] = (0.002, 'm')
        comp.options[Aircraft.SPAR_DENSITY] = (1500.0, 'kg/m**3')

        # Options that only exist for certain mass components
        if hasattr(Aircraft, 'NUM_STRINGERS'):
            comp.options[Aircraft.NUM_STRINGERS] = 2.0
        if hasattr(Aircraft, 'FLOOR_THICKNESS'):
            comp.options[Aircraft.FLOOR_THICKNESS] = (0.003, 'm')
        if hasattr(Aircraft, 'FLOOR_DENSITY'):
            comp.options[Aircraft.FLOOR_DENSITY] = (340.0, 'kg/m**3')
        if hasattr(Aircraft, 'FLOOR_LENGTH'):
            comp.options[Aircraft.FLOOR_LENGTH] = (2 / 3, 'm')
        if hasattr(Aircraft, 'STRINGER_THICKNESS'):
            comp.options[Aircraft.STRINGER_THICKNESS] = (0.005, 'm')
        if hasattr(Aircraft, 'STRINGER_DENSITY'):
            comp.options[Aircraft.STRINGER_DENSITY] = (160, 'kg/m**3')
        if hasattr(Aircraft, 'BULKHEAD_LIGHTENING_FACTOR'):
            comp.options[Aircraft.BULKHEAD_LIGHTENING_FACTOR] = 0.18
        if hasattr(Aircraft, 'FOAM_DENSITY'):
            comp.options[Aircraft.FOAM_DENSITY] = (2.0, 'kg/m**3')
        if hasattr(Aircraft, 'ROD_DENSITY'):
            comp.options[Aircraft.ROD_DENSITY] = (1500.0, 'kg/m**3')
        if hasattr(Aircraft, 'ROD_RADIUS'):
            comp.options[Aircraft.ROD_RADIUS] = (0.003, 'm')
        if hasattr(Aircraft, 'ROD_THICKNESS'):
            comp.options[Aircraft.ROD_THICKNESS] = (0.0005, 'm')

        # Sheeting
        comp.options[Aircraft.SHEETING_THICKNESS] = (0.003, 'm')
        comp.options[Aircraft.SHEETING_DENSITY] = (160.0, 'kg/m**3')
        comp.options[Aircraft.SHEETING_COVERAGE] = 1.0
        comp.options[Aircraft.SHEETING_LIGHTENING_FACTOR] = 0.3

        # Skin + glue
        comp.options[Aircraft.AREAL_SKIN_DENSITY] = (0.08, 'kg/m**2')
        comp.options[Aircraft.GLUE_FACTOR] = 0.08

        # Misc
        comp.options[Aircraft.MISC_MASS] = (0.0, 'kg')

    def setUp(self):
        base = os.path.dirname(os.path.dirname(__file__))
        airfoil_dir = os.path.join(base, 'utils')
        airfoil = os.path.abspath(os.path.join(airfoil_dir, 'mh84-il.csv'))

        self.prob = om.Problem()
        self.prob.model = MassPremission()

        self.prob.setup()

        wing = self.prob.model.wing_mass
        htail = self.prob.model.horizontal_tail_mass
        vtail = self.prob.model.vertical_tail_mass
        fuse = self.prob.model.fuselage_mass

        # Setting rib parameters
        rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
        rib_thicks = np.ones(20) * 0.004

        # Setting UAV defaults
        self.set_defaults(wing, Aircraft.Wing)
        self.set_defaults(htail, Aircraft.HorizontalTail)
        self.set_defaults(vtail, Aircraft.VerticalTail)
        self.set_defaults(fuse, Aircraft.Fuselage)

        # Setting necessary options
        wing.options[Aircraft.Wing.RIB_MATERIALS] = rib_materials
        wing.options[Aircraft.Wing.RIB_THICKNESS] = (rib_thicks, 'm')
        wing.options[Aircraft.Wing.RIB_LIGHTENING_FACTOR] = 2 / 3
        wing.options[Aircraft.Wing.AIRFOIL_PATH] = airfoil
        wing.options[Aircraft.Wing.TYPE] = WingType.MEDIUM

        htail.options[Aircraft.HorizontalTail.RIB_MATERIALS] = rib_materials
        htail.options[Aircraft.HorizontalTail.RIB_THICKNESS] = (rib_thicks, 'm')
        htail.options[Aircraft.HorizontalTail.RIB_LIGHTENING_FACTOR] = 2 / 3
        htail.options[Aircraft.HorizontalTail.AIRFOIL_PATH] = airfoil

        vtail.options[Aircraft.VerticalTail.RIB_MATERIALS] = rib_materials
        vtail.options[Aircraft.VerticalTail.RIB_THICKNESS] = (rib_thicks, 'm')
        vtail.options[Aircraft.VerticalTail.RIB_LIGHTENING_FACTOR] = 2 / 3
        vtail.options[Aircraft.VerticalTail.AIRFOIL_PATH] = airfoil

        fuse.options[Aircraft.Fuselage.BULKHEAD_MATERIALS] = rib_materials
        fuse.options[Aircraft.Fuselage.BULKHEAD_THICKNESS] = (rib_thicks, 'm')

        # Setting geometry values:
        self.prob.set_val(Aircraft.Wing.SPAN, 1.4225, units='m')
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')

        self.prob.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.508, units='m')
        self.prob.set_val(Aircraft.HorizontalTail.SPAN, 1.4225, units='m')

        self.prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.508, units='m')
        self.prob.set_val(Aircraft.VerticalTail.SPAN, 1.4225, units='m')

        self.prob.set_val(Aircraft.Fuselage.LENGTH, 1.33, units='m')
        self.prob.set_val(Aircraft.Fuselage.AVG_HEIGHT, 0.07, units='m')
        self.prob.set_val(Aircraft.Fuselage.AVG_WIDTH, 0.05, units='m')

        self.prob.run_model()

    def test_outputs(self):
        """Do all promoted mass outputs match."""

        wing = self.prob.get_val(Aircraft.Wing.MASS)[0]
        ht = self.prob.get_val(Aircraft.HorizontalTail.MASS)[0]
        vt = self.prob.get_val(Aircraft.VerticalTail.MASS)[0]
        fuse = self.prob.get_val(Aircraft.Fuselage.MASS)[0]
        total = self.prob.get_val(Aircraft.Design.STRUCTURE_MASS)[0]

        print('wing', wing)
        print('ht',ht)
        print('vt',vt)

        self.assertAlmostEqual(wing, 0.734733)
        self.assertAlmostEqual(ht, 0.734733)
        self.assertAlmostEqual(vt, 0.734733)
        self.assertAlmostEqual(fuse, 0.3191996)
        self.assertAlmostEqual(total, 2.52339859)


if __name__ == '__main__':
    unittest.main()
