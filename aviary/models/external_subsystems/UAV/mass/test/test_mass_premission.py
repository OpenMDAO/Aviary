import unittest
import numpy as np
import openmdao.api as om
import os

from aviary.models.external_subsystems.UAV.mass.model.mass_premission import MassPremission
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft
from aviary.models.external_subsystems.UAV.mass.utils.UAV_enums import WingType
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import (
    ExtendedMetaData,
)


class TestMassPremission(unittest.TestCase):
    def setUp(self):
        base = os.path.dirname(os.path.dirname(__file__))
        airfoil_dir = os.path.join(base, 'utils')
        airfoil = os.path.abspath(os.path.join(airfoil_dir, 'mh84-il.csv'))

        self.prob = om.Problem()
        self.prob.model = MassPremission()

        options = AviaryValues()

        # Setting rib parameters
        rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
        rib_thicks = np.ones(20) * 0.004

        # Setting necessary options
        # Wing
        options.set_val(Aircraft.Wing.NUM_SPARS, 1.0)
        options.set_val(Aircraft.Wing.SPAR_OUTER_DIAMETER, 0.02, units='m')
        options.set_val(Aircraft.Wing.SPAR_WALL_THICKNESS, 0.002, units='m')
        options.set_val(Aircraft.Wing.SPAR_DENSITY, 1500.0, units='kg/m**3')
        options.set_val(Aircraft.Wing.SHEETING_THICKNESS, 0.003, units='m')
        options.set_val(Aircraft.Wing.SHEETING_DENSITY, 160.0, units='kg/m**3')
        options.set_val(Aircraft.Wing.SHEETING_COVERAGE, 1.0)
        options.set_val(Aircraft.Wing.SHEETING_LIGHTENING_FACTOR, 0.3)
        options.set_val(Aircraft.Wing.AREAL_SKIN_DENSITY, 0.08, units='kg/m**2')
        options.set_val(Aircraft.Wing.GLUE_FACTOR, 0.08)
        options.set_val(Aircraft.Wing.MISC_MASS, 0.0, units='kg')
        options.set_val(Aircraft.Wing.RIB_MATERIALS, rib_materials)
        options.set_val(Aircraft.Wing.RIB_THICKNESS, rib_thicks, units='m')
        options.set_val(Aircraft.Wing.RIB_LIGHTENING_FACTOR, 2 / 3)
        options.set_val(Aircraft.Wing.AIRFOIL_PATH, airfoil)
        options.set_val(Aircraft.Wing.TYPE, WingType.MEDIUM)
        options.set_val(Aircraft.Wing.NUM_STRINGERS, 2.0)
        options.set_val(Aircraft.Wing.STRINGER_THICKNESS, 0.005, units='m')
        options.set_val(Aircraft.Wing.STRINGER_DENSITY, 160.0, units='kg/m**3')
        options.set_val(Aircraft.Wing.FOAM_DENSITY, 2.0, units='kg/m**3')
        options.set_val(Aircraft.Wing.ROD_DENSITY, 1500.0, units='kg/m**3')
        options.set_val(Aircraft.Wing.ROD_RADIUS, 0.003, units='m')
        options.set_val(Aircraft.Wing.ROD_THICKNESS, 0.0005, units='m')

        # Horizontal tail
        options.set_val(Aircraft.HorizontalTail.NUM_SPARS, 1.0)
        options.set_val(Aircraft.HorizontalTail.SPAR_OUTER_DIAMETER, 0.02, units='m')
        options.set_val(Aircraft.HorizontalTail.SPAR_WALL_THICKNESS, 0.002, units='m')
        options.set_val(Aircraft.HorizontalTail.SPAR_DENSITY, 1500.0, units='kg/m**3')
        options.set_val(Aircraft.HorizontalTail.SHEETING_THICKNESS, 0.003, units='m')
        options.set_val(Aircraft.HorizontalTail.SHEETING_DENSITY, 160.0, units='kg/m**3')
        options.set_val(Aircraft.HorizontalTail.SHEETING_COVERAGE, 1.0)
        options.set_val(Aircraft.HorizontalTail.SHEETING_LIGHTENING_FACTOR, 0.3)
        options.set_val(Aircraft.HorizontalTail.AREAL_SKIN_DENSITY, 0.08, units='kg/m**2')
        options.set_val(Aircraft.HorizontalTail.GLUE_FACTOR, 0.08)
        options.set_val(Aircraft.HorizontalTail.MISC_MASS, 0.0, units='kg')
        options.set_val(Aircraft.HorizontalTail.RIB_MATERIALS, rib_materials)
        options.set_val(Aircraft.HorizontalTail.RIB_THICKNESS, rib_thicks, units='m')
        options.set_val(Aircraft.HorizontalTail.RIB_LIGHTENING_FACTOR, 2 / 3)
        options.set_val(Aircraft.HorizontalTail.AIRFOIL_PATH, airfoil)
        options.set_val(Aircraft.HorizontalTail.NUM_STRINGERS, 2.0)
        options.set_val(Aircraft.HorizontalTail.STRINGER_THICKNESS, 0.005, units='m')
        options.set_val(Aircraft.HorizontalTail.STRINGER_DENSITY, 160.0, units='kg/m**3')

        # Vertical tail
        options.set_val(Aircraft.VerticalTail.NUM_SPARS, 1.0)
        options.set_val(Aircraft.VerticalTail.SPAR_OUTER_DIAMETER, 0.02, units='m')
        options.set_val(Aircraft.VerticalTail.SPAR_WALL_THICKNESS, 0.002, units='m')
        options.set_val(Aircraft.VerticalTail.SPAR_DENSITY, 1500.0, units='kg/m**3')
        options.set_val(Aircraft.VerticalTail.SHEETING_THICKNESS, 0.003, units='m')
        options.set_val(Aircraft.VerticalTail.SHEETING_DENSITY, 160.0, units='kg/m**3')
        options.set_val(Aircraft.VerticalTail.SHEETING_COVERAGE, 1.0)
        options.set_val(Aircraft.VerticalTail.SHEETING_LIGHTENING_FACTOR, 0.3)
        options.set_val(Aircraft.VerticalTail.AREAL_SKIN_DENSITY, 0.08, units='kg/m**2')
        options.set_val(Aircraft.VerticalTail.GLUE_FACTOR, 0.08)
        options.set_val(Aircraft.VerticalTail.MISC_MASS, 0.0, units='kg')
        options.set_val(Aircraft.VerticalTail.RIB_MATERIALS, rib_materials)
        options.set_val(Aircraft.VerticalTail.RIB_THICKNESS, rib_thicks, units='m')
        options.set_val(Aircraft.VerticalTail.RIB_LIGHTENING_FACTOR, 2 / 3)
        options.set_val(Aircraft.VerticalTail.AIRFOIL_PATH, airfoil)
        options.set_val(Aircraft.VerticalTail.NUM_STRINGERS, 2.0)
        options.set_val(Aircraft.VerticalTail.STRINGER_THICKNESS, 0.005, units='m')
        options.set_val(Aircraft.VerticalTail.STRINGER_DENSITY, 160.0, units='kg/m**3')

        # Fuselage
        options.set_val(Aircraft.Fuselage.BULKHEAD_MATERIALS, rib_materials)
        options.set_val(Aircraft.Fuselage.BULKHEAD_THICKNESS, rib_thicks, units='m')
        options.set_val(Aircraft.Fuselage.SHEETING_THICKNESS, 0.003, units='m')
        options.set_val(Aircraft.Fuselage.SHEETING_DENSITY, 160.0, units='kg/m**3')
        options.set_val(Aircraft.Fuselage.SHEETING_COVERAGE, 1.0)
        options.set_val(Aircraft.Fuselage.SHEETING_LIGHTENING_FACTOR, 0.3)
        options.set_val(Aircraft.Fuselage.AREAL_SKIN_DENSITY, 0.08, units='kg/m**2')
        options.set_val(Aircraft.Fuselage.GLUE_FACTOR, 0.08)
        options.set_val(Aircraft.Fuselage.MISC_MASS, 0.0, units='kg')
        options.set_val(Aircraft.Fuselage.FLOOR_THICKNESS, 0.003, units='m')
        options.set_val(Aircraft.Fuselage.FLOOR_DENSITY, 340.0, units='kg/m**3')
        options.set_val(Aircraft.Fuselage.FLOOR_LENGTH, 2 / 3, units='m')
        options.set_val(Aircraft.Fuselage.STRINGER_THICKNESS, 0.005, units='m')
        options.set_val(Aircraft.Fuselage.STRINGER_DENSITY, 160.0, units='kg/m**3')
        options.set_val(Aircraft.Fuselage.BULKHEAD_LIGHTENING_FACTOR, 0.18)
        options.set_val(Aircraft.Fuselage.NUM_SPARS, 1.0)
        options.set_val(Aircraft.Fuselage.SPAR_OUTER_DIAMETER, 0.02, units='m')
        options.set_val(Aircraft.Fuselage.SPAR_WALL_THICKNESS, 0.002, units='m')
        options.set_val(Aircraft.Fuselage.SPAR_DENSITY, 1500.0, units='kg/m**3')

        setup_model_options(self.prob, options, ExtendedMetaData)

        self.prob.setup()

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

        self.assertAlmostEqual(wing, 0.734733)
        self.assertAlmostEqual(ht, 0.734733)
        self.assertAlmostEqual(vt, 0.734733)
        self.assertAlmostEqual(fuse, 0.3191996)
        self.assertAlmostEqual(total, 2.52339859)


if __name__ == '__main__':
    unittest.main()
