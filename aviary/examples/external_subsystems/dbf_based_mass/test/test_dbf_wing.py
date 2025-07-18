import unittest
import numpy as np

import openmdao.api as om

from aviary.variable_info.variables import Aircraft
from aviary.examples.external_subsystems.dbf_based_mass.dbf_wing import DBFWingMass
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


class TestDBFWingMass(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.dbf = DBFWingMass()

        self.prob.model.add_subsystem(
            'dbf_wing', self.dbf, promotes_inputs=['*'], promotes_outputs=['*']
        )

        # Set required options
        ribs = np.array([0] * 15 + [1] * 5)
        rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
        rib_thicks = np.where(ribs != 0, 0.125, 0.125)

        self.dbf.options['rib_materials'] = rib_materials
        self.dbf.options['rib_thicknesses'] = (rib_thicks, 'inch')
        self.dbf.options['rib_lightening_factor'] = (2 / 3, 'unitless')
        self.dbf.options['num_spars'] = (1.1, 'unitless')
        self.dbf.options['spar_outer_diameter'] = (1, 'inch')
        self.dbf.options['spar_wall_thickness'] = (0.0625, 'inch')
        self.dbf.options['spar_density'] = (2, 'g/cm**3')
        self.dbf.options['skin_density'] = (20, 'g/m**2')
        self.dbf.options['glue_factor'] = (0.15, 'unitless')
        self.dbf.options['stringer_thickness'] = (0.375, 'inch')
        self.dbf.options['stringer_density'] = (160, 'kg/m**3')
        self.dbf.options['num_stringers'] = (2.5, 'unitless')
        self.dbf.options['sheeting_thickness'] = (0.03125, 'inch')
        self.dbf.options['sheeting_density'] = (160, 'kg/m**3')
        self.dbf.options['sheeting_coverage'] = (0.4, 'unitless')
        self.dbf.options['sheeting_lightening_factor'] = (1.0, 'unitless')
        self.dbf.options['airfoil_data_file'] = (
            'aviary/examples/external_subsystems/dbf_based_mass/mh84-il.csv'
        )
        self.dbf.options['miscel_mass'] = (0.0, 'kg')

        self.prob.setup(force_alloc_complex=True)

        # Inputs to the component (defined via add_aviary_input)
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, val=20, units='inch')
        self.prob.set_val(Aircraft.Wing.SPAN, val=4.667, units='ft')
        self.prob.set_val(Aircraft.Wing.WETTED_AREA, val=0.85, units='m**2')

    def test_mass_output(self):
        self.prob.run_model()

        actual_mass = self.prob.get_val(Aircraft.Wing.MASS, units='kg')
        print('Computed Mass:', actual_mass)

        expected_mass = 0.799  # <<< Update to match new output once verified
        tol = 1e-2

        assert_near_equal(actual_mass, expected_mass, tolerance=tol)

    def test_partials(self):
        self.prob.run_model()
        partials_data = self.prob.check_partials(compact_print=False, method='cs')
        assert_check_partials(partials_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
