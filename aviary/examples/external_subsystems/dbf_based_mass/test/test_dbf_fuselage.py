import unittest
import numpy as np

import openmdao.api as om

from aviary.variable_info.variables import Aircraft
from aviary.examples.external_subsystems.dbf_based_mass.dbf_fuselage import DBFFuselageMass
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials


class TestDBFFuselageMass(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.dbf = DBFFuselageMass()

        self.prob.model.add_subsystem(
            'dbf_fuselage', self.dbf, promotes_inputs=['*'], promotes_outputs=['*']
        )

        # === Define rib layout ===
        ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])
        bulkhead_materials = np.where(ribs != 0, 'Ply', 'Balsa').tolist()
        rib_thicks = np.where(ribs == 2, 0.25, 0.125)

        self.dbf.options['bulkhead_materials'] = bulkhead_materials
        self.dbf.options['bulkhead_thicknesses'] = (rib_thicks, 'inch')
        self.dbf.options['num_spars'] = (0.5, 'unitless')
        self.dbf.options['bulkhead_lightening_factor'] = (0.18, 'unitless')
        self.dbf.options['sheeting_coverage'] = (1, 'unitless')
        self.dbf.options['sheeting_density'] = (160, 'kg/m**3')
        self.dbf.options['sheeting_lightening_factor'] = (0.3, 'unitless')
        self.dbf.options['sheeting_thickness'] = (0.03125, 'inch')
        self.dbf.options['glue_factor'] = (0.08, 'unitless')
        self.dbf.options['stringer_density'] = (160, 'kg/m**3')
        self.dbf.options['stringer_thickness'] = (0.375, 'inch')
        self.dbf.options['floor_length'] = (2, 'ft')
        self.dbf.options['floor_density'] = (340, 'kg/m**3')
        self.dbf.options['floor_thickness'] = (0.125, 'inch')
        self.dbf.options['skin_density'] = (20, 'g/m**2')
        self.dbf.options['spar_density'] = (2, 'g/cm**3')
        self.dbf.options['spar_outer_diameter'] = (1, 'inch')
        self.dbf.options['spar_wall_thickness'] = (0.0625, 'inch')
        self.dbf.options['misc_mass'] = (0.0, 'kg')

        self.prob.setup(force_alloc_complex=True)

        # === Inputs ===
        self.prob.set_val(Aircraft.Fuselage.LENGTH, val=4, units='ft')
        self.prob.set_val(Aircraft.Fuselage.AVG_HEIGHT, val=5, units='inch')
        self.prob.set_val(Aircraft.Fuselage.AVG_WIDTH, val=4, units='inch')
        self.prob.set_val(Aircraft.Fuselage.WETTED_AREA, val=904, units='inch**2')

    def test_mass_output(self):
        self.prob.run_model()

        actual_mass = self.prob.get_val(Aircraft.Fuselage.MASS, units='kg')
        print('Computed Mass:', actual_mass)

        # Update expected_mass based on verified value
        expected_mass = 0.405
        tol = 1e-3

        assert_near_equal(actual_mass, expected_mass, tolerance=tol)

    def test_partials(self):
        self.prob.run_model()
        partials_data = self.prob.check_partials(compact_print=True, method='cs')
        assert_check_partials(partials_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
