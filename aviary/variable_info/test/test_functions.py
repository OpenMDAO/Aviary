import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.utils.develop_metadata import add_meta_data
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output


class InputOutputOptionTest(unittest.TestCase):
    """Test the use of unit conversion when adding Aviary inputs, outputs, options."""

    def test_unit_conversion(self):
        comp = DummyComp()
        prob = om.Problem()
        prob.model.add_subsystem('comp', comp, promotes=['*'])
        prob.setup()
        prob.run_model()
        tol = 1e-4  # whether or not a unit is converted does not need high tol
        assert_near_equal(43.0556, prob.get_val('area'), tolerance=tol)
        assert_near_equal(9.84252, prob.get_val('length_out'), tolerance=tol)
        assert_near_equal(5000, prob.get_val('mass'), tolerance=tol)
        assert_near_equal(11.0231, prob.get_val('mass_out'), tolerance=tol)

        # Note: Checking OpenMDAO internals for primal_name.
        self.assertEqual(prob.model.comp._valid_name_map['length'], 'aa')
        self.assertEqual(prob.model.comp._valid_name_map['mass'], 'zz')


class DummyComp(om.ExplicitComponent):
    """Simple component to test unit conversion."""

    def initialize(self):
        add_aviary_option(self, 'mass', units='lbm', meta_data=dummy_metadata)

    def setup(self):
        add_aviary_input(self, 'length', units='ft', meta_data=dummy_metadata, primal_name='aa')

        add_aviary_output(self, 'area', units='ft**2', meta_data=dummy_metadata)
        self.add_output('length_out', val=0)
        add_aviary_output(self, 'mass', units='g', meta_data=dummy_metadata, primal_name='zz')
        self.add_output('mass_out', val=0)

    def compute(self, inputs, outputs):
        mass = self.options['mass'][0]  # should be 5 kg -> lbm
        length = inputs['length']  # should be 3 m -> ft

        # outputs['area'] should default to 4 m**2 -> ft**2
        outputs['length_out'] = length  # should return input 'length' in ft, NOT 3
        # outputs['mass'] should default to 5 kg -> g
        outputs['mass_out'] = mass  # should return option 'mass' in lbm, NOT 5


dummy_metadata = {}
add_meta_data('mass', dummy_metadata, units='kg', default_value=5)
add_meta_data('length', dummy_metadata, units='m', default_value=3)
add_meta_data('area', dummy_metadata, units='m**2', default_value=4)

if __name__ == '__main__':
    unittest.main()
