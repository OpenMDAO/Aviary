import numpy as np
import unittest

from openmdao.components.interp_util.interp_semi import InterpNDSemi
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.propeller.propeller_map import PropellerMap
from aviary.subsystems.propulsion.utils import PropellerModelVariables as keys
from aviary.variable_info.enums import OutMachType
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class PropellerMapTest(unittest.TestCase):
    """
    Test propeller map using OpenMDAO interpolator to make sure 
    it provides correct error and gets correct Mach type.
    """

    def test_general_aviation(self):
        # The case when prop_type is helical_mach.
        tol = 0.005
        aviary_options = get_option_defaults()
        prop_file_path = 'models/propellers/general_aviation.prop'
        aviary_options.set_val(
            Aircraft.Engine.PROPELLER_DATA_FILE, val=prop_file_path, units='unitless')
        aviary_options.set_val(
            Aircraft.Engine.INTERPOLATION_METHOD, val='slinear', units='unitless')
        aviary_options.set_val(
            Aircraft.Engine.USE_PROPELLER_MAP, val=True, units='unitless')
        prop_model = PropellerMap('prop', aviary_options)
        prop_model.build_propeller_interpolator(3, aviary_options)

        x1 = prop_model.data[keys.MACH]
        x2 = prop_model.data[keys.CP]
        x3 = prop_model.data[keys.J]
        y = prop_model.data[keys.CT]
        grid = np.array([x1, x2, x3]).T
        interp = InterpNDSemi(grid, y, method='slinear')
        # Mach, CP, J from general_aviation, expected CT: 0.0934
        x = np.array([0.8, 0.1, 0.75])
        f, df_dx = interp.interpolate(x, compute_derivative=True)
        assert_near_equal(f, 0.0934, tolerance=tol)

    def test_propfan(self):
        # The case when prop_type is mach.
        tol = 0.005
        aviary_options = get_option_defaults()
        prop_file_path = 'models/propellers/PropFan.prop'
        aviary_options.set_val(
            Aircraft.Engine.PROPELLER_DATA_FILE, val=prop_file_path, units='unitless')
        aviary_options.set_val(
            Aircraft.Engine.INTERPOLATION_METHOD, val='slinear', units='unitless')
        aviary_options.set_val(
            Aircraft.Engine.USE_PROPELLER_MAP, val=True, units='unitless')
        prop_model = PropellerMap('prop', aviary_options)
        prop_model.build_propeller_interpolator(3, aviary_options)

        x1 = prop_model.data[keys.MACH]
        x2 = prop_model.data[keys.CP]
        x3 = prop_model.data[keys.J]
        y = prop_model.data[keys.CT]
        grid = np.array([x1, x2, x3]).T
        interp = InterpNDSemi(grid, y, method='slinear')
        # Mach, CP, J from PropFan, expected CT: 0.095985
        x = np.array([0.25, 0.0465, 0.2])
        f, df_dx = interp.interpolate(x, compute_derivative=True)
        assert_near_equal(f, 0.095985, tolerance=tol)

    def test_mach_type(self):
        # Test reading prop_type from .prop file.
        aviary_options = get_option_defaults()
        prop_file_path = 'models/propellers/general_aviation.prop'
        aviary_options.set_val(
            Aircraft.Engine.PROPELLER_DATA_FILE, val=prop_file_path, units='unitless')
        aviary_options.set_val(
            Aircraft.Engine.INTERPOLATION_METHOD, val='slinear', units='unitless')
        aviary_options.set_val(
            Aircraft.Engine.USE_PROPELLER_MAP, val=True, units='unitless')
        prop_model = PropellerMap('prop', aviary_options)
        out_mach_type = prop_model.read_and_set_mach_type(prop_file_path)
        self.assertEqual(out_mach_type, OutMachType.HELICAL_MACH)


if __name__ == "__main__":
    unittest.main()
