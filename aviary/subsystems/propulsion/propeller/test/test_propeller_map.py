import unittest

import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp_semi import InterpNDSemi
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.propeller.propeller_map import PropellerMap
from aviary.subsystems.propulsion.utils import PropellerModelVariables as keys
from aviary.utils.csv_data_file import read_data_file
from aviary.variable_info.enums import OutMachType
from aviary.variable_info.functions import setup_model_options
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
        prop_file_path = 'models/engines/propellers/general_aviation.prop'
        aviary_options.set_val(Aircraft.Engine.Propeller.DATA_FILE, prop_file_path)

        prob = om.Problem()
        prob.model.add_subsystem('propeller_map', PropellerMap(), promotes=['*'])
        setup_model_options(prob, aviary_options)
        prob.setup()
        prob.set_val('helical_mach', 0.8)
        prob.set_val('power_coefficient', 0.1)
        prob.set_val('advance_ratio', 0.75)
        prob.run_model()

        # Mach, CP, J from general_aviation, expected CT: 0.0934
        ct = prob.get_val('thrust_coefficient')
        assert_near_equal(ct, 0.0934, tolerance=tol)

    def test_propfan(self):
        # The case when prop_type is mach.
        tol = 0.005
        aviary_options = get_option_defaults()
        prop_file_path = 'models/engines/propellers/PropFan.prop'
        aviary_options.set_val(
            Aircraft.Engine.Propeller.DATA_FILE, val=prop_file_path, units='unitless'
        )

        prob = om.Problem()
        prob.model.add_subsystem('propeller_map', PropellerMap(), promotes=['*'])
        setup_model_options(prob, aviary_options)
        prob.setup()
        prob.set_val('mach', 0.25)
        prob.set_val('power_coefficient', 0.0465)
        prob.set_val('advance_ratio', 0.2)
        prob.run_model()

        # Mach, CP, J from PropFan, expected CT: 0.095985
        ct = prob.get_val('thrust_coefficient')
        assert_near_equal(ct, 0.095985, tolerance=tol)

    # def test_mach_type(self):
    #     # Test reading prop_type from .prop file.
    #     aviary_options = get_option_defaults()
    #     prop_file_path = 'models/engines/propellers/general_aviation.prop'
    #     aviary_options.set_val(
    #         Aircraft.Engine.Propeller.DATA_FILE, val=prop_file_path, units='unitless'
    #     )
    #     aviary_options.set_val(
    #         Aircraft.Engine.INTERPOLATION_METHOD, val='slinear', units='unitless'
    #     )
    #     prop_model = PropellerMap('prop', aviary_options)
    #     out_mach_type = prop_model.read_and_set_mach_type(prop_file_path)
    #     self.assertEqual(out_mach_type, OutMachType.HELICAL_MACH)


if __name__ == '__main__':
    unittest.main()
