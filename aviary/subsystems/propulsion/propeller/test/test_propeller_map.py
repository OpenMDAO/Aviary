import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.propeller.propeller_map import PropellerMap
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft
from aviary.utils.aviary_values import AviaryValues


class PropellerMapTest(unittest.TestCase):
    """
    Test propeller map using OpenMDAO interpolator to make sure
    it provides correct error and gets correct Mach type.
    """

    def test_general_aviation(self):
        # The case when prop_type is helical_mach.
        tol = 0.005
        aviary_options = AviaryValues()
        prop_file_path = 'models/engines/propellers/general_aviation.csv'
        aviary_options.set_val(Aircraft.Engine.Propeller.DATA_FILE, prop_file_path)

        prob = om.Problem()
        prob.model.add_subsystem('propeller_map', PropellerMap(), promotes=['*'])
        setup_model_options(prob, aviary_options)
        prob.setup()
        prob.set_val('mach', 0.56568)  # targeting helical mach of 0.8
        prob.set_val('tip_mach', 0.56568)
        prob.set_val('power_coefficient', 0.1)
        prob.set_val('advance_ratio', 0.75)
        prob.run_model()

        # Mach, CP, J from general_aviation, expected CT: 0.0934
        ct = prob.get_val('thrust_coefficient')
        assert_near_equal(ct, 0.0934, tolerance=tol)

    def test_propfan(self):
        # The case when prop_type is mach.
        tol = 0.005
        aviary_options = AviaryValues()
        prop_file_path = 'models/engines/propellers/PropFan.csv'
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


if __name__ == '__main__':
    unittest.main()
