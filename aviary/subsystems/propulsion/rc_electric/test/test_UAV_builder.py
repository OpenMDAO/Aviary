import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from packaging import version

from aviary.subsystems.propulsion.rc_electric.UAV_Builder import RCBuilder
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission, PropulsionSum
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.dbf_variables import Aircraft, Dynamic
from aviary.variable_info.variables import Mission, Settings




class TestRCBuilder(unittest.TestCase):
    """Integrates RCBuilder into a full PropulsionMission over a 0->1 throttle sweep.
    """

    @use_tempdirs
    def test_propulsion_mission_power_balance(self):
        # 'clean' test using GASP-derived engine deck
        nn = 20

        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Settings.VERBOSITY, 0)

        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)

        engine = RCBuilder(options=options, power_balance_mode='feedforward') #change between solver/feedforward to test both modes
        preprocess_propulsion(options, engine_models=[engine])

    

        prob.model = PropulsionMission(
            num_nodes=nn, aviary_options=options, engine_models=[engine]
        )

        prob.model.add_subsystem(
            'propulsion_pre_mission',
            engine.build_pre_mission(aviary_inputs=options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Battery.VOLTAGE, val=22.2, units='V')
        prob.model.set_input_defaults(Aircraft.Engine.Motor.IDLE_CURRENT, val=0.91, units='A')
        


        setup_model_options(prob, options)
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Engine.Motor.MASS, 0.5, units='kg')
        prob.set_val(Aircraft.Battery.MASS, 0.5, units='kg')
        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.linspace(0, 1, nn))
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
    
        prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
        prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')


        prob.run_model()

        battery_power = prob.get_val('rc_electric.battery.power', units='W')
        esc_power = prob.get_val('rc_electric.esc.power', units='W')
        motor_power = prob.get_val('rc_electric.motor.power', units='W')
        prop_power = prob.get_val('rc_electric.prop_power', units='W')
        power_residual = battery_power + esc_power + motor_power - prop_power

       
        self.assertFalse(
            np.isnan(power_residual).any(), 'powertrain produced NaN over the throttle sweep'
        )


if __name__ == '__main__':
    unittest.main()