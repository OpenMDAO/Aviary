import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging import version

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.propulsion_mission import (
    PropulsionMission, PropulsionSum)
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class PropulsionMissionTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"), "Skipping due to OpenMDAO version being too low (<3.26)")
    def test_case_1(self):
        # 'clean' test using GASP-derived engine deck
        nn = 20

        filename = get_path(
            'models/engines/turbofan_24k_1.deck')

        options = AviaryValues()
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
        options.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h')
        options.set_val(Aircraft.Engine.SCALE_PERFORMANCE, True)
        options.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 0.5)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, False)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')

        engine = EngineDeck(options=options)
        preprocess_propulsion(options, [engine])

        self.prob.model = PropulsionMission(num_nodes=nn, aviary_options=options)

        IVC = om.IndepVarComp(Dynamic.Mission.MACH,
                              np.linspace(0, 0.8, nn),
                              units='unitless')
        IVC.add_output(Dynamic.Mission.ALTITUDE,
                       np.linspace(0, 40000, nn),
                       units='ft')
        IVC.add_output(Dynamic.Mission.THROTTLE,
                       np.linspace(1, 0.7, nn),
                       units='unitless')
        self.prob.model.add_subsystem('IVC', IVC, promotes=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, options.get_val(
            Aircraft.Engine.SCALE_FACTOR), units='unitless')

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Mission.THRUST_TOTAL, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/h')

        expected_thrust = np.array([26559.90955398, 24186.4637312, 21938.65874407,
                                    19715.77939805, 17507.00655484, 15461.29892872,
                                    13781.56317005, 12281.64477782, 10975.64977233,
                                    9457.34056514,  7994.85977229,  7398.22905691,
                                    7147.50679938,  6430.71565916,  5774.57932944,
                                    5165.15558103,  4583.1380952,  3991.15088149,
                                    3338.98524687, 2733.56788119])

        expected_fuel_flow = np.array([-14707.1792863, -14065.2831058, -13383.11681516,
                                       -12535.21693425, -11524.37848035, -10514.44342419,
                                       -9697.03653898,  -8936.66146966, -8203.85487648,
                                       -8447.54167564,  -8705.14277314,  -7470.29404109,
                                       -5980.15247732,  -5493.23821772,  -5071.79842346,
                                       -4660.12833977,  -4260.89619679,  -3822.61002621,
                                       -3344.41332545,  -2889.68646353])

        assert_near_equal(thrust, expected_thrust, tolerance=1e-10)
        assert_near_equal(fuel_flow, expected_fuel_flow, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_propulsion_sum(self):
        nn = 2
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([3, 2]))
        # it doesn't matter what goes in engine models, as long as it is length 2
        options.set_val('engine_models', [1, 1])
        self.prob.model = om.Group()
        self.prob.model.add_subsystem('propsum',
                                      PropulsionSum(num_nodes=nn,
                                                    aviary_options=options),
                                      promotes=['*'])

        self.prob.setup(force_alloc_complex=True)

        self.prob.set_val(Dynamic.Mission.THRUST, np.array(
            [[500.4, 423.001], [325, 6780]]))
        self.prob.set_val(Dynamic.Mission.THRUST_MAX,
                          np.array([[602.11, 3554], [100, 9000]]))
        self.prob.set_val(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                          np.array([[123, -221.44], [-765.2, -1]]))
        self.prob.set_val(Dynamic.Mission.ELECTRIC_POWER,
                          np.array([[3.01, -12], [484.2, 8123]]))
        self.prob.set_val(Dynamic.Mission.NOX_RATE,
                          np.array([[322, 4610], [1.54, 2.844]]))

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Mission.THRUST_TOTAL, units='lbf')
        thrust_max = self.prob.get_val(Dynamic.Mission.THRUST_MAX_TOTAL, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lb/h')
        electric_power = self.prob.get_val(
            Dynamic.Mission.ELECTRIC_POWER_TOTAL, units='kW')
        nox = self.prob.get_val(Dynamic.Mission.NOX_RATE_TOTAL, units='lb/h')

        expected_thrust = np.array([2347.202, 14535])
        expected_thrust_max = np.array([8914.33, 18300])
        expected_fuel_flow = np.array([-73.88, -2297.6])
        expected_electric_power = np.array([-14.97, 17698.6])
        expected_nox = np.array([10186, 10.308])

        assert_near_equal(thrust, expected_thrust, tolerance=1e-12)
        assert_near_equal(thrust_max, expected_thrust_max, tolerance=1e-12)
        assert_near_equal(fuel_flow, expected_fuel_flow, tolerance=1e-12)
        assert_near_equal(electric_power, expected_electric_power, tolerance=1e-12)
        assert_near_equal(nox, expected_nox, tolerance=1e-12)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_case_2(self):
        # takes the large single aisle 2 test case and add a second set of engines to test summation
        nn = 20

        options = get_flops_inputs('LargeSingleAisle2FLOPS')

        engine = options.get_val('engine_models')[0]
        engine2 = options.deepcopy().get_val('engine_models')[0]
        engine2.name = 'engine2'
        preprocess_propulsion(options, [engine, engine2])

        self.prob.model = PropulsionMission(num_nodes=20, aviary_options=options)

        self.prob.model.add_subsystem(Dynamic.Mission.MACH,
                                      om.IndepVarComp(Dynamic.Mission.MACH,
                                                      np.linspace(0, 0.85, nn),
                                                      units='unitless'),
                                      promotes=['*'])

        self.prob.model.add_subsystem(
            Dynamic.Mission.ALTITUDE,
            om.IndepVarComp(
                Dynamic.Mission.ALTITUDE,
                np.linspace(0, 40000, nn),
                units='ft'),
            promotes=['*'])
        throttle = np.linspace(1.0, 0.6, nn)
        self.prob.model.add_subsystem(
            Dynamic.Mission.THROTTLE, om.IndepVarComp(Dynamic.Mission.THROTTLE, np.vstack((throttle, throttle)).transpose(), units='unitless'), promotes=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, [0.975], units='unitless')

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Mission.THRUST_TOTAL, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/h')
        nox_rate = self.prob.get_val(Dynamic.Mission.NOX_RATE_TOTAL, units='lbm/h')

        expected_thrust = np.array([103583.64726051,  92899.15059987,  82826.62014006,  73006.74478288,
                                    63491.73778033,  55213.71927899,  48317.05801159,  42277.98362824,
                                    36870.43915515,  29716.58670587,  26271.29434561,  24680.25359966,
                                    22043.65303425,  19221.1253513,  16754.1861966,   14405.43665682,
                                    12272.31373152,  10141.72397926,   7869.3816548,    5792.62871788])

        expected_fuel_flow = np.array([-38238.66614438, -36078.76817864, -33777.65206416, -31057.41872898,
                                       -28036.92997813, -25279.48301301, -22902.98616678, -20749.08916211,
                                       -19058.23299911, -19972.32193796, -17701.86829646, -14370.68121827,
                                       -12584.1724091,  -11320.06786905, -10192.11938107,  -9100.08365082,
                                       -8100.4835652,   -7069.62950088,  -5965.78834865,  -4914.94081538])

        expected_nox_rate = np.array(
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        assert_near_equal(thrust, expected_thrust, tolerance=1e-10)
        assert_near_equal(fuel_flow, expected_fuel_flow, tolerance=1e-10)
        assert_near_equal(nox_rate, expected_nox_rate, tolerance=1e-9)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
