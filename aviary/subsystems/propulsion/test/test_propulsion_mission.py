import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging import version

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission, PropulsionSum
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


class PropulsionMissionTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.options = AviaryValues()
        self.options.set_val(Settings.VERBOSITY, 0)

    @unittest.skipIf(
        version.parse(openmdao.__version__) < version.parse('3.26'),
        'Skipping due to OpenMDAO version being too low (<3.26)',
    )
    def test_case_1(self):
        # 'clean' test using GASP-derived engine deck
        nn = 20

        filename = get_path('models/engines/turbofan_24k_1.deck')

        options = self.options
        options.set_val(Aircraft.Engine.DATA_FILE, filename)
        options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        options.set_val(Aircraft.Engine.SUBSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SUPERSONIC_FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_CONSTANT_TERM, 0.0)
        options.set_val(Aircraft.Engine.FUEL_FLOW_SCALER_LINEAR_TERM, 1.0)
        options.set_val(Aircraft.Engine.CONSTANT_FUEL_CONSUMPTION, 0.0, units='lbm/h')
        options.set_val(Mission.Summary.FUEL_FLOW_SCALER, 1.0)
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 0.5)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)

        engine = EngineDeck(options=options)
        preprocess_propulsion(options, [engine])

        self.prob.model = PropulsionMission(
            num_nodes=nn, aviary_options=options, engine_models=[engine]
        )

        IVC = om.IndepVarComp(Dynamic.Atmosphere.MACH, np.linspace(0, 0.8, nn), units='unitless')
        IVC.add_output(Dynamic.Mission.ALTITUDE, np.linspace(0, 40000, nn), units='ft')
        IVC.add_output(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            np.linspace(1, 0.7, nn),
            units='unitless',
        )
        self.prob.model.add_subsystem('IVC', IVC, promotes=['*'])

        setup_model_options(self.prob, options)

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(
            Aircraft.Engine.SCALE_FACTOR,
            options.get_val(Aircraft.Engine.SCALE_FACTOR),
            units='unitless',
        )

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/h'
        )

        expected_thrust = np.array(
            [
                26561.59369395,
                24186.86894359,
                21938.27488056,
                19715.05735655,
                17506.16718894,
                15460.34459449,
                13780.48894973,
                12280.8193203,
                10975.41682925,
                9457.56468145,
                7995.21902953,
                7398.69940308,
                7148.11078578,
                6431.41457704,
                5775.06520451,
                5165.40974506,
                4583.11663348,
                3991.15103423,
                3339.07858092,
                2733.73087418,
            ]
        )

        expected_fuel_flow = np.array(
            [
                -14708.13129181,
                -14065.48817451,
                -13382.86563425,
                -12534.77028836,
                -11523.83568308,
                -10513.77300372,
                -9696.27706444,
                -8936.08244404,
                -8203.69933068,
                -8447.76373904,
                -8705.57372767,
                -7470.81543322,
                -5980.73136927,
                -5493.90702754,
                -5072.25036487,
                -4660.36809371,
                -4260.86577106,
                -3822.5941721,
                -3344.49786121,
                -2889.82801889,
            ]
        )

        assert_near_equal(thrust, expected_thrust, tolerance=1e-10)
        assert_near_equal(fuel_flow, expected_fuel_flow, tolerance=1e-10)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_propulsion_sum(self):
        nn = 2
        options = {
            Aircraft.Engine.NUM_ENGINES: np.array([3, 2]),
        }
        self.prob.model = om.Group()
        self.prob.model.add_subsystem(
            'propsum', PropulsionSum(num_nodes=nn, **options), promotes=['*']
        )

        self.prob.setup(force_alloc_complex=True)

        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.THRUST, np.array([[500.4, 423.001], [325, 6780]])
        )
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
            np.array([[602.11, 3554], [100, 9000]]),
        )
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            np.array([[123, -221.44], [-765.2, -1]]),
        )
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN, np.array([[3.01, -12], [484.2, 8123]])
        )
        self.prob.set_val(
            Dynamic.Vehicle.Propulsion.NOX_RATE, np.array([[322, 4610], [1.54, 2.844]])
        )

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        thrust_max = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lb/h'
        )
        electric_power_in = self.prob.get_val(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN_TOTAL, units='kW'
        )
        nox = self.prob.get_val(Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL, units='lb/h')

        expected_thrust = np.array([2347.202, 14535])
        expected_thrust_max = np.array([8914.33, 18300])
        expected_fuel_flow = np.array([-73.88, -2297.6])
        expected_electric_power_in = np.array([-14.97, 17698.6])
        expected_nox = np.array([10186, 10.308])

        assert_near_equal(thrust, expected_thrust, tolerance=1e-12)
        assert_near_equal(thrust_max, expected_thrust_max, tolerance=1e-12)
        assert_near_equal(fuel_flow, expected_fuel_flow, tolerance=1e-12)
        assert_near_equal(electric_power_in, expected_electric_power_in, tolerance=1e-12)
        assert_near_equal(nox, expected_nox, tolerance=1e-12)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_case_multiengine(self):
        # takes the large single aisle 2 test case and add a second set of engines to test summation
        nn = 20

        options = get_flops_inputs('LargeSingleAisle2FLOPS')
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)

        engine = build_engine_deck(options)
        engine2 = build_engine_deck(options)
        engine2.name = 'engine2'
        engine_models = [engine, engine2]
        preprocess_propulsion(options, engine_models=engine_models)

        model = self.prob.model
        prop = PropulsionMission(
            num_nodes=20,
            aviary_options=options,
            engine_models=engine_models,
        )
        model.add_subsystem('core_propulsion', prop, promotes=['*'])

        self.prob.model.add_subsystem(
            Dynamic.Atmosphere.MACH,
            om.IndepVarComp(Dynamic.Atmosphere.MACH, np.linspace(0, 0.85, nn), units='unitless'),
            promotes=['*'],
        )

        self.prob.model.add_subsystem(
            Dynamic.Mission.ALTITUDE,
            om.IndepVarComp(Dynamic.Mission.ALTITUDE, np.linspace(0, 40000, nn), units='ft'),
            promotes=['*'],
        )
        throttle = np.linspace(1.0, 0.6, nn)
        self.prob.model.add_subsystem(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            om.IndepVarComp(
                Dynamic.Vehicle.Propulsion.THROTTLE,
                np.vstack((throttle, throttle)).transpose(),
                units='unitless',
            ),
            promotes=['*'],
        )

        setup_model_options(self.prob, options, engine_models=engine_models)

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, [0.975, 0.975], units='unitless')

        self.prob.run_model()

        thrust = self.prob.get_val(Dynamic.Vehicle.Propulsion.THRUST_TOTAL, units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE_TOTAL, units='lbm/h'
        )
        nox_rate = self.prob.get_val(Dynamic.Vehicle.Propulsion.NOX_RATE_TOTAL, units='lbm/h')

        # block auto-formatting of tables
        # fmt: off
        expected_thrust = np.array(
            [
                103590.21540641, 92900.83040046, 82825.70799328, 73005.10411666, 63489.74235503,
                55210.75770546, 48313.84938232, 42275.86826606, 36870.28719096, 29717.82022574,
                26272.78176894, 24682.2638022, 22044.68474877, 19221.64939296, 16753.74585058,
                14404.83725986, 12273.31369208, 10143.03504195, 7869.72781898, 5794.48172967
            ]
        )

        expected_fuel_flow = np.array(
            [
                -38241.14135872, -36079.34764117, -33777.26289895, -31056.78302442, -28036.07645153,
                -25278.09940003, -22901.48613868, -20748.0936975, -19058.14550597, -19973.09349768,
                -17702.71563899, -14371.77422339, -12584.74775338, -11320.39115751, -10191.86597545,
                -9099.77210032, -8101.06611515, -7070.33673028, -5965.98165626, -4915.97493174
            ]
        )

        expected_nox_rate = np.array(
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        )

        # fmt: on

        assert_near_equal(thrust, expected_thrust, tolerance=1e-10)
        assert_near_equal(fuel_flow, expected_fuel_flow, tolerance=1e-10)
        assert_near_equal(nox_rate, expected_nox_rate, tolerance=1e-9)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
