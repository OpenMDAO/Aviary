import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.subsystems.propulsion.utils import EngineDataInterpolator, build_engine_deck
from aviary.subsystems.propulsion.utils import EngineModelVariables as keys
from aviary.utils.named_values import NamedValues
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.variables import Aircraft, Dynamic


class DataInterpolationTest(unittest.TestCase):
    def test_data_interpolation(self):
        tol = 1e-6

        aviary_values = get_flops_inputs('LargeSingleAisle2FLOPS')
        aviary_values.set_val(Aircraft.Engine.GLOBAL_THROTTLE, True)

        model = build_engine_deck(aviary_values)

        mach_number = model.data[keys.MACH]
        altitude = model.data[keys.ALTITUDE]
        throttle = model.data[keys.THROTTLE]
        thrust = model.data[keys.THRUST]
        fuel_flow_rate = model.data[keys.FUEL_FLOW]

        inputs = NamedValues()
        inputs.set_val(Dynamic.Atmosphere.MACH, mach_number)
        inputs.set_val(Dynamic.Mission.ALTITUDE, altitude, units='ft')
        inputs.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, throttle)

        outputs = {
            Dynamic.Vehicle.Propulsion.THRUST: 'lbf',
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE: 'lbm/h',
        }

        test_mach_list = np.linspace(0, 0.85, 5)
        test_alt_list = np.linspace(0, 40_000, 5)
        test_throttle_list = np.linspace(0, 1, 5)

        test_mach, test_alt, test_throttle = np.meshgrid(
            test_mach_list, test_alt_list, test_throttle_list
        )

        num_nodes = len(test_mach.flatten())

        engine_data = om.IndepVarComp()
        engine_data.add_output(
            Dynamic.Atmosphere.MACH + '_train',
            val=np.array(mach_number),
            units='unitless',
        )
        engine_data.add_output(
            Dynamic.Mission.ALTITUDE + '_train',
            val=np.array(altitude),
            units='ft',
        )
        engine_data.add_output(
            Dynamic.Vehicle.Propulsion.THROTTLE + '_train',
            val=np.array(throttle),
            units='unitless',
        )
        engine_data.add_output(
            Dynamic.Vehicle.Propulsion.THRUST + '_train',
            val=np.array(thrust),
            units='lbf',
        )
        engine_data.add_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE + '_train',
            val=np.array(fuel_flow_rate),
            units='lbm/h',
        )

        engine_interpolator = EngineDataInterpolator(
            num_nodes=num_nodes,
            interpolator_inputs=inputs,
            interpolator_outputs=outputs,
            interpolation_method='slinear',
        )

        prob = om.Problem()
        prob.model.add_subsystem('engine_data', engine_data, promotes=['*'])
        prob.model.add_subsystem('interpolator', engine_interpolator, promotes=['*'])

        prob.setup()

        prob.set_val(Dynamic.Atmosphere.MACH, np.array(test_mach.flatten()), 'unitless')
        prob.set_val(Dynamic.Mission.ALTITUDE, np.array(test_alt.flatten()), 'ft')
        prob.set_val(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            np.array(test_throttle.flatten()),
            'unitless',
        )

        prob.run_model()

        interp_thrust = prob.get_val(Dynamic.Vehicle.Propulsion.THRUST, 'lbf')
        interp_fuel_flow = prob.get_val(Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE, 'lbm/h')

        expected_thrust = [
            0.00000000e00,
            3.54196788e02,
            6.13575369e03,
            1.44653862e04,
            2.65599096e04,
            -3.53133516e02,
            5.80901330e01,
            4.31423671e03,
            1.15501937e04,
            2.19787604e04,
            -4.73948819e03,
            -5.95006857e02,
            3.45048183e03,
            9.91821892e03,
            1.90355375e04,
            -5.43678184e03,
            -1.26223518e03,
            2.84220746e03,
            8.67608781e03,
            1.73404730e04,
            -6.15599576e03,
            -1.94268249e03,
            1.99628555e03,
            7.48137208e03,
            1.59182336e04,
            -9.08601805e03,
            5.39769591e02,
            5.35677809e03,
            1.21142135e04,
            2.26706698e04,
            -3.58679582e03,
            -2.69259549e01,
            3.89947345e03,
            9.58195492e03,
            1.81800588e04,
            -3.78789411e03,
            -4.33551733e02,
            3.14978564e03,
            8.31981857e03,
            1.57364339e04,
            -4.37631447e03,
            -9.62213760e02,
            2.53081763e03,
            7.40602072e03,
            1.43984450e04,
            -5.01565861e03,
            -1.51671152e03,
            1.87792594e03,
            6.52078396e03,
            1.34001292e04,
            -1.81720361e04,
            7.25342394e02,
            4.57780250e03,
            9.76304079e03,
            1.87814301e04,
            -3.32023617e03,
            -2.63838155e02,
            3.32181093e03,
            7.48346225e03,
            1.35905054e04,
            -1.70547950e03,
            -1.07909398e03,
            7.12762853e02,
            5.99809055e03,
            1.17250860e04,
            -3.31584709e03,
            -6.62192338e02,
            2.21942781e03,
            6.13595362e03,
            1.14564170e04,
            -3.87532147e03,
            -1.09074056e03,
            1.75956633e03,
            5.56019585e03,
            1.08820247e04,
            -2.72580542e04,
            9.10915198e02,
            3.79882691e03,
            7.41186808e03,
            1.48921903e04,
            -3.05367653e03,
            -5.00750355e02,
            2.74414840e03,
            5.38496959e03,
            9.00095204e03,
            1.42902796e03,
            -2.56968919e03,
            -3.71509100e03,
            2.94919637e03,
            7.57037021e03,
            -2.70147046e03,
            -5.76530275e02,
            1.71686482e03,
            4.88952762e03,
            8.31589720e03,
            -2.80337875e03,
            -7.11794624e02,
            1.54949925e03,
            4.54611557e03,
            8.21531510e03,
            -3.63440722e04,
            1.09648800e03,
            3.01985132e03,
            5.06069537e03,
            1.10029506e04,
            -2.78711688e03,
            -7.37662555e02,
            2.16648587e03,
            3.28647692e03,
            4.41139864e03,
            4.56353542e03,
            -4.06028441e03,
            -8.14294484e03,
            -9.96978057e01,
            3.41565444e03,
            -3.71372809e03,
            -1.50841466e03,
            6.94449058e02,
            3.10611945e03,
            5.29028554e03,
            -2.72709143e03,
            -1.02697915e03,
            7.18371174e02,
            2.92355330e03,
            5.46566595e03,
        ]

        expected_fuel_flow = [
            0.0,
            784.57285754,
            2348.89278057,
            5207.14987319,
            9804.78619087,
            -82.1064164,
            767.9051634,
            2370.28450696,
            5385.95260343,
            10005.6473849,
            -1257.46789462,
            640.93270326,
            2491.41105639,
            5619.63531352,
            10283.2705124,
            -1731.30009381,
            479.01364582,
            2654.76634078,
            5884.39467187,
            11678.58301406,
            -2209.3392748,
            317.43357365,
            2677.74875142,
            6101.09281052,
            12085.93480086,
            -1324.33749838,
            576.62564909,
            2003.50853694,
            4297.49813227,
            8920.18884072,
            -818.75078773,
            517.92823002,
            2021.46552219,
            4336.76135656,
            8303.53636898,
            -1009.09452361,
            492.92667628,
            2113.44966162,
            4587.663812,
            8727.89498417,
            -1361.40028651,
            402.52184706,
            2214.55868319,
            4870.95427868,
            9392.5084187,
            -1766.02516195,
            287.54068271,
            2279.99341748,
            5129.0875517,
            9982.6035355,
            -2648.67499676,
            368.67844064,
            1658.1242933,
            3387.84639134,
            8035.59149057,
            -891.35864118,
            248.9028017,
            1628.47391324,
            3270.2157656,
            5931.5008023,
            -346.72644954,
            103.68457403,
            1132.82096916,
            4165.19469995,
            8111.56286758,
            -991.50047921,
            326.03004831,
            1774.35102561,
            3857.51388549,
            7106.43382334,
            -1322.7110491,
            257.64779176,
            1882.23808354,
            4157.08229287,
            7879.27227013,
            -3973.01249515,
            160.73123218,
            1312.74004967,
            2478.19465042,
            7150.99414042,
            -963.96649463,
            -20.12262663,
            1235.48230429,
            2203.67017464,
            3559.46523561,
            662.96289568,
            -556.75610165,
            -368.87273293,
            4389.93259999,
            9274.46251656,
            -821.01947442,
            190.11665662,
            1293.98507822,
            2959.36438862,
            5327.19716906,
            -922.52811183,
            214.43709568,
            1454.9322017,
            3192.99190078,
            5922.9275864,
            -5297.34999353,
            -47.21597627,
            967.35580604,
            1568.5429095,
            6266.39679027,
            -1036.57434808,
            -289.14805495,
            842.49069534,
            1137.12458369,
            1187.42966893,
            1672.6522409,
            -1217.19677732,
            -1870.56643501,
            4614.67050003,
            10437.36216555,
            -1385.91734397,
            -350.97279954,
            683.12046854,
            1935.47982391,
            3474.12897476,
            -1052.57680083,
            -120.81294188,
            835.8741762,
            2105.98459735,
            3898.27697684,
        ]

        assert_near_equal(interp_thrust, expected_thrust, tolerance=tol)
        assert_near_equal(interp_fuel_flow, expected_fuel_flow, tolerance=tol)


if __name__ == '__main__':
    # unittest.main()
    test = DataInterpolationTest()
    test.test_data_interpolation()
