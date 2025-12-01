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

        # fmt: off
        expected_thrust = [
            0.00000000e+00, 3.55065352e+02, 6.13698222e+03, 1.44665836e+04,
            2.65615937e+04, -3.53119530e+02, 5.78899978e+01, 4.31466665e+03,
            1.15498581e+04, 2.19794592e+04, -4.73864181e+03, -5.94671264e+02,
            3.45070726e+03, 9.92076882e+03, 1.90369137e+04, -5.43519088e+03,
            -1.26106424e+03, 2.84361024e+03, 8.67974734e+03, 1.73441197e+04,
            -6.15575104e+03, -1.94432076e+03, 1.99215933e+03, 7.48264753e+03,
            1.59228344e+04, -9.08865342e+03, 5.37685968e+02, 5.35583466e+03,
            1.21151072e+04, 2.26660362e+04, -3.58793342e+03, -2.71834035e+01,
            3.89987504e+03, 9.58185277e+03, 1.81789167e+04, -3.78809006e+03,
            -4.33378636e+02, 3.15036858e+03, 8.32044337e+03, 1.57379685e+04,
            -4.37553297e+03, -9.61693742e+02, 2.53140782e+03, 7.40786480e+03,
            1.44001589e+04, -5.01552946e+03, -1.51762853e+03, 1.87563989e+03,
            6.52178915e+03, 1.34026165e+04, -1.81773068e+04, 7.20306584e+02,
            4.57468711e+03, 9.76363079e+03, 1.87704788e+04, -3.32285440e+03,
            -2.63984092e+02, 3.32372050e+03, 7.48270002e+03, 1.35883187e+04,
            -1.70555042e+03, -1.07904612e+03, 7.12944116e+02, 5.99814421e+03,
            1.17247455e+04, -3.31587507e+03, -6.62323244e+02, 2.21920540e+03,
            6.13598226e+03, 1.14561982e+04, -3.87530788e+03, -1.09093631e+03,
            1.75912045e+03, 5.56093076e+03, 1.08823986e+04, -2.72659602e+04,
            9.02927199e+02, 3.79353955e+03, 7.41215439e+03, 1.48749213e+04,
            -3.05777539e+03, -5.00784780e+02, 2.74756596e+03, 5.38354727e+03,
            8.99772073e+03, 1.42906311e+03, -2.56961554e+03, -3.71517220e+03,
            2.94773975e+03, 7.56546280e+03, -2.70156421e+03, -5.76393414e+02,
            1.71710215e+03, 4.88958815e+03, 8.31651069e+03, -2.80337627e+03,
            -7.11736241e+02, 1.54948167e+03, 4.54627324e+03, 8.21492229e+03,
            -3.63546137e+04, 1.08554782e+03, 3.01239200e+03, 5.06067799e+03,
            1.09793639e+04, -2.79269638e+03, -7.37585468e+02, 2.17141142e+03,
            3.28439452e+03, 4.40712274e+03, 4.56367663e+03, -4.06018495e+03,
            -8.14328851e+03, -1.02664701e+02, 3.40618008e+03, -3.71362507e+03,
            -1.50857755e+03, 6.94269750e+02, 3.10639193e+03, 5.29068910e+03,
            -2.72727452e+03, -1.02679445e+03, 7.18732279e+02, 2.92409044e+03,
            5.46623231e+03
        ]

        expected_fuel_flow = [
            0., 784.80638673, 2349.31436014, 5207.56361423,
            9805.42086121, -82.10073461, 767.83063507, 2370.44372031,
            5385.82159643, 10005.94167255, -1257.08525784, 641.08961166,
            2491.53226474, 5620.96913826, 10283.8742694, -1730.5170532,
            479.59595976, 2655.50337931, 5886.47010294, 11680.97519949,
            -2209.1681144, 316.50921259, 2675.30699721, 6102.00265701,
            12089.27460107, -1325.03725809, 576.06448338, 2003.17555803,
            4297.86775644, 8918.31003046, -819.17336107, 517.83396036,
            2021.62580201, 4336.71915396, 8302.99796048, -1009.1751228,
            493.00527239, 2113.7188098, 4587.99071223, 8728.77501625,
            -1361.01696195, 402.77963737, 2214.86937809, 4872.00105095,
            9393.63638683, -1765.93365194, 287.0228723, 2278.63827718,
            5129.78265006, 9984.41243031, -2650.07451618, 367.32258004,
            1657.03675592, 3388.17189864, 8031.19919972, -892.34016611,
            248.85134815, 1629.24168652, 3269.88662379, 5930.49680185,
            -346.75263699, 103.70879831, 1132.91036663, 4165.20338232,
            8111.3038636, -991.5168707, 325.96331497, 1774.23537687,
            3857.53199897, 7106.29757417, -1322.69918947, 257.53653201,
            1881.96955716, 4157.56264311, 7879.55025955, -3975.11177427,
            158.58067669, 1310.8979538, 2478.47604084, 7144.08836897,
            -965.50697114, -20.13126406, 1236.85757102, 2203.05409363,
            3557.99564321, 662.99850057, -556.71835084, -368.89137529,
            4389.08545714, 9271.4815779, -821.06449641, 190.18169299,
            1294.09858554, 2959.39577555, 5327.64621324, -922.52712826,
            214.46850819, 1454.92120063, 3193.09136912, 5922.60625685,
            -5300.14903236, -50.16122665, 964.75915169, 1568.78018304,
            6256.97753822, -1038.67377618, -289.11387628, 844.47345552,
            1136.22156347, 1185.49448458, 1672.74963814, -1217.14549998,
            -1870.69311722, 4612.96753196, 10431.65929221, -1385.86919921,
            -351.04997996, 683.04014878, 1935.63927112, 3474.42264817,
            -1052.67755001, -120.71167359, 836.07027945, 2106.31322146,
            3898.69137097
        ]

        assert_near_equal(interp_thrust, expected_thrust, tolerance=tol)
        assert_near_equal(interp_fuel_flow, expected_fuel_flow, tolerance=tol)


if __name__ == '__main__':
    # unittest.main()
    test = DataInterpolationTest()
    test.test_data_interpolation()
