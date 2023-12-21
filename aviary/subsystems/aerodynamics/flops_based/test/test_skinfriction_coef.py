import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.flops_based.skin_friction import SkinFriction
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class SkinFrictionCoeffTest(unittest.TestCase):

    def test_derivs(self):
        n = 12
        nc = 3

        machs = np.array([.2, .3, .4, .5, .6, .7, .75, .775, .8, .825, .85, .875])
        alts = np.linspace(41000, 41000, n)
        lens = np.linspace(1, 2, nc)
        temp = np.ones(n) * 389.97
        pres = np.ones(n) * 2.60239151

        prob = om.Problem()
        model = prob.model

        options = {}
        options[Aircraft.VerticalTail.NUM_TAILS] = (0, 'unitless')
        options[Aircraft.Fuselage.NUM_FUSELAGES] = (1, 'unitless')
        options[Aircraft.Engine.NUM_ENGINES] = ([0], 'unitless')

        model.add_subsystem(
            'cf', SkinFriction(num_nodes=n, aviary_options=AviaryValues(options)))

        prob.setup(force_alloc_complex=True)

        prob.set_val('cf.temperature', temp[:n])
        prob.set_val('cf.static_pressure', pres[:n])
        prob.set_val('cf.mach', machs[:n])
        prob.set_val('cf.characteristic_lengths', lens[:nc])

        prob.run_model()

        derivs = prob.check_partials(method='cs', out_stream=None)

        # Atol set higher because some derivs are on the order 1e7
        # TODO: need to test values too
        assert_check_partials(derivs, atol=1e-08, rtol=1e-12)

    def test_skin_friction_algorithm(self):
        # Test vs aviary1 algorithm output.
        n = 12
        nc = 3

        machs = np.array([.2, .3, .4, .5, .6, .7, .75, .775, .8, .825, .85, .875])
        alts = np.linspace(41000, 41000, n)
        lens = np.linspace(1, 2, nc)
        temp = np.ones(n) * 389.97
        pres = np.ones(n) * 374.74437747

        prob = om.Problem()
        model = prob.model

        options = {}
        options[Aircraft.VerticalTail.NUM_TAILS] = (0, 'unitless')
        options[Aircraft.Fuselage.NUM_FUSELAGES] = (1, 'unitless')
        options[Aircraft.Engine.NUM_ENGINES] = ([0], 'unitless')

        model.add_subsystem(
            'cf', SkinFriction(num_nodes=n, aviary_options=AviaryValues(options)))

        prob.setup(force_alloc_complex=True)

        prob.set_val('cf.temperature', temp[:n])
        prob.set_val('cf.static_pressure', pres[:n])
        prob.set_val('cf.mach', machs[:n])
        prob.set_val('cf.characteristic_lengths', lens[:nc])

        prob.run_model()

        data = np.array([
            [0.005396523989004909, 365059.9023227319],
            [0.004966765578135659, 547589.8534840979],
            [0.004690457476125405, 730119.8046454638],
            [0.004944328211075446, 547589.8534840978],
            [0.004562147270958159, 821384.7802261466],
            [0.004315727215872133, 1095179.7069681955],
            [0.004643056166682007, 730119.8046454638],
            [0.00429139345173995, 1095179.7069681957],
            [0.004064221738701749, 1460239.6092909276],
            [0.004414783804315559, 912649.7558068297],
            [0.004085425540068907, 1368974.6337102444],
            [0.0038723696464216576, 1825299.5116136593],
            [0.004228034334994051, 1095179.7069681955],
            [0.0039162899726263006, 1642769.5604522931],
            [0.0037144161374653376, 2190359.413936391],
            [0.004067313880014865, 1277709.6581295615],
            [0.0037702154884929626, 1916564.487194342],
            [0.003577666619089076, 2555419.316259123],
            [0.003993874389670162, 1368974.6337102444],
            [0.003703301621412952, 2053461.9505653665],
            [0.0035149162534008037, 2737949.267420489],
            [0.0039585444977749615, 1414607.121500586],
            [0.003671073155610844, 2121910.682250879],
            [0.003484668543701865, 2829214.243001172],
            [0.003924040592737333, 1460239.6092909276],
            [0.0036395743600069483, 2190359.4139363915],
            [0.003455090229668116, 2920479.2185818553],
            [0.003890297607565131, 1505872.097081269],
            [0.0036087474909260266, 2258808.1456219032],
            [0.003426128150494066, 3011744.194162538],
            [0.003857257490610407, 1551504.5848716104],
            [0.0035785410558282315, 2327256.8773074155],
            [0.0033977349180204213, 3103009.169743221],
            [0.00382486832849627, 1597137.0726619519],
            [0.003548909032785302, 2395705.6089929277],
            [0.003369868196564652, 3194274.1453239038]
        ])

        cf = prob.get_val('cf.skin_friction_coeff').ravel()
        Re = prob.get_val('cf.Re').ravel()

        cf_diff = np.abs(data[:, 0] - cf) / np.max(cf)
        Re_diff = np.abs(data[:, 1] - Re) / np.max(Re)

        assert_near_equal(np.max(cf_diff), 0.0, 1e-4)
        assert_near_equal(np.max(Re_diff), 0.0, 1e-4)


if __name__ == "__main__":
    unittest.main()
