import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials

from aviary.subsystems.aerodynamics.flops_based.mux_component import MuxComponent
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


class TestMuxComponent(unittest.TestCase):

    def test_mux(self):
        prob = om.Problem()
        model = prob.model

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.VerticalTail.NUM_TAILS, 1)
        aviary_options.set_val(Aircraft.Fuselage.NUM_FUSELAGES, 1)
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

        model.add_subsystem(
            'mux', MuxComponent(aviary_options=aviary_options),
            promotes_inputs=['*'])

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.WETTED_AREA, 2.0)
        prob.set_val(Aircraft.Wing.FINENESS, 3.0)
        prob.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 4.0)
        prob.set_val(Aircraft.Wing.LAMINAR_FLOW_UPPER, 0.01)
        prob.set_val(Aircraft.Wing.LAMINAR_FLOW_LOWER, 0.02)

        prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, 12.0)
        prob.set_val(Aircraft.HorizontalTail.FINENESS, 13.0)
        prob.set_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, 14.0)
        prob.set_val(Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER, 0.03)
        prob.set_val(Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER, 0.04)

        prob.set_val(Aircraft.VerticalTail.WETTED_AREA, 22.0)
        prob.set_val(Aircraft.VerticalTail.FINENESS, 23.0)
        prob.set_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, 24.0)
        prob.set_val(Aircraft.VerticalTail.LAMINAR_FLOW_UPPER, 0.05)
        prob.set_val(Aircraft.VerticalTail.LAMINAR_FLOW_LOWER, 0.06)

        prob.set_val(Aircraft.Fuselage.WETTED_AREA, 32.0)
        prob.set_val(Aircraft.Fuselage.FINENESS, 33.0)
        prob.set_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, 34.0)
        prob.set_val(Aircraft.Fuselage.LAMINAR_FLOW_UPPER, 0.07)
        prob.set_val(Aircraft.Fuselage.LAMINAR_FLOW_LOWER, 0.08)

        prob.set_val(Aircraft.Nacelle.WETTED_AREA, np.array([42.0]))
        prob.set_val(Aircraft.Nacelle.FINENESS, np.array([43.0]))
        prob.set_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, np.array([44.0]))
        prob.set_val(Aircraft.Nacelle.LAMINAR_FLOW_UPPER, np.array([0.09]))
        prob.set_val(Aircraft.Nacelle.LAMINAR_FLOW_LOWER, np.array([0.10]))

        prob.run_model()
        prob.check_partials(compact_print=True, method="cs")

        derivs = prob.check_partials(out_stream=None, method="cs")

        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

        # Truth values for each variable
        wetted_areas_truth = np.array([2., 12., 22., 32., 42., 42.])
        fineness_ratios_truth = np.array([3., 13., 23., 33., 43., 43.])
        characteristic_lengths_truth = np.array([4., 14., 24., 34., 44., 44.])
        laminar_fractions_upper_truth = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.09])
        laminar_fractions_lower_truth = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.1])

        wetted_areas_output = prob['mux.wetted_areas']
        fineness_ratios_output = prob['mux.fineness_ratios']
        characteristic_lengths_output = prob['mux.characteristic_lengths']
        laminar_fractions_upper_output = prob['mux.laminar_fractions_upper']
        laminar_fractions_lower_output = prob['mux.laminar_fractions_lower']

        # Assert that the outputs are near the expected values
        assert_near_equal(wetted_areas_output, wetted_areas_truth, 1e-7)
        assert_near_equal(fineness_ratios_output, fineness_ratios_truth, 1e-7)
        assert_near_equal(characteristic_lengths_output,
                          characteristic_lengths_truth, 1e-7)
        assert_near_equal(laminar_fractions_upper_output,
                          laminar_fractions_upper_truth, 1e-7)
        assert_near_equal(laminar_fractions_lower_output,
                          laminar_fractions_lower_truth, 1e-7)


if __name__ == "__main__":
    unittest.main()
