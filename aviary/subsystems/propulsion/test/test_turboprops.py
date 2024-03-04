import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging import version

from aviary.subsystems.propulsion.engine_deck import TurboPropDeck
from aviary.subsystems.propulsion.propulsion_mission import (
    PropulsionMission, PropulsionSum)
from aviary.interface.utils.markdown_utils import round_it
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.functions import get_path
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class TurboPropTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def prepare_model(self, filename, test_points=[(0, 0, 0), (0, 0, 1)]):
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
        options.set_val(Aircraft.Engine.SCALE_FACTOR, 1)
        options.set_val(Aircraft.Engine.GENERATE_FLIGHT_IDLE, False)
        options.set_val(Aircraft.Engine.IGNORE_NEGATIVE_THRUST, False)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_THRUST_FRACTION, 0.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MAX_FRACTION, 1.0)
        options.set_val(Aircraft.Engine.FLIGHT_IDLE_MIN_FRACTION, 0.08)
        options.set_val(Aircraft.Engine.GEOPOTENTIAL_ALT, False)
        options.set_val(Aircraft.Engine.INTERPOLATION_METHOD, 'slinear')

        engine = TurboPropDeck(options=options)
        preprocess_propulsion(options, [engine])

        self.prob.model = PropulsionMission(
            num_nodes=len(test_points), aviary_options=options)

        machs, alts, throttles = zip(*test_points)
        IVC = om.IndepVarComp(Dynamic.Mission.MACH,
                              np.array(machs),
                              units='unitless')
        IVC.add_output(Dynamic.Mission.ALTITUDE,
                       np.array(alts),
                       units='ft')
        IVC.add_output(Dynamic.Mission.THROTTLE,
                       np.array(throttles),
                       units='unitless')
        self.prob.model.add_subsystem('IVC', IVC, promotes=['*'])

        self.prob.setup(force_alloc_complex=True)
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, options.get_val(
            Aircraft.Engine.SCALE_FACTOR), units='unitless')

    def get_results(self, point_names=None, display_results=False):
        # shp = self.prob.get_val(Dynamic.Mission.SHAFT_POWER_CORRECTED, units='hp')
        shp = self.prob.get_val('engine_deck.shaft_power_corrected_unscaled', units='hp')
        total_thrust = self.prob.get_val(Dynamic.Mission.THRUST, units='lbf')
        prop_thrust = self.prob.get_val(
            'engine_deck.total_thrust.prop_thrust', units='lbf')
        tailpipe_thrust = self.prob.get_val(
            'engine_deck.total_thrust.tailpipe_thrust', units='lbf')
        fuel_flow = self.prob.get_val(
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE, units='lbm/h')

        results = {'SHP': shp, 'total_thrust': total_thrust, 'prop_thrust': prop_thrust,
                   'tailpipe_thrust': tailpipe_thrust, 'fuel_flow': fuel_flow}
        if display_results:
            results['names'] = point_names
            if point_names:
                for n, name in enumerate(point_names):
                    print(name,
                          round_it(shp[n]),
                          round_it(total_thrust[n]),
                          round_it(prop_thrust[n]),
                          round_it(tailpipe_thrust[n]),
                          round_it(fuel_flow[n]))
            else:
                print(shp)
                print(total_thrust)
                print(prop_thrust)
                print(tailpipe_thrust)
                print(fuel_flow)
        results = []
        for n, _ in enumerate(shp):
            results.append(
                (round_it(shp[n]),
                 round_it(tailpipe_thrust[n]),
                 round_it(fuel_flow[n][0])))
        return results

    def test_case_1(self):
        # test using GASP-derived engine deck that includes tailpipe thrust
        filename = get_path('models/engines/turboprop_1120hp.deck')
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        point_names = ['idle', 'SLS', 'TOC']
        truth_vals = [(112, 37.7, 195.8), (1120, 136.3, 644), (1742.5, 21.3, 839.7)]
        self.prepare_model(filename, test_points)

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)

    def test_case_2(self):
        # test using GASP-derived engine deck that does not include tailpipe thrust
        filename = get_path('models/engines/turboprop_1120hp_no_tailpipe.deck')
        test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        point_names = ['idle', 'SLS', 'TOC']
        truth_vals = [(112, 0, 195.8), (1120, 0, 644), (1742.5, 0, 839.7)]
        self.prepare_model(filename, test_points)

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)


if __name__ == "__main__":
    unittest.main()
