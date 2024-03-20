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
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings
from aviary.variable_info.enums import SpeedType, Verbosity
from aviary.variable_info.options import get_option_defaults


class TurboPropTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def prepare_model(self, filename, test_points=[(0, 0, 0), (0, 0, 1)], prop_model=None):
        options = get_option_defaults()
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

        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=4, units='unitless')

        engine = TurboPropDeck(options=options, prop_model=prop_model)
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
        self.prob.set_val(Aircraft.Engine.SCALE_FACTOR, 1, units='unitless')

    def get_results(self, point_names=None, display_results=False):
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

    def test_case_3(self):
        # test case using GASP-derived engine deck and user specified prop model
        filename = get_path('models/engines/turboprop_1120hp.deck')
        test_points = [(0, 0, 1)]
        point_names = ['SLS',]
        truth_vals = [(1120, 136.3, 644),]
        # test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        # point_names = ['idle', 'SLS', 'TOC']
        # truth_vals = [(112, 0, 195.8), (1120, 0, 644), (1742.5, 0, 839.7)]

        from aviary.subsystems.propulsion.prop_performance import PropPerf
        from aviary.variable_info.options import get_option_defaults
        from aviary.mission.gasp_based.flight_conditions import FlightConditions
        options = get_option_defaults()
        options.set_val(Aircraft.Design.COMPUTE_INSTALLATION_LOSS,
                        val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_BLADES,
                        val=4, units='unitless')
        options.set_val('speed_type', SpeedType.MACH)
        prop_group = om.Group()

        prop_group.add_subsystem(
            "fc",
            FlightConditions(num_nodes=1, input_speed_type=SpeedType.MACH),
            promotes_inputs=["rho", Dynamic.Mission.SPEED_OF_SOUND, 'mach'],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE,
                              'EAS', ('TAS', 'velocity')],
        )

        pp = prop_group.add_subsystem(
            'pp',
            PropPerf(aviary_options=options),
            promotes_inputs=['*'],
            promotes_outputs=["*", ('Thrust', 'prop_thrust')],
        )

        pp.set_input_defaults(Aircraft.Engine.PROPELLER_DIAMETER, 10, units="ft")
        pp.set_input_defaults(Dynamic.Mission.PROPELLER_TIP_SPEED, 800, units="ft/s")
        pp.set_input_defaults(Dynamic.Mission.VELOCITY, 0, units="knot")
        pp.options.set(num_nodes=len(test_points))

        self.prepare_model(filename, test_points, prop_group)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT, 0.5, units="unitless")

        if options.get_val(Settings.VERBOSITY, units='unitless') is Verbosity.DEBUG:
            om.n2(
                self.prob,
                outfile="n2.html",
                show_browser=False,
            )

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)

    def test_case_4(self):
        # test case using GASP-derived engine deck and default HS prop model.
        filename = get_path('models/engines/turboprop_1120hp.deck')
        test_points = [(0, 0, 1)]
        point_names = ['SLS',]
        truth_vals = [(1120, 136.3, 644),]
        # test_points = [(0, 0, 0), (0, 0, 1), (.6, 25000, 1)]
        # point_names = ['idle', 'SLS', 'TOC']
        # truth_vals = [(112, 0, 195.8), (1120, 0, 644), (1742.5, 0, 839.7)]

        self.prepare_model(filename, test_points, True)

        self.prob.set_val(Aircraft.Engine.PROPELLER_DIAMETER, 10.5, units="ft")
        self.prob.set_val(Aircraft.Engine.PROPELLER_ACTIVITY_FACTOR,
                          114.0, units="unitless")
        self.prob.set_val(
            Aircraft.Engine.PROPELLER_INTEGRATED_LIFT_COEFFICENT, 0.5, units="unitless")

        self.prob.run_model()
        results = self.get_results(point_names)
        assert_near_equal(results, truth_vals)


if __name__ == "__main__":
    unittest.main()
