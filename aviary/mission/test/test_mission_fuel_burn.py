import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.variable_info.variables import Mission
from aviary.models.missions.energy_state_default import phase_info
import aviary.api as av


@use_tempdirs
class FuelBurnTestCases(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = av.AviaryProblem()

        # Load aircraft and options data from provided sources
        self.prob.aviary_inputs = self.prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv', phase_info
        )

        # clear out the takeoff fuel burn because in the csv it's non-zero
        self.prob.aviary_inputs.set_val(Mission.Takeoff.FUEL, val=0, units='lbm')

        self.prob.check_and_preprocess_inputs()

        self.prob.build_model()

        self.prob.add_driver('SLSQP', max_iter=50)

        self.prob.add_design_variables()

        self.prob.add_objective()

    def test_mission_fuel_burned(self):
        self.prob.setup()

        tol = 1e-4
        self.prob.run_aviary_problem()

        fuel_burned = self.prob.get_val(Mission.FUEL, units='lbm')
        block_fuel = self.prob.get_val(Mission.BLOCK_FUEL, units='lbm')

        assert_near_equal(fuel_burned, 13234.43186723, tol)
        assert_near_equal(block_fuel, 13234.43186723, tol)

    def test_takeoff_fuel_burned(self):
        self.prob.aviary_inputs.set_val(Mission.Takeoff.FUEL, val=500, units='lbm')

        self.prob.setup()

        tol = 1e-4
        self.prob.run_aviary_problem()

        fuel_burned = self.prob.get_val(Mission.FUEL, units='lbm')
        block_fuel = self.prob.get_val(Mission.BLOCK_FUEL, units='lbm')

        assert_near_equal(fuel_burned, 13736.6226374, tol)
        assert_near_equal(block_fuel, 13736.6226374, tol)

    def test_taxi_out_fuel_burned(self):
        self.prob.aviary_inputs.set_val(Mission.Taxi.FUEL_TAXI_OUT, val=200, units='lbm')

        self.prob.setup()

        tol = 1e-4
        self.prob.run_aviary_problem()

        fuel_burned = self.prob.get_val(Mission.FUEL, units='lbm')
        block_fuel = self.prob.get_val(Mission.BLOCK_FUEL, units='lbm')

        assert_near_equal(fuel_burned, 13435.30839861, tol)
        assert_near_equal(block_fuel, 13435.30839861, tol)

    def test_taxi_in_fuel_burned(self):
        self.prob.aviary_inputs.set_val(Mission.Taxi.FUEL_TAXI_IN, val=100, units='lbm')

        self.prob.setup()

        tol = 1e-4
        self.prob.run_aviary_problem()

        fuel_burned = self.prob.get_val(Mission.FUEL, units='lbm')
        block_fuel = self.prob.get_val(Mission.BLOCK_FUEL, units='lbm')

        assert_near_equal(fuel_burned, 13234.43186723, tol)
        assert_near_equal(block_fuel, 13334.43186723, tol)
