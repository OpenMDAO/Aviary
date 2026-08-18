import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import aviary.api as av
from aviary.models.missions.energy_state_default import phase_info
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


@use_tempdirs
class MinimumAircraftTestCase(unittest.TestCase):
    """Test the minimum FLOPS aircraft with the default energy-state mission."""

    def test_minimum_aircraft_mission(self):
        prob = av.AviaryProblem()
        prob.aviary_inputs = prob.load_inputs(
            'models/aircraft/minimum_single_aisle/minimum_single_aisle_FLOPS.csv',
            phase_info,
        )
        prob.aviary_inputs.set_val(Mission.Takeoff.FUEL_MASS, 0.0, units='lbm')
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('SLSQP', max_iter=50)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem(make_plots=False)

        assert_near_equal(prob.get_val(Mission.FUEL_MASS, units='lbm'), 18672.21, 1.0e-4)
        assert_near_equal(prob.get_val(Mission.OPERATING_MASS, units='lbm'), 86251.10, 1.0e-4)
        assert_near_equal(prob.get_val(Aircraft.Design.GROSS_MASS, units='lbm'), 135723.32, 1.0e-4)
        assert_near_equal(prob.get_val(Mission.RANGE, units='nmi'), 1906.0, 1.0e-6)

        positive_variables = (
            Aircraft.AirConditioning.MASS,
            Aircraft.Avionics.MASS,
            Aircraft.Electrical.MASS,
            Aircraft.Fuel.FUEL_SYSTEM_MASS,
            Aircraft.Furnishings.MASS,
            Aircraft.Wing.MASS,
            Aircraft.Fuselage.MASS,
            Aircraft.HorizontalTail.MASS,
            Aircraft.Hydraulics.MASS,
            Aircraft.Instruments.MASS,
            Aircraft.VerticalTail.MASS,
            Aircraft.Engine.MASS,
            Aircraft.LandingGear.TOTAL_MASS,
            Aircraft.Nacelle.MASS,
            Aircraft.Wing.AREA,
            Aircraft.Wing.SPAN,
            Aircraft.Wing.ASPECT_RATIO,
        )
        for variable in positive_variables:
            values = np.asarray(prob.get_val(variable))
            self.assertTrue(np.all(np.isfinite(values)), variable)
            self.assertTrue(np.all(values > 0.0), variable)

        for variable in (
            Dynamic.Vehicle.LIFT,
            Dynamic.Vehicle.DRAG,
            Dynamic.Vehicle.LIFT_COEFFICIENT,
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
        ):
            values = np.asarray(prob.get_val(f'traj.climb.timeseries.{variable}'))
            self.assertTrue(np.all(np.isfinite(values)), variable)
            self.assertTrue(np.all(values > 0.0), variable)


if __name__ == '__main__':
    unittest.main()
