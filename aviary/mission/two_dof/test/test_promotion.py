import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.two_dof_default import phase_info
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class Test2DOFMissionPromotion(unittest.TestCase):
    """Testing parameter promotion from the phases."""

    def test_promotion(self):

        prob = AviaryProblem()

        prob.load_inputs(
            "validation_cases/validation_data/test_models/aircraft_for_bench_GwGm.csv",
            phase_info,
            verbosity=0,
        )

        # These values should be promoted to the top.
        prob.aviary_inputs.set_val(Mission.Takeoff.AIRPORT_ALTITUDE, 2345.0, units='ft')
        prob.aviary_inputs.set_val(Mission.Landing.AIRPORT_ALTITUDE, 3456.0, units='ft')
        prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 15.0, units='deg')
        prob.aviary_inputs.set_val(Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF, 7.0, units='deg')

        prob.check_and_preprocess_inputs()
        prob.build_model()

        prob.add_design_variables()
        prob.add_objective(objective_type='fuel_burned')

        prob.setup(check=False)
        prob.set_initial_guesses()

        prob.run_model()

        airport_alt = prob.get_val(Mission.Takeoff.AIRPORT_ALTITUDE, units='ft')
        assert_near_equal(airport_alt, 2345.0)

        airport_alt = prob.get_val(f'traj.groundroll.rhs_all.{Mission.Takeoff.AIRPORT_ALTITUDE}', units='ft')
        assert_near_equal(airport_alt, 2345.0)
        airport_alt = prob.get_val(f'traj.rotation.rhs_all.{Mission.Takeoff.AIRPORT_ALTITUDE}', units='ft')
        assert_near_equal(airport_alt, 2345.0)
        airport_alt = prob.get_val(f'traj.ascent.rhs_all.{Mission.Takeoff.AIRPORT_ALTITUDE}', units='ft')
        assert_near_equal(airport_alt, 2345.0)

        # Bypass promotion system to make sure they are promoted by checking the value.
        landing_alt = prob.model.landing.aerodynamics.kclge._inputs['airport_alt']
        assert_near_equal(landing_alt, 3456.0)
        landing_alt = prob.model.landing.aerodynamics.drag_coef._inputs['airport_alt']
        assert_near_equal(landing_alt, 3456.0)
        landing_alt = prob.model.landing.aero_td.kclge._inputs['airport_alt']
        assert_near_equal(landing_alt, 3456.0)

        incidence = prob.get_val(f'traj.groundroll.rhs_all.{Aircraft.Wing.INCIDENCE}', units='deg')
        assert_near_equal(incidence, 15.0)
        incidence = prob.get_val(f'traj.climb1.rhs_all.{Aircraft.Wing.INCIDENCE}', units='deg')
        assert_near_equal(incidence, 15.0)
        incidence = prob.get_val(f'traj.desc1.rhs_all.{Aircraft.Wing.INCIDENCE}', units='deg')
        assert_near_equal(incidence, 15.0)

        # Bypass promotion system to make sure they are promoted by checking the value.
        defl = prob.model.traj.phases.groundroll.rhs_all.aerodynamics.kclge._inputs['flap_defl']
        assert_near_equal(defl, 7.0)
        defl = prob.model.traj.phases.ascent.rhs_all.aerodynamics.kclge._inputs['flap_defl']
        assert_near_equal(defl, 7.0)


if __name__ == '__main__':
    unittest.main()
