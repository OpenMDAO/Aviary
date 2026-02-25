import unittest
from copy import deepcopy

from openmdao.core.problem import _clear_problem_names
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.models.missions.two_dof_default import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.variable_info.enums import PhaseType


@use_tempdirs
class ProblemPhaseTestCase(unittest.TestCase):
    """
    Test the setup and run of a large single aisle commercial transport aircraft using
    GASP mass and aero method and TWO_DEGREES_OF_FREEDOM mission method. Expected outputs
    based on 'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv' model.
    """

    def setUp(self):
        _clear_problem_names()  # need to reset these to simulate separate runs

    def check_values(self, prob):
        self.assertTrue(prob.result.success)

        rtol = 1e-3

        # There are no truth values for these.
        assert_near_equal(
            prob.get_val(Mission.Design.GROSS_MASS, units='lbm'),
            171595.06049335,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.OPERATING_MASS, units='lbm'),
            95089.98897716,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm'),
            40505.07151619,
            tolerance=rtol,
        )

        assert_near_equal(
            prob.get_val(Mission.Landing.GROUND_DISTANCE, units='ft'),
            2657.88663983,
            tolerance=rtol,
        )

        assert_near_equal(prob.get_val(Mission.Summary.RANGE, units='NM'), 3675.0, tolerance=rtol)

        assert_near_equal(
            prob.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm'),
            136087.98897716,
            tolerance=rtol,
        )

    @require_pyoptsparse(optimizer='IPOPT')
    def test_bench_GwGm_IPOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='IPOPT',
            verbosity=0,
        )
        self.check_values(prob)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_bench_GwGm_SNOPT(self):
        local_phase_info = deepcopy(phase_info)
        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='SNOPT',
            verbosity=0,
        )
        self.check_values(prob)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_bench_GwGm_IPOPT_Breguet_Cruise(self):
        local_phase_info = deepcopy(phase_info)

        local_phase_info['cruise'] = {
            'subsystem_options': {'aerodynamics': {'method': 'cruise'}},
            'user_options': {
                'phase_builder': PhaseType.BREGUET_RANGE,
                'alt_cruise': (37.5e3, 'ft'),
                'mach_cruise': 10.8,
            },
            'initial_guesses': {
                # [Initial mass, delta mass] for special cruise phase.
                'mass': ([171481.0, -35000], 'lbm'),
                'initial_distance': (200.0e3, 'ft'),
                'initial_time': (1516.0, 's'),
                'altitude': (37.5e3, 'ft'),
                'mach': (0.8, 'unitless'),
            },
        }

        prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv',
            local_phase_info,
            optimizer='IPOPT',
            verbosity=0,
        )
        self.check_values(prob)


if __name__ == '__main__':
    # unittest.main()
    test = ProblemPhaseTestCase()
    test.setUp()
    test.test_bench_GwGm_IPOPT_Breguet_Cruise()
