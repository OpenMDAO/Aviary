"""
Test the conversion between phase info and phase builder to ensure
consistency and correctness.
"""

import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.default_phase_info.height_energy import phase_info as ph_in_height_energy
from aviary.interface.default_phase_info.height_energy import (
    phase_info_parameterization as phase_info_parameterization_height_energy,
)
from aviary.interface.default_phase_info.two_dof import phase_info as ph_in_two_dof
from aviary.interface.default_phase_info.two_dof import (
    phase_info_parameterization as phase_info_parameterization_two_dof,
)
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.mission.phase_builder_base import PhaseBuilderBase as PhaseBuilder
from aviary.mission.phase_builder_base import phase_info_to_builder
from aviary.variable_info.variables import Mission


@use_tempdirs
class TestParameterizePhaseInfo(unittest.TestCase):
    def test_phase_info_parameterization_two_dof(self):
        phase_info = deepcopy(ph_in_two_dof)

        prob = AviaryProblem()

        csv_path = 'models/small_single_aisle/small_single_aisle_GASP.csv'

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        # We can set some crazy vals, since we aren't going to optimize.
        prob.aviary_inputs.set_val(Mission.Design.RANGE, 5000, 'km')
        prob.aviary_inputs.set_val(Mission.Design.CRUISE_ALTITUDE, 31000, units='ft')
        prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 120000, 'lbm')
        prob.aviary_inputs.set_val(Mission.Design.MACH, 0.6, 'unitless')

        prob.add_pre_mission_systems()
        prob.add_phases(phase_info_parameterization=phase_info_parameterization_two_dof)
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        assert_near_equal(
            prob.get_val('traj.desc2.timeseries.input_values:distance', units='km')[-1], 5000.0
        )
        assert_near_equal(
            prob.get_val('traj.climb2.timeseries.input_values:altitude', units='ft')[-1], 31000.0
        )
        assert_near_equal(
            prob.get_val('traj.groundroll.timeseries.input_values:mass', units='lbm')[0], 120000.0
        )
        assert_near_equal(prob.get_val('traj.cruise.rhs.mach')[0], 0.6)

    def test_phase_info_parameterization_height_energy(self):
        phase_info = deepcopy(ph_in_height_energy)

        prob = AviaryProblem()

        csv_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        # We can set some crazy vals, since we aren't going to optimize.
        prob.aviary_inputs.set_val(Mission.Design.RANGE, 5000.0, 'km')
        prob.aviary_inputs.set_val(Mission.Design.CRUISE_ALTITUDE, 31000.0, units='ft')
        prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 195000.0, 'lbm')
        prob.aviary_inputs.set_val(Mission.Summary.CRUISE_MACH, 0.6, 'unitless')

        prob.add_pre_mission_systems()
        prob.add_phases(phase_info_parameterization=phase_info_parameterization_height_energy)
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        range_resid = prob.get_val(Mission.Constraints.RANGE_RESIDUAL, units='nmi')[-1]
        assert_near_equal(range_resid, 1906, tolerance=1e-3)
        assert_near_equal(prob.get_val('traj.cruise.timeseries.altitude', units='ft')[0], 31000.0)
        assert_near_equal(prob.get_val('traj.cruise.timeseries.mach')[0], 0.6)


# To run the tests
if __name__ == '__main__':
    unittest.main()
    # test = TestParameterizePhaseInfo()
    # test.test_phase_info_parameterization_two_dof()
