import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.default_phase_info.two_dof import phase_info as ph_in_gasp
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.test_aircraft.GwFm_phase_info import phase_info as ph_in_flops
from aviary.variable_info.variables import Aircraft, Mission


# NOTE this test is probably in the wrong place, it isn't really testing a part of the
#      interface. Also test name is `PreMissionGroupTest` which is completely inaccurate.
#      This test is for checking if reserve missions are being properly applied, which
#      it only partially does (should be checking that the reserve mission properly
#      exists in traj as well)
@use_tempdirs
class PreMissionGroupTest(unittest.TestCase):
    def test_post_mission_promotion(self):
        phase_info = deepcopy(ph_in_flops)

        prob = AviaryProblem()

        csv_path = 'models/test_aircraft/aircraft_for_bench_GwFm.csv'

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        prob.aviary_inputs.set_val(Aircraft.Design.RESERVE_FUEL_ADDITIONAL, 10000.0, units='lbm')

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.add_design_variables()
        prob.add_objective(objective_type='mass', ref=-1e5)

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        fuel_burned = prob.model.get_val(Mission.Summary.FUEL_BURNED, units='lbm')
        total_fuel = prob.model.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm')

        assert_near_equal(total_fuel - fuel_burned, 10000.0, 1e-3)

    def test_gasp_relative_reserve(self):
        phase_info = deepcopy(ph_in_gasp)

        prob = AviaryProblem()

        csv_path = 'models/small_single_aisle/small_single_aisle_GASP.csv'

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        prob.aviary_inputs.set_val(Mission.Summary.GROSS_MASS, 140000.0, units='lbm')

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.add_design_variables()
        prob.add_objective(objective_type='mass', ref=-1e5)

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        res_frac = prob.aviary_inputs.get_val(
            Aircraft.Design.RESERVE_FUEL_FRACTION, units='unitless'
        )
        td_mass = prob.model.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm')
        reserve = prob.model.get_val(Mission.Design.RESERVE_FUEL, units='lbm')
        assert_near_equal(reserve, res_frac * (140000.0 - td_mass), 1e-3)


if __name__ == '__main__':
    unittest.main()
