from copy import deepcopy
import pkg_resources
import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.interface.default_phase_info.height_energy import phase_info as ph_in_flops
from aviary.interface.default_phase_info.two_dof import phase_info as ph_in_gasp
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class StaticGroupTest(unittest.TestCase):

    def test_post_mission_promotion(self):
        phase_info = deepcopy(ph_in_flops)

        prob = AviaryProblem()

        csv_path = pkg_resources.resource_filename(
            "aviary", "models/test_aircraft/aircraft_for_bench_GwFm.csv")

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        # TODO: This needs to be converted into a reserve and a scaler so that it can
        # be given proper units.
        # The units here are lbm.
        prob.aviary_inputs.set_val(Aircraft.Design.RESERVES, 10000.0, units='unitless')

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.add_design_variables()
        prob.add_objective(objective_type="mass", ref=-1e5)

        prob.setup()

        prob.run_model()

        fuel_burned = prob.model.get_val('fuel_burned', units='lbm')
        total_fuel = prob.model.get_val(Mission.Summary.TOTAL_FUEL_MASS, units='lbm')

        assert_near_equal(total_fuel - fuel_burned, 10000.0, 1e-3)

    def test_gasp_relative_reserve(self):
        phase_info = deepcopy(ph_in_gasp)

        prob = AviaryProblem()

        csv_path = pkg_resources.resource_filename(
            "aviary", "models/small_single_aisle/small_single_aisle_GwGm.csv")

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        prob.aviary_inputs.set_val(Mission.Summary.GROSS_MASS, 140000.0, units='lbm')

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.add_design_variables()
        prob.add_objective(objective_type="mass", ref=-1e5)

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        res_frac = prob.aviary_inputs.get_val(Aircraft.Design.RESERVES, units='unitless')
        td_mass = prob.model.get_val(Mission.Landing.TOUCHDOWN_MASS, units='lbm')
        reserve = prob.model.get_val(Mission.Design.RESERVE_FUEL, units='lbm')
        assert_near_equal(reserve, -res_frac * (140000.0 - td_mass), 1e-3)


if __name__ == '__main__':
    unittest.main()
