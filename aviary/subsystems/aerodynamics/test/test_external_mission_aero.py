import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

import aviary.api as av
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder

phase_info = deepcopy(av.default_height_energy_phase_info)


@use_tempdirs
class TestExternalAero(av.TestSubsystemBuilderBase):
    """
    Test replacing internal drag calculation with an external subsystem.

    Mainly, this shows that the "external" method works, and that the external
    subsystems in mission are correctly promoting inputs/outputs.
    """

    @require_pyoptsparse(optimizer='IPOPT')
    def test_external_drag(self):
        # Just do cruise in this example.
        phase_info.pop('climb')
        phase_info.pop('descent')

        # Add custom aero.
        # TODO: This API for replacing aero will be changed an upcoming release.
        phase_info['cruise']['external_subsystems'] = [CustomAeroBuilder()]

        # Disable internal aero
        # TODO: This API for replacing aero will be changed an upcoming release.
        phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {
            'method': 'external',
        }

        prob = av.AviaryProblem()

        # Load aircraft and options data from user
        prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        # SLSQP didn't work so well here.
        prob.add_driver('IPOPT')

        prob.add_design_variables()
        prob.add_objective()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_aviary_problem(suppress_solver_print=True)

        drag = prob.get_val('traj.cruise.rhs_all.drag', units='lbf')
        assert_near_equal(drag[0], 7272.0265, tolerance=1e-3)


if __name__ == '__main__':
    unittest.main()
