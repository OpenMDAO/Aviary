"""Run the a mission with a simple external component that computes the wing and horizontal tail mass."""

from copy import deepcopy

import aviary.api as av
from aviary.examples.external_subsystems.custom_mass.custom_mass_builder import WingMassBuilder

phase_info = deepcopy(av.default_height_energy_phase_info)
# Here we just add the simple weight system to only the pre-mission
phase_info['pre_mission']['external_subsystems'] = [WingMassBuilder()]

if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver('SLSQP')

    prob.add_design_variables()

    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()

    prob.run_aviary_problem(suppress_solver_print=True)

    print('Engine Mass', prob.get_val(av.Aircraft.Engine.MASS))
    print('Wing Mass', prob.get_val(av.Aircraft.Wing.MASS))
    print('Horizontal Tail Mass', prob.get_val(av.Aircraft.HorizontalTail.MASS))

    print('done')
