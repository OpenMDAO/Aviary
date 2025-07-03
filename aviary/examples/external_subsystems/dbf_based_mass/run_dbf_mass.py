"""
Run the a mission with a DBF build-style that computes the mass of all components.
"""

from copy import deepcopy

import aviary.api as av
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variable_meta_data import (
    ExtendedMetaData,
)
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variables import Aircraft
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder

phase_info = deepcopy(av.default_height_energy_phase_info)
# Here we just add the simple weight system to only the pre-mission
phase_info['pre_mission']['external_subsystems'] = [DBFMassBuilder(meta_data=ExtendedMetaData)]

if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(
        'models/test_aircraft/aircraft_for_bench_FwFm.csv',
        # 'aviary/examples/external_subsystems/dbf_based_mass/dbf_aircraft_inputs.csv',
        phase_info,
        meta_data=ExtendedMetaData,
    )

    print(sorted(dir(prob)))

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

    print('Fuselage Mass', prob.get_val(Aircraft.Fuselage.MASS))
    print('Wing Mass', prob.get_val(Aircraft.Wing.MASS))
    print('Horizontal Tail Mass', prob.get_val(Aircraft.HorizontalTail.MASS))
    print('Vertical Tail Mass', prob.get_val(Aircraft.VerticalTail.MASS))

    print('done')
