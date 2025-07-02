"""
Run the a mission with a simple external component that computes the wing
and horizontal tail mass.
"""

from copy import deepcopy

import aviary.api as av
from aviary.subsystems.mass.simple_mass.mass_builder import MassBuilderBase
from aviary.variable_info.variables import Aircraft

import jax.numpy as jnp

phase_info = deepcopy(av.default_height_energy_phase_info)
# Here we just add the simple weight system to only the pre-mission
phase_info['pre_mission']['external_subsystems'] = [MassBuilderBase()]

if __name__ == '__main__':
    prob = av.AviaryProblem()

    n_points = 10 # = num_sections
    x = jnp.linspace(0, 1, n_points)
    max_thickness_chord_ratio = 0.12
    thickness_dist = 5 * max_thickness_chord_ratio * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info)
    prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, val=1.0, units='m')
    prob.aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, val=1.0, units='m')
    prob.aviary_inputs.set_val(Aircraft.Wing.SPAN, 3.74904, units='m')  
    prob.aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 0.40005, units='m')  
    prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 2.5, units='m')


    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver('IPOPT')

    prob.add_design_variables()

    prob.add_objective()

    prob.setup()

    prob.set_val('pre_mission.simple_mass.tip_chord_tail', 0.5)
    prob.set_val('pre_mission.simple_mass.thickness_ratio', 0.12)
    prob.set_val('pre_mission.simple_mass.skin_thickness', 0.002)
    prob.set_val('pre_mission.simple_mass.tip_chord', 0.100076)
    prob.set_val('pre_mission.simple_mass.thickness_dist', thickness_dist)
    prob.set_val('pre_mission.simple_mass.base_diameter', 0.5)
    prob.set_val('pre_mission.simple_mass.tip_diameter', 0.3)
    prob.set_val('pre_mission.simple_mass.curvature', 0.0)
    prob.set_val('pre_mission.simple_mass.thickness', 0.05)

    prob.set_initial_guesses()

    prob.run_aviary_problem(suppress_solver_print=True)

    #prob.model.list_vars(units=True, print_arrays=True)

    #print('Engine Mass', prob.get_val(av.Aircraft.Engine.MASS))
    print('Wing Mass', prob.get_val(av.Aircraft.Wing.MASS))
    print('Horizontal Tail Mass', prob.get_val(av.Aircraft.HorizontalTail.MASS))
    print('Fuselage Mass', prob.get_val(av.Aircraft.Fuselage.MASS))

    print('done')
