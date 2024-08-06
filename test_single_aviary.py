import aviary.api as av
from easy_phase_info_max import phase_info as max_phase_info
from easy_phase_info_inter import phase_info as inter_phase_info
from aviary.variable_info.variables import Mission, Aircraft


outputs = {Mission.Summary.FUEL_BURNED: [],
           Aircraft.Design.EMPTY_MASS: []}
for plane, phase_info in zip(
    ['c5_maxpayload', 'c5_intermediate'],
        [max_phase_info, inter_phase_info]):

    prob = av.AviaryProblem()
    prob.load_inputs(f"c5_models/{plane}.csv", phase_info)
    prob.check_and_preprocess_inputs()
    prob.add_pre_mission_systems()
    prob.add_phases()
    prob.add_post_mission_systems()
    prob.link_phases()
    prob.add_driver('SLSQP')
    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()
    # prob.final_setup()
    prob.run_aviary_problem()

    for key in outputs.keys():
        outputs[key].append(prob.get_val(key)[0])

print("\n\n=============================\n")
for key, item in outputs.items():
    print(f"Variable: {key}")
    print(f"Values: {item}")

"""
Current output:
Variable: mission:summary:fuel_burned
Values: [140116.12301917037, 276762.7974282048]
Variable: aircraft:design:empty_mass
Values: [302105.1606792437, 315604.5402712884]
"""
