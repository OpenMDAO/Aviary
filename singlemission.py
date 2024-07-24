import aviary.api as av
from c5_ferry_phase_info import phase_info

if __name__ == '__main__':
    prob = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs('c5.csv', phase_info)

    # Preprocess inputs
    prob.check_and_preprocess_inputs()

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver("SLSQP", max_iter=50)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective()  # output from execcomp goes here)

    prob.setup()

    prob.set_initial_guesses()

    # remove all plots and extras
    prob.run_aviary_problem(record_filename='c5_ferry.db')
    # prob.get_val()  # look at final fuel burn
