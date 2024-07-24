"""
Goal: use single aircraft description but optimize it for multiple missions simultaneously, 
i.e. all missions are on the range-payload line instead of having excess performance
Aircraft csv: defines plane, but also defines payload (passengers, cargo) which can vary with mission
    These will have to be specified in some alternate way such as a list correspond to mission #
Phase info: defines a particular mission, will have multiple phase infos
"""
import aviary.api as av
from aviary.examples.example_phase_info import phase_info
from copy import deepcopy
import openmdao.api as om

# TODO: modify one of these to represent a different mission (e.g. change cruise length)
phase_info1 = phase_info
phase_info2 = phase_info

if __name__ == '__main__':
    super_prob = om.Problem.model()
    prob1 = av.AviaryProblem()
    prob2 = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob1.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info1)
    prob2.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info1)

    # Preprocess inputs
    prob1.check_and_preprocess_inputs()
    prob2.check_and_preprocess_inputs()

    prob1.add_pre_mission_systems()
    prob2.add_pre_mission_systems()

    prob1.add_phases()
    prob2.add_phases()

    prob1.add_post_mission_systems()
    prob2.add_post_mission_systems()

    # Link phases and variables
    prob1.link_phases()
    prob2.link_phases()

    super_prob.add_driver("SLSQP", max_iter=100)

    super_prob.add_parameter('mission.design.gross_mass')

    # use submodelcomp to instantiate both problems inside of the super problem
    # https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/submodel_comp.html?highlight=subproblem
    # promote the correct inputs and outputs from the submodels
    # add and execComp that sums the fuel burn output

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    super_prob.add_objective()  # output from execcomp goes here)

    super_prob.setup()

    prob1.set_initial_guesses()
    prob2.set_initial_guesses()

    # remove all plots and extras
    super_prob.run_aviary_problem(record_filename='reserve_mission_fixedrange.db')
    super_prob.get_val()  # look at final fuel burn
