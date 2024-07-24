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
phase_info1 = deepcopy(phase_info)
phase_info2 = deepcopy(phase_info)

if __name__ == '__main__':
    super_prob = om.Problem.model()
    prob1 = av.AviaryProblem()
    prob2 = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    # TODO: modify one of these to represent a different payload case
    prob1.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info1)
    prob2.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info2)

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


"""
Ferry mission phase info:
Times (min):   0,    30,   810, 835
   Alt (ft):   0, 32000, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3


Hard to find multiple payload/range values for FwFm (737), so use C-5 instead
Based on: 
    https://en.wikipedia.org/wiki/Lockheed_C-5_Galaxy#Specifications_(C-5M), 
    https://www.af.mil/About-Us/Fact-Sheets/Display/Article/1529718/c-5-abc-galaxy-and-c-5m-super-galaxy/ 

MTOW: 840,000 lb
Max Payload: 281,000 lb
Max Fuel: 341,446 lb
Empty Weight: 380,000 lb -> leaves 460,000 lb for fuel+payload (max fuel + max payload = 622,446 lb)

Payload/range:
    281,000 lb payload -> 2,150 nmi range (AF.mil) [max payload case]
    120,000 lb payload -> 4,800 nmi range (AF.mil) [intermediate case]
          0 lb payload -> 7,000 nmi range (AF.mil) [ferry case]

Flight characteristics: 
    Cruise at M0.77 at 33k ft 
    Max rate of climb: 2100 ft/min
"""
