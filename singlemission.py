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

    prob.add_driver("SLSQP", max_iter=100)

    prob.add_design_variables()

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    prob.add_objective()  # output from execcomp goes here)

    prob.setup()

    prob.set_initial_guesses()

    # remove all plots and extras
    prob.run_aviary_problem(record_filename='c5_ferry.db')
    # prob.get_val()  # look at final fuel burn

"""
Ferry mission phase info:
Times (min):   0,    30,   810, 835
   Alt (ft):   0, 32000, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Notes: 32k in 30 mins too fast for aviary, 

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
