from c5_models.c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_models.c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_models.c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info
import aviary.api as av
import openmdao.api as om
import sys


def modify_plane(orig_filename, payloads, ranges):
    if len(sys.argv) > 1:
        plane_name = orig_filename.split(".csv")[0]
        mission = sys.argv[1].lower()
        temp_filename = f"{plane_name}_{mission}.csv"
        with open(orig_filename, "r") as orig_plane:
            orig_lines = orig_plane.readlines()
        with open(temp_filename, "w") as temp_plane:
            try:
                payload, misrange = payloads[mission], ranges[mission]
            except KeyError:
                payload, misrange = "", ""
            for line in orig_lines:
                if "aircraft:crew_and_payload:cargo_mass" in line and payload != "":
                    line = ",".join(
                        [line.split(",")[0],
                         str(payload),
                         line.split(",")[2]])

                elif "mission:design:range" in line and misrange != "":
                    line = ",".join(
                        [line.split(",")[0],
                         str(misrange),
                         line.split(",")[2]])

                temp_plane.write(line)
    else:
        return orig_filename
    return temp_filename, mission


if __name__ == '__main__':
    makeN2 = True if len(sys.argv) > 2 and "n2" in sys.argv else False
    prob = av.AviaryProblem()
    plane_file = 'c5.csv'
    payloads = {"ferry": 0, "intermediate": 120e3, "maxpayload": 281e3}
    ranges = {"ferry": 7e3, "intermediate": 4.8e3, "maxpayload": 2.15e3}
    plane_file, mission_name = modify_plane(plane_file, payloads, ranges)
    phase_info = c5_maxpayload_phase_info
    if mission_name == "intermediate":
        phase_info = c5_intermediate_phase_info
    elif mission_name == "ferry":
        phase_info = c5_ferry_phase_info

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob.load_inputs(plane_file, phase_info)

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

    if makeN2:
        om.n2(prob, outfile='single_aviary.html')

    prob.set_initial_guesses()

    # remove all plots and extras
    prob.run_aviary_problem(
        record_filename=f'{plane_file.split(".csv")[0]}.db', suppress_solver_print=True)
    # prob.get_val()  # look at final fuel burn
    print(
        f"Fuel burned: {prob.get_val(av.Mission.Summary.FUEL_BURNED,units='lbm')[0]:.3f} lbm")

"""
Ferry mission phase info:
Times (min):   0,    50,   812, 843
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 7001 nmi
Notes: 32k in 30 mins too fast for aviary, climb to low alt then slow rise 

Intermediate mission phase info:
Times (min):   0,    50,   560, 590
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 4839 nmi

Max Payload mission phase info:
Times (min):   0,    50,   260, 290
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 2272 nmi

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
