"""
Goal: use single aircraft description but optimize it for multiple missions simultaneously, 
i.e. all missions are on the range-payload line instead of having excess performance
Aircraft csv: defines plane, but also defines payload (passengers, cargo) which can vary with mission
    These will have to be specified in some alternate way such as a list correspond to mission #
Phase info: defines a particular mission, will have multiple phase infos
"""
import aviary.api as av
import openmdao.api as om
import dymos as dm
from c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info

planes = ['c5_maxpayload.csv', 'c5_intermediate.csv']
phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]

if __name__ == '__main__':
    super_prob = om.Problem()
    prob1 = av.AviaryProblem()
    prob2 = av.AviaryProblem()

    # Load aircraft and options data from user
    # Allow for user overrides here
    prob1.load_inputs(planes[0], phase_infos[0])
    prob2.load_inputs(planes[1], phase_infos[1])

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

    super_prob.model.add_subsystem('prob1', prob1)
    super_prob.model.add_subsystem('prob2', prob2)
    super_prob.model.add_subsystem('compound',
                                   om.ExecComp('compound = a + b'),
                                   promotes=['compound', 'a', 'b'])
    super_prob.model.connect('prob1.mission.design.gross_mass', 'a')
    super_prob.model.connect('prob2.mission.design.gross_mass', 'b')
    super_prob.model.connect('prob1.mission.design.gross_mass',
                             'prob2.mission.design.gross_mass')
    super_prob.add_driver("SLSQP", max_iter=100)

    # super_prob.add_parameter('mission.design.gross_mass')

    # use submodelcomp to instantiate both problems inside of the super problem
    # https://openmdao.org/newdocs/versions/latest/features/building_blocks/components/submodel_comp.html?highlight=subproblem
    # promote the correct inputs and outputs from the submodels
    # add an execComp that sums the fuel burn output

    # Load optimization problem formulation
    # Detail which variables the optimizer can control
    super_prob.model.add_objective('compound')  # output from execcomp goes here)

    super_prob.setup()

    prob1.set_initial_guesses()
    prob2.set_initial_guesses()

    # remove all plots and extras
    dm.run_problem(super_prob)
    # super_prob.run_aviary_problem(record_filename='reserve_mission_fixedrange.db')
    # super_prob.get_val()  # look at final fuel burn


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
