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
weights = [1, 1]
num_missions = len(weights)

if __name__ == '__main__':
    super_prob = om.Problem()
    probs = []

    # define individual aviary problems
    for i, (plane, phase_info) in enumerate(zip(planes, phase_infos)):
        prob = av.AviaryProblem()
        prob.load_inputs(plane, phase_info)
        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()
        probs.append(prob)

        subcomp = om.SubmodelComp(problem=prob,
                                  inputs=['mission:design:gross_mass'],
                                  outputs=['mission:summary:fuel_burned'])
        super_prob.model.add_subsystem(f'subcomp_{i}', subcomp)

    # creating variable strings that will represent fuel burn from each mission
    fuel_burned_vars = [f"fuel_{i}" for i in range(num_missions)]
    weighted_str = "+".join([f"{fuel}*{weight}"
                             for fuel, weight in zip(fuel_burned_vars, weights)])
    # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]

    # adding compound execComp to super problem
    super_prob.model.add_subsystem('compound', om.ExecComp(
        "compound = "+weighted_str), promotes=["compound", *fuel_burned_vars])

    # connecting each subcomponent's fuel burn to super problem's unique fuel variables
    for i in range(num_missions):
        super_prob.model.connect(f"subcomp_{i}.mission:summary:fuel_burned", f"fuel_{i}")

    # create an output within superprob that connects to each subcomponents gross mass input
    IVC = super_prob.model.add_subsystem('IVC', om.IndepVarComp(), promotes=['*'])
    IVC.add_output('gross_mass', val=500e3, units='lbm')

    # specify gross mass as a design var
    super_prob.model.add_design_var('gross_mass', lower=100e3, upper=1000e3, units='lbm')

    # connect gross mass output of super problem to each subcomponent's input gross mass
    for i in range(num_missions):
        super_prob.model.connect('gross_mass', f"subcomp_{i}.mission:design:gross_mass")

    super_prob.driver = om.ScipyOptimizeDriver()
    super_prob.driver.options['optimizer'] = 'SLSQP'

    super_prob.model.add_objective('compound')  # output from execcomp goes here

    super_prob.setup()

    # set initial guesses for each aviary problem
    for prob in probs:
        prob.set_initial_guesses()

    dm.run_problem(super_prob)


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
