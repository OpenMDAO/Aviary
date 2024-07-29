"""
Goal: use single aircraft description but optimize it for multiple missions simultaneously,
i.e. all missions are on the range-payload line instead of having excess performance
Aircraft csv: defines plane, but also defines payload (passengers, cargo) which can vary with mission
    These will have to be specified in some alternate way such as a list correspond to mission #
Phase info: defines a particular mission, will have multiple phase infos
"""
import sys
import warnings
import aviary.api as av
import openmdao.api as om
import dymos as dm
from aviary.variable_info.enums import ProblemType
from c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info
from aviary.variable_info.variable_meta_data import _MetaData as MetaData

planes = ['c5_maxpayload.csv', 'c5_intermediate.csv']
phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
weights = [1, 1]
num_missions = len(weights)

# "comp?.a can be used to reference multiple comp1.a comp2.a etc"


class MultiMissionProblem(om.Problem):
    def __init__(self):
        super().__init__()
        self.model.add_subsystem()

    def add_design_variable():
        pass

    def link_pre_post_traj():
        pass

    def set_initial_values():
        pass


def add_design_var():
    pass


def setupprob(super_prob):
    # Aviary's problem setup wrapper uses these ignored warnings to suppress
    # some warnings related to variable promotion. Replicating that here with
    # setup for the super problem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", om.OpenMDAOWarning)
        warnings.simplefilter("ignore", om.PromotionWarning)
        super_prob.setup()


if __name__ == '__main__':
    super_prob = om.Problem()
    probs = []
    prefix = "problem_"

    makeN2 = False
    if len(sys.argv) > 1:
        if "n2" in sys.argv:
            makeN2 = True

    # define individual aviary problems
    for i, (plane, phase_info) in enumerate(zip(planes, phase_infos)):
        prob = av.AviaryProblem()
        prob.load_inputs(plane, phase_info)
        prob.check_and_preprocess_inputs()
        prob.add_pre_mission_systems()
        traj = prob.add_phases()  # save dymos traj to add to super problem as a subsystem
        prob.add_post_mission_systems()
        prob.link_phases()  # this is half working / connect statements from outside of traj to inside are failing
        prob.problem_type = ProblemType.ALTERNATE  # adds summary gross mass as design var
        prob.add_design_variables()
        probs.append(prob)

        group = om.Group()  # this group will contain all the promoted aviary vars
        group.add_subsystem("pre", prob.pre_mission)
        group.add_subsystem("traj", traj)
        group.add_subsystem("post", prob.post_mission)

        # setting defaults for these variables to suppress errors
        longlst = [
            'mission:summary:gross_mass', 'aircraft:wing:sweep',
            'aircraft:wing:thickness_to_chord', 'aircraft:wing:area',
            'aircraft:wing:taper_ratio', 'mission:design:gross_mass']
        for var in longlst:
            group.set_input_defaults(
                var, val=MetaData[var]['default_value'],
                units=MetaData[var]['units'])

        # add group and promote design gross mass (common input amongst multiple missions)
        # in this way it represents the MTOW
        super_prob.model.add_subsystem(prefix+f'{i}', group, promotes=[
                                       'mission:design:gross_mass'])

    # add design gross mass as a design var
    super_prob.model.add_design_var(
        'mission:design:gross_mass', lower=100e3, upper=1000e3)

    for i in range(num_missions):
        # connecting each subcomponent's fuel burn to super problem's unique fuel variables
        super_prob.model.connect(
            prefix+f"{i}.mission:summary:fuel_burned", f"fuel_{i}")

        # create constraint to force each mission's summary gross mass to not
        # exceed the common mission design gross mass (aka MTOW)
        super_prob.model.add_subsystem(f'MTOW_constraint{i}', om.ExecComp(
            'mtow_resid = design_gross_mass - summary_gross_mass'),
            promotes=[('summary_gross_mass', prefix+f'{i}.mission:summary:gross_mass'),
                      ('design_gross_mass', 'mission:design:gross_mass')])

        super_prob.model.add_constraint(f'MTOW_constraint{i}.mtow_resid', lower=0.)

    # creating variable strings that will represent fuel burn from each mission
    fuel_burned_vars = [f"fuel_{i}" for i in range(num_missions)]
    weighted_str = "+".join([f"{fuel}*{weight}"
                             for fuel, weight in zip(fuel_burned_vars, weights)])
    # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]

    # adding compound execComp to super problem
    super_prob.model.add_subsystem('compound_fuel_burn_objective', om.ExecComp(
        "compound = "+weighted_str), promotes=["compound", *fuel_burned_vars])

    super_prob.driver = om.ScipyOptimizeDriver()
    super_prob.driver.options['optimizer'] = 'SLSQP'
    super_prob.model.add_objective('compound')  # output from execcomp goes here

    setupprob(super_prob)
    if makeN2:
        om.n2(super_prob, outfile="multi_mission_importTraj_N2.html")  # create N2 diagram

    # cannot use this b/c initial guesses (i.e. setval func) has to be called on super prob level
    # for prob in probs:
    #     # prob.setup()
    #     prob.set_initial_guesses()

    # dm.run_problem(super_prob)


"""
Ferry mission phase info:
Times (min):   0,    50,   812, 843
   Alt (ft):   0, 29500, 32000,   0
       Mach: 0.3,  0.77,  0.77, 0.3
Est. Range: 7001 nmi
Notes: 32k in 30 mins too fast for aviary, climb to low alt then slow rise through cruise

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
