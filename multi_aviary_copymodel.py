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
import numpy as np
from c5_models.c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_models.c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_models.c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info
from easy_phase_info_inter import phase_info as easy_inter
from easy_phase_info_max import phase_info as easy_max
from aviary.variable_info.variables import Mission, Aircraft
from aviary.variable_info.enums import ProblemType

# "comp?.a can be used to reference multiple comp1.a comp2.a etc"


class MultiMissionProblem(om.Problem):
    def __init__(self, planes, phase_infos, weights):
        super().__init__()
        self.num_missions = len(planes)
        # phase infos and planes length must match - this maybe unnecessary if
        # different planes (payloads) fly same mission (say pax vs cargo)
        # or if same payload flies 2 different missions (altitude/mach differences)
        if self.num_missions != len(phase_infos):
            raise Exception("Length of planes and phase_infos must be the same!")

        # if fewer weights than planes are provided, assign equal weights for all planes
        if len(weights) < self.num_missions:
            weights = [1]*self.num_missions
        # if more weights than planes, raise exception
        elif len(weights) > self.num_missions:
            raise Exception("Length of weights cannot exceed length of planes!")
        self.weights = weights

        self.group_prefix = 'group'
        self.probs = []
        # define individual aviary problems
        for i, (plane, phase_info) in enumerate(zip(planes, phase_infos)):
            prob = av.AviaryProblem()
            prob.load_inputs(plane, phase_info)
            prob.check_and_preprocess_inputs()
            prob.add_pre_mission_systems()
            prob.add_phases()
            prob.add_post_mission_systems()
            prob.link_phases()
            prob.problem_type = ProblemType.ALTERNATE
            prob.add_design_variables()  # should not work at super prob level
            self.probs.append(prob)

            self.model.add_subsystem(
                self.group_prefix + f'_{i}', prob.model,
                promotes=['mission:design:gross_mass', 'mission:design:range'])

    def add_design_variables(self):
        self.model.add_design_var('mission:design:gross_mass', lower=10., upper=900e3)

    def add_driver(self):
        self.driver = om.pyOptSparseDriver()

        self.driver.options["optimizer"] = "SNOPT"
        # driver.declare_coloring(True) # do we need this anymore of we're specifying the below?
        # maybe we're getting matrixes that are too sparse, decrease tolerance to avoid missing corellatiton
        # set coloring at this value. 1e-45 didn't seem to make much difference
        self.driver.declare_coloring(tol=1e-25, orders=None)
        self.driver.opt_settings["Major iterations limit"] = 60
        self.driver.opt_settings["Major optimality tolerance"] = 1e-6
        self.driver.opt_settings["Major feasibility tolerance"] = 1e-6
        self.driver.opt_settings["iSumm"] = 6
        self.driver.opt_settings['Verify level'] = -1
        # self.driver.options['maxiter'] = 1e3
        # self.driver.declare_coloring()
        # self.model.linear_solver = om.DirectSolver()
        """scipy SLSQP results
            Iteration limit reached    (Exit mode 9)
            Current function value: -43.71865402878029
            Iterations: 200
            Function evaluations: 1018
            Gradient evaluations: 200
            Optimization FAILED.
            Iteration limit reached"""

    def add_objective(self):
        weights = [float(weight/sum(self.weights)) for weight in self.weights]
        fuel_burned_vars = [f"fuel_{i}" for i in range(self.num_missions)]
        weighted_str = "+".join([f"{fuel}*{weight}"
                                for fuel, weight in zip(fuel_burned_vars, weights)])
        # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]

        # adding compound execComp to super problem
        self.model.add_subsystem('compound_fuel_burn_objective', om.ExecComp(
            "compound = "+weighted_str), promotes=["compound", *fuel_burned_vars])

        for i in range(self.num_missions):
            # connecting each subcomponent's fuel burn to super problem's unique fuel variables
            self.model.connect(
                self.group_prefix+f"_{i}.{Mission.Objectives.FUEL}", f"fuel_{i}")
        self.model.add_objective('compound', ref=1)

    def setup_wrapper(self):
        """Wrapper for om.Problem setup with warning ignoring and setting options"""
        for prob in self.probs:
            prob.model.options['aviary_options'] = prob.aviary_inputs
            prob.model.options['aviary_metadata'] = prob.meta_data
            prob.model.options['phase_info'] = prob.phase_info

        # Aviary's problem setup wrapper uses these ignored warnings to suppress
        # some warnings related to variable promotion. Replicating that here with
        # setup for the super problem
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", om.OpenMDAOWarning)
            warnings.simplefilter("ignore", om.PromotionWarning)
            self.setup(check='all')

    def run(self):
        # self.run_model()
        # self.check_totals(method='fd', compact_print=True)
        self.model.set_solver_print(0)

        # self.run_driver()
        dm.run_problem(self, make_plots=True)

    def get_design_range(self, phase_infos):
        design_range = 0
        for phase_info in phase_infos:
            get_range = phase_info['post_mission']['target_range'][0]  # TBD add units
            if get_range > design_range:
                design_range = get_range
        return design_range


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False
    planes = ['c5_models/c5_maxpayload.csv', 'c5_models/c5_intermediate.csv']
    # phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
    phase_infos = [easy_max, easy_inter]
    weights = [1, 1]
    super_prob = MultiMissionProblem(planes, phase_infos, weights)
    super_prob.add_driver()
    super_prob.add_design_variables()
    super_prob.add_objective()
    super_prob.model.set_input_defaults('mission:design:range', val=4000)
    super_prob.setup_wrapper()
    super_prob.set_val('mission:design:range', super_prob.get_design_range(phase_infos))
    for i, prob in enumerate(super_prob.probs):
        prob.set_initial_guesses(super_prob, super_prob.group_prefix+f"_{i}.")

    # super_prob.final_setup()
    if makeN2:
        from createN2 import createN2
        createN2(__file__, super_prob)
    super_prob.run()

    print("\n\n=========================\nEmpty masses:")
    print(super_prob.get_val(f'group_0.{Aircraft.Design.EMPTY_MASS}'))
    print(super_prob.get_val(f'group_1.{Aircraft.Design.EMPTY_MASS}'))
    print("Fuel burned")
    print(super_prob.get_val(f'group_0.{Mission.Summary.FUEL_BURNED}'))
    print(super_prob.get_val(f'group_1.{Mission.Summary.FUEL_BURNED}'))
    print("Summary Gross Mass")
    print(super_prob.get_val(f'group_0.{Mission.Summary.GROSS_MASS}'))
    print(super_prob.get_val(f'group_1.{Mission.Summary.GROSS_MASS}'))
    # super_prob.model.group_1.list_vars(units=True, print_arrays=True)
