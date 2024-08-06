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
import matplotlib.pyplot as plt
from c5_models.c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_models.c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_models.c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info
from easy_phase_info_inter import phase_info as easy_inter
from easy_phase_info_max import phase_info as easy_max
from aviary.variable_info.variables import Mission, Aircraft
from aviary.variable_info.enums import ProblemType
from openmdao.api import CaseReader


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
                promotes=[Mission.Design.GROSS_MASS, Mission.Design.RANGE])

    def add_design_variables(self):
        self.model.add_design_var('mission:design:gross_mass', lower=10., upper=900e3)

    def add_driver(self):
        # pyoptsparse SLSQP errors out w pos directional derivative line search (obj scaler = 1) and
        # inequality constraints incompatible (obj scaler = -1) - fuel burn obj
        # pyoptsparse IPOPT keeps iterating (seen upto 1000+ iters) in the IPOPT.out file but no result
        # scipy SLSQP reaches iter limit and fails optimization
        self.driver = om.pyOptSparseDriver()
        self.driver.options['optimizer'] = 'SLSQP'
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
        dm.run_problem(self, solution_record_file='res.db')


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False
    planes = ['c5_models/c5_maxpayload.csv', 'c5_models/c5_intermediate.csv']
    # phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
    phase_infos = [easy_max, easy_inter]
    phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
    weights = [1, 1]
    super_prob = MultiMissionProblem(planes, phase_infos, weights)
    super_prob.add_driver()
    super_prob.add_design_variables()
    super_prob.add_objective()
    super_prob.setup_wrapper()
    for i, prob in enumerate(super_prob.probs):
        super_prob.set_val(
            super_prob.group_prefix +
            f"_{i}.aircraft:design:landing_to_takeoff_mass_ratio", 0.5)
        prob.set_initial_guesses(super_prob, super_prob.group_prefix+f"_{i}.")
        print(super_prob.get_val(super_prob.group_prefix +
                                 f"_{i}.aircraft:design:landing_to_takeoff_mass_ratio"))
        print(super_prob.get_val(super_prob.group_prefix +
                                 f"_{i}.mission:summary:range"))
    # super_prob.final_setup()
    if makeN2:
        from createN2 import createN2
        createN2(__file__, super_prob)
    super_prob.run()

    outputs = {Mission.Summary.FUEL_BURNED: [],
               Aircraft.Design.EMPTY_MASS: [],
               Mission.Summary.GROSS_MASS: []}

    print("\n\n=========================\n")
    for key in outputs.keys():
        val1 = super_prob.get_val(f'group_0.{key}', units='lbm')[0]
        val2 = super_prob.get_val(f'group_1.{key}', units='lbm')[0]
        print(f"Variable: {key}")
        print(f"Values: {val1}, {val2} (lbm)")

    comps = ['vertical_tail', 'horizontal_tail', 'wing', 'fuselage']
    print("\nmass comparisons")
    for comp in comps:
        v1 = super_prob.get_val(f'group_0.aircraft:{comp}:mass')
        v2 = super_prob.get_val(f'group_1.aircraft:{comp}:mass')
        print(f'{comp}: {v1} vs. {v2}')

    sol = CaseReader('res.db').get_case('final')
    # super_prob.model.list_vars()
    # for i in range(2):
    var = 'throttle'
    t = np.concatenate([sol.get_val('group_0.traj.climb_1.t'),
                        sol.get_val('group_0.traj.climb_2.t'),
                        sol.get_val('group_0.traj.descent_1.t')])
    alt = np.concatenate([sol.get_val(f'group_0.traj.climb_1.timeseries.{var}'),
                          sol.get_val(f'group_0.traj.climb_2.timeseries.{var}'),
                          sol.get_val(f'group_0.traj.descent_1.timeseries.{var}')])
    t2 = np.concatenate([sol.get_val('group_1.traj.climb_1.t'),
                        sol.get_val('group_1.traj.climb_2.t'),
                        sol.get_val('group_1.traj.descent_1.t')])
    alt2 = np.concatenate([sol.get_val(f'group_1.traj.climb_1.timeseries.{var}'),
                          sol.get_val(f'group_1.traj.climb_2.timeseries.{var}'),
                          sol.get_val(f'group_1.traj.descent_1.timeseries.{var}')])
    plt.plot(t, alt, 'r*')
    plt.plot(t2, alt2, 'b*')
    plt.title(f"Time vs {var}")
    plt.legend(['group_0', 'group_1'])
    plt.grid()
    plt.show()

"""
Variable: mission:summary:fuel_burned
Values: 13730.910584707046, 15740.545749454643
Variable: aircraft:design:empty_mass
Values: 336859.7179064408, 337047.85745526763
"""
