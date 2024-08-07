"""
authors: Jatin Soni, Eliot Aretskin
Multi Mission Optimization Example using Aviary
"""
from os.path import join
import sys
import warnings
import dymos as dm
import numpy as np
import matplotlib.pyplot as plt

import aviary.api as av
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.variables import Mission, Aircraft

import openmdao.api as om
from openmdao.api import CaseReader

from c5_models.c5_ferry_phase_info import phase_info as c5_ferry_phase_info
from c5_models.c5_intermediate_phase_info import phase_info as c5_intermediate_phase_info
from c5_models.c5_maxpayload_phase_info import phase_info as c5_maxpayload_phase_info


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
        self.phase_infos = phase_infos

        self.group_prefix = 'group'
        self.probs = []
        self.fuel_vars = []
        self.phases = {}
        # define individual aviary problems
        for i, (plane, phase_info) in enumerate(zip(planes, phase_infos)):
            prob = av.AviaryProblem()
            prob.load_inputs(plane, phase_info)
            prob.check_and_preprocess_inputs()
            prob.add_pre_mission_systems()
            prob.add_phases()
            prob.add_post_mission_systems()
            prob.link_phases()

            # alternate prevents use of equality constraint b/w design and summary gross mass
            prob.problem_type = ProblemType.ALTERNATE
            prob.add_design_variables()
            self.probs.append(prob)
            # phase names for each traj (can be used later to make plots/print outputs)
            self.phases[f"{self.group_prefix}_{i}"] = list(prob.traj._phases.keys())

            # design range and gross mass are promoted, these are Max Range/Max Takeoff Mass
            # and must be the same for each aviary problem. Subsystems within aviary are sized
            # using these - empty mass is same across all aviary problems.
            # the fuel objective is also promoted since that's used in the compound objective
            promoted_name = f"{self.group_prefix}_{i}_fuelobj"
            self.fuel_vars.append(promoted_name)
            self.model.add_subsystem(
                self.group_prefix + f'_{i}', prob.model,
                promotes=[Mission.Design.GROSS_MASS,
                          Mission.Design.RANGE,
                          (Mission.Objectives.FUEL, promoted_name)])

    def add_design_variables(self):
        self.model.add_design_var('mission:design:gross_mass', lower=10., upper=900e3)

    def add_driver(self):
        self.driver = om.pyOptSparseDriver()
        self.driver.options["optimizer"] = "SLSQP"
        self.driver.declare_coloring()
        # linear solver causes nan entry error for landing to takeoff mass ratio param
        # self.model.linear_solver = om.DirectSolver()

    def add_objective(self):
        # weights are normalized - e.g. for given weights 3:1, the normalized
        # weights are 0.75:0.25
        weights = [float(weight/sum(self.weights)) for weight in self.weights]
        weighted_str = "+".join([f"{fuelobj}*{weight}"
                                for fuelobj, weight in zip(self.fuel_vars, weights)])
        # weighted_str looks like: fuel_0 * weight[0] + fuel_1 * weight[1]

        # adding compound execComp to super problem
        self.model.add_subsystem('compound_fuel_burn_objective', om.ExecComp(
            "compound = "+weighted_str), promotes=["compound"])
        self.model.add_objective('compound')

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
        self.model.set_solver_print(0)
        dm.run_problem(self, make_plots=True)

    def get_design_range(self):
        """Finds the longest mission and sets its range as the design range for all
            Aviary problems. Used within Aviary for sizing subsystems."""
        design_range = 0
        for phase_info in self.phase_infos:
            get_range = phase_info['post_mission']['target_range'][0]  # TBD add units
            if get_range > design_range:
                design_range = get_range
        return design_range


def C5_example():
    plane_dir = 'c5_models'
    planes = ['c5_maxpayload.csv', 'c5_intermediate.csv']
    planes = [join(plane_dir, plane) for plane in planes]
    phase_infos = [c5_maxpayload_phase_info, c5_intermediate_phase_info]
    weights = [1, 1]

    super_prob = MultiMissionProblem(planes, phase_infos, weights)
    super_prob.add_driver()
    super_prob.add_design_variables()
    super_prob.add_objective()
    # set input default to prevent error, value doesn't matter since set val is used later
    super_prob.model.set_input_defaults(Mission.Design.RANGE, val=1.)
    super_prob.setup_wrapper()
    super_prob.set_val(Mission.Design.RANGE, super_prob.get_design_range())

    for i, prob in enumerate(super_prob.probs):
        prob.set_initial_guesses(super_prob, super_prob.group_prefix+f"_{i}.")

    if makeN2:
        from createN2 import createN2
        createN2(__file__, super_prob)

    super_prob.run()
    return super_prob


def createTimeseriesPlots(super_prob, plotvars):
    for plotidx, (var, unit) in enumerate(plotvars):
        plt.subplot(int(np.ceil(len(plotvars)/2)), 2, plotidx+1)
        for i in range(super_prob.num_missions):
            time = np.array([])
            yvar = np.array([])
            for phase in super_prob.phases[f"{super_prob.group_prefix}_{i}"]:
                rawt = super_prob.get_val(
                    f"{super_prob.group_prefix}_{i}.traj.{phase}.timeseries.time")
                rawy = super_prob.get_val(
                    f"{super_prob.group_prefix}_{i}.traj.{phase}.timeseries.{var}",
                    units=unit)
                time = np.hstack([time, np.ndarray.flatten(rawt)])
                yvar = np.hstack([yvar, np.ndarray.flatten(rawy)])
            plt.plot(time, yvar, 'o')
        plt.xlabel("Time (s)")
        plt.ylabel(f"{var.title()} ({unit})")
        plt.grid()
    plt.figlegend([f"Plane {i}" for i in range(super_prob.num_missions)])
    plt.show()


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False

    super_prob = C5_example()
    plotvars = [('altitude', 'ft'),
                ('mass', 'lbm'),
                ('drag', 'lbf'),
                ('distance', 'nmi'),
                ('throttle', 'unitless')]
    createTimeseriesPlots(super_prob, plotvars)

    outputs = {Aircraft.Design.EMPTY_MASS: [],
               Mission.Summary.FUEL_BURNED: [],
               Mission.Summary.GROSS_MASS: []}

    print("\n\n=========================\n")
    for key in outputs.keys():
        print(f"Variable: {key}")
        for i in range(super_prob.num_missions):
            val = super_prob.get_val(f'group_{i}.{key}', units='lbm')[0]
            print(f"\tPlane {i}: {val} (lbm)")


"""
1:1
Variable: mission:summary:fuel_burned
Values: 164988.61664962117, 306345.04737967893 (lbm)
Variable: aircraft:design:empty_mass
Values: 378204.91862045845, 378204.91862045845 (lbm)
Variable: mission:summary:gross_mass
Values: 598462.8871182877, 710244.3178483456 (lbm)

1.5:1
Variable: mission:summary:fuel_burned
Values: 164988.61476287164, 306345.04738991836 (lbm)
Variable: aircraft:design:empty_mass
Values: 378204.91862045845, 378204.91862045845 (lbm)
Variable: mission:summary:gross_mass
Values: 598462.8852315382, 710244.3178585849 (lbm)

2:1
Variable: mission:summary:fuel_burned
Values: 164988.61651039496, 306345.04738988867 (lbm)
Variable: aircraft:design:empty_mass
Values: 378204.91862045845, 378204.91862045845 (lbm)
Variable: mission:summary:gross_mass
Values: 598462.8869790616, 710244.3178585552 (lbm)
"""
