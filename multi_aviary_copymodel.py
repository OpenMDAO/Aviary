"""
authors: Jatin Soni, Eliot Aretskin
Multi Mission Optimization Example using Aviary
"""
import sys
import warnings
import dymos as dm
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import openmdao.api as om
import aviary.api as av
from aviary.variable_info.enums import ProblemType
from aviary.variable_info.variables import Mission, Aircraft

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
                promotes_inputs=[Mission.Design.GROSS_MASS,
                                 Mission.Design.RANGE,
                                 Aircraft.Wing.SPAN,
                                 Aircraft.Wing.AREA],
                promotes_outputs=[(Mission.Objectives.FUEL, promoted_name)])

    def add_design_variables(self):
        self.model.add_design_var(Mission.Design.GROSS_MASS,
                                  lower=10., upper=900e3, units='lbm')
        self.model.add_design_var(Aircraft.Wing.SPAN, lower=100., upper=500., units='ft')
        self.model.add_design_var(Aircraft.Wing.AREA, lower=10.,
                                  upper=1e6, units='ft**2')

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

    def create_timeseries_plots(self, plotvars=[], show=True):
        """Creates timeseries plots for any variables within timeseries. Specify variables
        and units by setting plotvars = [('altitude','ft')]. Any number of vars can be added."""
        plt.figure()
        for plotidx, (var, unit) in enumerate(plotvars):
            plt.subplot(int(np.ceil(len(plotvars)/2)), 2, plotidx+1)
            for i in range(self.num_missions):
                time = np.array([])
                yvar = np.array([])
                # this loop concatenates data from all phases
                for phase in self.phases[f"{self.group_prefix}_{i}"]:
                    rawt = self.get_val(
                        f"{self.group_prefix}_{i}.traj.{phase}.timeseries.time",
                        units='s')
                    rawy = self.get_val(
                        f"{self.group_prefix}_{i}.traj.{phase}.timeseries.{var}",
                        units=unit)
                    time = np.hstack([time, np.ndarray.flatten(rawt)])
                    yvar = np.hstack([yvar, np.ndarray.flatten(rawy)])
                plt.plot(time, yvar, linewidth=self.num_missions-i)
            plt.xlabel("Time (s)")
            plt.ylabel(f"{var.title()} ({unit})")
            plt.grid()
        plt.figlegend([f"Plane {i}" for i in range(self.num_missions)])
        if show:
            plt.show()

    def create_payload_range_plot(self, show=True):
        """Creates payload range diagram for the super problem. Appends a point for max payload
            and 0 range. """
        payloads = []
        ranges = []
        for i in range(self.num_missions):
            ref = f"{self.group_prefix}_{i}"
            payloads.append(
                self.get_val(
                    f"{ref}.{Aircraft.CrewPayload.CARGO_MASS}", units='lbm'))
            lastphase = self.phases[ref][-1]
            ranges.append(
                self.get_val(
                    f"{ref}.traj.{lastphase}.timeseries.distance",
                    units='nmi', indices=-1)[0])
        payloads, ranges = zip(*sorted(zip(payloads, ranges)))
        payloads, ranges = list(payloads), list(ranges)
        payloads.append(payloads[-1])
        ranges.append(0)
        plt.figure()
        plt.plot(ranges, payloads)
        plt.xlabel("Range (nmi)")
        plt.ylabel("Payload (lbm)")
        plt.grid()
        if show:
            plt.show()

    def print_vars(self, vars=[]):
        """Specify vars with name and unit in a tuple, e.g. vars = [ (Mission.Summary.FUEL_BURNED, 'lbm') ]"""

        print("\n\n=========================\n")
        print(f"{'':40}", end=': ')
        for i in range(self.num_missions):
            name = f"Mission {i}"
            print(f"{name:^30}", end='| ')
        print()
        for var, unit in vars:
            varname = f"Variable: {var.replace(':','.').upper()}"
            print(f"{varname:40}", end=": ")
            for i in range(self.num_missions):
                val = self.get_val(f'group_{i}.{var}', units=unit)[0]
                printstatement = f"{val} ({unit})"
                print(f"{printstatement:^30}", end="| ")
            print()


def C5_example(makeN2=False):
    plane_dir = 'c5_models'
    planes = ['c5_maxpayload.csv', 'c5_intermediate.csv', 'c5_ferry.csv']
    planes = [join(plane_dir, plane) for plane in planes]
    phase_infos = [c5_maxpayload_phase_info,
                   c5_intermediate_phase_info, c5_ferry_phase_info]
    optalt, optmach = False, False
    for phaseinfo in phase_infos:
        for key in phaseinfo.keys():
            if "user_options" in phaseinfo[key].keys():
                phaseinfo[key]["user_options"]["optimize_mach"] = optmach
                phaseinfo[key]["user_options"]["optimize_altitude"] = optalt

    weights = [1, 1, 1]

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
    printoutputs = [(Aircraft.Design.EMPTY_MASS, 'lbm'),
                    (Mission.Summary.FUEL_BURNED, 'lbm'),
                    (Mission.Summary.GROSS_MASS, 'lbm'),
                    (Aircraft.Wing.SPAN, 'ft'),
                    (Aircraft.Wing.AREA, 'ft**2')]
    super_prob.print_vars(vars=printoutputs)

    plotvars = [('altitude', 'ft'),
                ('mass', 'lbm'),
                ('drag', 'lbf'),
                ('distance', 'nmi'),
                ('throttle', 'unitless'),
                ('mach', 'unitless')]
    super_prob.create_timeseries_plots(plotvars=plotvars, show=False)

    super_prob.create_payload_range_plot(show=False)
    plt.show()

    return super_prob


if __name__ == '__main__':
    makeN2 = True if (len(sys.argv) > 1 and "n2" in sys.argv[1]) else False

    super_prob = C5_example(makeN2=makeN2)

"""
1:1:1 and 5:1:1 and 1:1:5 and 1:5:1
Variable: aircraft:design:empty_mass
        Plane 0: 377110.8519657982 (lbm),       Plane 1: 377110.8519657982 (lbm),       Plane 2: 377110.8519657982 (lbm),
Variable: mission:summary:fuel_burned
        Plane 0: 167530.46091225138 (lbm),      Plane 1: 308428.65141334233 (lbm),      Plane 2: 412689.6603555365 (lbm),
Variable: mission:summary:gross_mass
        Plane 0: 599904.1437916472 (lbm),       Plane 1: 711227.3342927382 (lbm),       Plane 2: 793323.6057604996 (lbm),
Variable: aircraft:wing:span
        Plane 0: 235.7734017415966 (ft),        Plane 1: 235.7734017415966 (ft),        Plane 2: 235.7734017415966 (ft),
Variable: aircraft:wing:area
        Plane 0: 6106.843788863166 (ft**2),     Plane 1: 6106.843788863166 (ft**2),     Plane 2: 6106.843788863166 (ft**2),

1:1:5

"""

"""
inter payload: 100k
Variable: aircraft:design:empty_mass
        Plane 0: 374179.5394606042 (lbm)
        Plane 1: 374179.5394606042 (lbm)
        Plane 2: 374179.5394606042 (lbm)
Variable: mission:summary:fuel_burned
        Plane 0: 165739.44323952997 (lbm)
        Plane 1: 306588.0148270305 (lbm)
        Plane 2: 411012.49583405076 (lbm)
Variable: mission:summary:gross_mass
        Plane 0: 595188.3345483424 (lbm)
        Plane 1: 702786.9061358428 (lbm)
        Plane 2: 788721.6496684301 (lbm)

inter payload: 140k
Variable: aircraft:design:empty_mass
        Plane 0: 374179.53946059936 (lbm)
        Plane 1: 374179.53946059936 (lbm)
        Plane 2: 374179.53946059936 (lbm)
Variable: mission:summary:fuel_burned
        Plane 0: 165739.44340813044 (lbm)
        Plane 1: 306588.0148359792 (lbm)
        Plane 2: 411012.4960230509 (lbm)
Variable: mission:summary:gross_mass
        Plane 0: 595188.3347169379 (lbm)
        Plane 1: 710136.9061447867 (lbm)
        Plane 2: 788721.6498574257 (lbm)

140k inter, 301k max
Variable: aircraft:design:empty_mass
        Plane 0: 374179.5394605969 (lbm)
        Plane 1: 374179.5394605969 (lbm)
        Plane 2: 374179.5394605969 (lbm)
Variable: mission:summary:fuel_burned
        Plane 0: 165739.44397328247 (lbm)
        Plane 1: 306588.01484213956 (lbm)
        Plane 2: 411012.4961383113 (lbm)
Variable: mission:summary:gross_mass
        Plane 0: 598863.3352823037 (lbm)
        Plane 1: 710136.9061509445 (lbm)
        Plane 2: 788721.6499726834 (lbm)

140k inter, 301k max, 10k ferry
Variable: aircraft:design:empty_mass
        Plane 0: 374179.5394605957 (lbm)
        Plane 1: 374179.5394605957 (lbm)
        Plane 2: 374179.5394605957 (lbm)
Variable: mission:summary:fuel_burned
        Plane 0: 165739.4440247023 (lbm)
        Plane 1: 306588.01484336343 (lbm)
        Plane 2: 411012.49619538186 (lbm)
Variable: mission:summary:gross_mass
        Plane 0: 598863.3353337225 (lbm)
        Plane 1: 710136.9061521672 (lbm)
        Plane 2: 790586.3875041857 (lbm)
"""

"""
optimize mach only:                                 optimize both mach and alt:                         optimize neither mach or altitude
Variable: aircraft:design:empty_mass                Variable: aircraft:design:empty_mass                Variable: aircraft:design:empty_mass
        Plane 0: 326047.42156711605 (lbm)                 Plane 0: 390475.71469928196 (lbm)                 Plane 0: 371320.23571631836 (lbm)
        Plane 1: 326047.42156711605 (lbm)                 Plane 1: 390475.71469928196 (lbm)                 Plane 1: 371320.23571631836 (lbm)
        Plane 2: 326047.42156711605 (lbm)                 Plane 2: 390475.71469928196 (lbm)                 Plane 2: 371320.23571631836 (lbm)
Variable: mission:summary:fuel_burned               Variable: mission:summary:fuel_burned               Variable: mission:summary:fuel_burned
        Plane 0: 175717.28062873823 (lbm)                 Plane 0: 178314.99288207013 (lbm)                 Plane 0: 167946.2102766974 (lbm)
        Plane 1: 306419.71885746776 (lbm)                 Plane 1: 325789.9474922766 (lbm)                  Plane 1: 309094.299337939 (lbm)
        Plane 2: 394791.91474366543 (lbm)                 Plane 2: 429716.0286375711 (lbm)                  Plane 2: 413746.64501453174 (lbm)
Variable: mission:summary:gross_mass                Variable: mission:summary:gross_mass                Variable: mission:summary:gross_mass
        Plane 0: 557034.0540440837 (lbm)                  Plane 0: 624060.059429561 (lbm)                   Plane 0: 594535.797841224 (lbm)
        Plane 1: 658161.4922728132 (lbm)                  Plane 1: 741960.0140397676 (lbm)                  Plane 1: 706108.8869024655 (lbm)
        Plane 2: 724368.9506845778 (lbm)                  Plane 2: 823721.3577106291 (lbm)                  Plane 2: 788596.4951046254 (lbm)
Variable: aircraft:wing:span                        Variable: aircraft:wing:span                        Variable: aircraft:wing:span
        Plane 0: 54.707828078624665 (ft)                  Plane 0: 287.14946969690783 (ft)                  Plane 0: 211.8281919608065 (ft)
        Plane 1: 54.707828078624665 (ft)                  Plane 1: 287.14946969690783 (ft)                  Plane 1: 211.8281919608065 (ft)
        Plane 2: 54.707828078624665 (ft)                  Plane 2: 287.14946969690783 (ft)                  Plane 2: 211.8281919608065 (ft)
Variable: aircraft:wing:aspect_ratio                Variable: aircraft:wing:aspect_ratio                Variable: aircraft:wing:aspect_ratio
        Plane 0: 6.263196240285538 (unitless)             Plane 0: 6.931291416800268 (unitless)             Plane 0: 7.576574965501183 (unitless)
        Plane 1: 6.263196240285538 (unitless)             Plane 1: 6.931291416800268 (unitless)             Plane 1: 7.576574965501183 (unitless)
        Plane 2: 6.263196240285538 (unitless)             Plane 2: 6.931291416800268 (unitless)             Plane 2: 7.576574965501183 (unitless)


"""
