import openmdao.api as om
import dymos as dm
from plot_helper import make_min_time_climb_plot
from min_time_climb import MinTimeClimbProblem
import matplotlib.pyplot as plt
import sys


class MultiMinTime(om.Problem):
    def __init__(
            self, heights, weights, optWing=False, modelinfo={}):
        super().__init__()
        if len(weights) > len(heights):
            raise Exception("Can't have more weights than heights!")
        elif len(weights) < len(heights):
            weights = [1]*len(heights)

        self.weights = weights
        self.num_missions = len(heights)
        self.probs = []

        for i, height in enumerate(heights):
            prob = MinTimeClimbProblem(
                target_height=height, modelinfo=modelinfo)
            prob.addTrajectory(optWing=optWing)
            self.model.add_subsystem(
                f"group_{i}", prob.model, promotes=[('phase0.parameters:S', 'S')])
            self.probs.append(prob)
        if optWing:
            self.model.add_design_var('S', lower=1, upper=100, units='m**2')

    def addCompoundObj(self):
        num_missions = self.num_missions
        weights = [float(weight/sum(self.weights)) for weight in self.weights]
        times = [f"time_{i}" for i in range(num_missions)]
        weighted_sum_str = "+".join([f"{time}*{weight}" for time,
                                    weight in zip(times, weights)])
        self.model.add_subsystem('compoundComp', om.ExecComp(
            "compound_time=" + weighted_sum_str),
            promotes=['compound_time', *times])

        for i in range(num_missions):
            self.model.connect(
                f"group_{i}.phase0.t", times[i],
                src_indices=-1)
        self.model.add_objective('compound_time')

    def addDriver(self, driver='pyoptsparse', optimizer='SLSQP'):
        self.driver = om.pyOptSparseDriver() \
            if driver == 'pyoptsparse' else \
            om.ScipyOptimizeDriver()
        self.driver.options['optimizer'] = optimizer  # 'IPOPT'
        self.driver.declare_coloring()
        self.model.linear_solver = om.DirectSolver()

    def setICs(self):
        for i, prob in enumerate(self.probs):
            prob.setInitialConditions(self, f"group_{i}.")


def multiExample():
    """Example of multi mission min time to climb problem."""
    makeN2 = True if "n2" in sys.argv else False
    super_prob = MultiMinTime(heights=[10e3, 15e3], weights=[1, 1],
                              optWing=True, modelinfo={'m_initial': 23e3})
    super_prob.addCompoundObj()
    super_prob.addDriver()
    super_prob.setup()
    super_prob.setICs()
    if makeN2:
        sys.path.append('../')
        from createN2 import createN2
        createN2(__file__, super_prob)
    dm.run_problem(super_prob, simulate=True)

    wing_area = super_prob.get_val('S', units='m**2')[0]
    print("\n\n=====================================")
    for i in range(super_prob.num_missions):
        timetoclimb = super_prob.get_val(f'group_{i}.phase0.t', units='s')[-1]
        print(f"TtoC: {timetoclimb}, S: {wing_area}")

    make_min_time_climb_plot(
        solfile='dymos_solution.db',
        simfile=['dymos_simulation.db', 'dymos_simulation_1.db'],
        solprefix='group', omitpromote='traj')


def weightCompare():
    """Runs the multi mission min time to climb problem for different weights. Shows
    impact of changing weights on 1 figure."""
    heights = [10e3, 15e3]
    weights_to_test = [[1, 3], [3, 1]]
    modelinfo = {'m_initial': 18e3, 'S': 49.24, 'v_initial': 104, 'h_initial': 100,
                 'mach_final': 1.0}
    solfiles, simfiles = [], []
    for i, weights in enumerate(weights_to_test):
        super_prob = MultiMinTime(heights=heights, weights=weights,
                                  optWing=True, modelinfo=modelinfo)
        super_prob.addCompoundObj()
        super_prob.addDriver(driver='scipy')
        super_prob.setup()
        super_prob.setICs()
        solfiles.append(f'weightsSol_{i}.db')
        simfiles.append(f'weightsSim_{i}.db')
        dm.run_problem(super_prob, simulate=True,
                       solution_record_file=solfiles[-1],
                       simulation_record_file=simfiles[-1])

    fig = plt.figure()
    colores = [['r', 'b'], ['g', 'm']]
    for i, (solfile, simfile, colors) in enumerate(zip(solfiles, simfiles, colores)):
        make_min_time_climb_plot(
            solfile=solfile,
            simfile=[simfile, simfile.replace(".db", "_1.db")],
            solprefix='group', omitpromote='traj', show=False, fig=fig,
            extratitle=", ".join([str(w) for w in weights_to_test[i]]),
            colors=colors)

    plt.show()


def comparison():
    """Runs min time to climb problem for 2 heights individually, as well as with the
    multi mission approach. Creates 2 figures showing the differences between the 2."""
    heights = [10e3, 15e3]
    weights = [1, 1]
    optimize_wing = True
    m_0 = 23e3
    modelinfo = {'m_initial': m_0, 'S': 49.24, 'v_initial': 104, 'h_initial': 100,
                 'mach_final': 1.0}

    solfiles, simfiles = [], []
    for i, height in enumerate(heights):
        p = MinTimeClimbProblem(target_height=height, modelinfo=modelinfo)
        p.addTrajectory(optWing=optimize_wing)
        p.addObjective()
        p.setOptimizer(driver='pyoptsparse')
        p.setup()
        p.setInitialConditions()
        solfile, simfile = f'Sol_Comp_{i}.db', f'Sim_Comp_{i}.db'
        solfiles.append(solfile)
        simfiles.append(simfile)
        dm.run_problem(
            p, simulate=True, solution_record_file=solfile,
            simulation_record_file=simfile)

    super_prob = MultiMinTime(heights=heights, weights=weights,
                              optWing=optimize_wing, modelinfo=modelinfo)
    super_prob.addCompoundObj()
    super_prob.addDriver()
    super_prob.setup()
    super_prob.setICs()
    dm.run_problem(super_prob, simulate=True)

    fig1, fig2 = plt.figure(1), plt.figure(2)
    make_min_time_climb_plot(
        solfile=solfiles, simfile=simfiles, omitpromote='traj', show=False, fig=fig1,
        extratitle=f"{m_0} kg")
    make_min_time_climb_plot(
        solfile='dymos_solution.db',
        simfile=['dymos_simulation.db', 'dymos_simulation_1.db'],
        solprefix='group', omitpromote='traj', show=False, fig=fig2,
        extratitle=f"{m_0} kg")
    plt.show()


if __name__ == '__main__':
    if "comparison" in sys.argv:
        comparison()
    elif "weights" in sys.argv:
        weightCompare()
    else:
        multiExample()
