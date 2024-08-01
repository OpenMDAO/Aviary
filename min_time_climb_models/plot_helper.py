import matplotlib.pyplot as plt
from openmdao.api import CaseReader


def make_min_time_climb_plot(
        solfile='dymos_solution.db', simfile='dymos_simulation.db', solprefix='',
        omitpromote=None):

    singleprob, multiprob, multitraj = True, False, False
    if isinstance(solfile, list) and isinstance(simfile, list):
        multiprob, singleprob = True, False
        if len(solfile) != len(simfile):
            raise Exception(
                "Must provide same number of solution and simulation files with separate problems")

    elif isinstance(simfile, list) and isinstance(solfile, str):
        multitraj, singleprob = True, False
        solfile = [solfile]*len(simfile)
        if solprefix == '':
            raise Exception(
                "When plotting multiple trajectories, must provide prefix to access solutions!")

    if singleprob:
        solfile, simfile = [solfile], [simfile]

    plotvars = [('r', 'h'),
                ('time', 'h'),
                ('time', 'v'),
                ('time', 'thrust'),
                ('time', 'gam'),
                ('time', 'alpha')]
    plotunits = [
        ('km', 'km'),
        ('s', 'km'),
        ('s', 'm/s'),
        ('s', 'kN'),
        ('s', 'deg'),
        ('s', 'deg')]

    numplots = len(plotvars)
    tsprefix = 'traj.phase0.timeseries.'

    sols = [CaseReader(file).get_case('final') for file in solfile]
    sims = [CaseReader(file).get_case('final') for file in simfile]
    colors = ['r', 'b', 'g', 'm']
    legend = []
    areas = []
    heights = [None]*len(simfile)

    for i, (solf, simf) in enumerate(zip(sols, sims)):
        for j, ((xvar, yvar), (xunit, yunit)) in enumerate(zip(plotvars, plotunits)):
            plt.subplot(2, int(numplots/2), j+1)
            xname, yname = addPrefix(tsprefix, (xvar, yvar))
            xsim = simf.get_val(xname, units=xunit)
            ysim = simf.get_val(yname, units=yunit)
            if omitpromote is not None:
                xname = xname.replace(omitpromote+".", "")
                yname = yname.replace(omitpromote+".", "")
            if multitraj:
                xname, yname = addPrefix(f"{solprefix}_{i}.", (xname, yname))
            xsol = solf.get_val(xname, units=xunit)
            ysol = solf.get_val(yname, units=yunit)
            if yvar == "h" and not heights[i]:
                heights[i] = ysol[-1][0]
            plt.plot(xsol, ysol, f"{colors[i]}o", fillstyle='none')
            plt.plot(xsim, ysim, colors[i])
            plt.xlabel(f"{xvar} ({xunit})")
            plt.ylabel(f"{yvar} ({yunit})")
            plt.grid(visible=True)
        legend.append(f"{heights[i]} {plotunits[0][1]} solution")
        legend.append(f"{heights[i]} {plotunits[0][1]} simulation")
        try:
            areas.append(round(solf.get_val('S')[0], 2))
        except KeyError:
            areas.append(round(solf.get_val('phase0.parameters:S')[0], 2))
    plt.figlegend(legend, ncols=len(simfile))
    plt.suptitle(f"Min Time to Climb, wing areas: {areas}")
    plt.tight_layout(pad=1)
    plt.show()


def addPrefix(prefix, iterable):
    return [prefix+item for item in iterable]


if __name__ == '__main__':
    make_min_time_climb_plot(solfile=['dymos_solution_0.db', 'dymos_solution_1.db'],
                             simfile=['dymos_simulation_0.db', 'dymos_simulation_1.db'])
