from dymos.models.atmosphere.atmos_1976 import USatm1976Data
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm
import numpy as np
import sys


class CannonballProblem(om.Problem):
    def __init__(self):
        super().__init__()
        self.model = CannonballGroup()
        self.pre_mission = PreMissionGroup()
        self.post_mission = PostMissionGroup()
        self.traj = None
        self.model.add_subsystem('pre_mission', self.pre_mission,
                                 promotes=['*'])
        self.model.add_subsystem('post_mission', self.post_mission)

    def add_trajectory(self, ke_max=1e2):
        traj = dm.Trajectory()
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)
        traj.add_phase('ascent', ascent)

        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)
        traj.add_phase('descent', descent)

        for phase in (ascent, descent):
            is_ascent = phase.name == "ascent"
            phase.set_time_options(fix_initial=True if is_ascent else False,
                                   duration_bounds=(1, 100), duration_ref=100, units='s')
            phase.set_state_options('r', fix_initial=is_ascent, fix_final=False)
            phase.set_state_options('h', fix_initial=is_ascent, fix_final=not is_ascent)
            phase.set_state_options('gam', fix_initial=False, fix_final=is_ascent)
            phase.set_state_options('v', fix_initial=False, fix_final=False)
            phase.add_parameter('S', units='m**2', static_target=True, val=0.005)
            phase.add_parameter('m', units='kg', static_target=True, val=1.0)
            phase.add_parameter('price', units='USD', static_target=True, val=10)
            phase.add_parameter('CD', units=None, static_target=True, val=0.5)

        # descent.add_objective('r', loc='final', scaler=-1.0)  # negative means to maximize
        for param in ('CD', 'm', 'S', 'price'):
            traj.add_parameter(param, static_target=True)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ascent', 'descent'], vars=['*'])
        # have to set muzzle energy here before setup for sim to run properly
        ascent.add_boundary_constraint('ke', loc='initial',
                                       upper=ke_max, lower=0, ref=100000)
        self.traj = traj
        self.model.add_subsystem('traj', traj)
        self.phases = [ascent, descent]

    def setDefaults(self):
        self.model.set_input_defaults('density', val=7.87, units='g/cm**3')

    def addDesVar(self):
        self.model.add_design_var('radius', lower=0.01, upper=0.10,
                                  ref0=0.01, ref=0.10, units='m')

    def connectPreTraj(self):
        self.model.connect('mass', 'traj.parameters:m')
        self.model.connect('S', 'traj.parameters:S')
        self.model.connect('price', 'traj.parameters:price')

    def setInitialVals(self):
        self.set_val('radius', 0.05, units='m')
        self.set_val('density', 7.87, units='g/cm**3')

        self.set_val("traj.parameters:CD", 0.5)

        self.set_val("traj.ascent.t_initial", 0.0)
        self.set_val("traj.ascent.t_duration", 10.0)
        # list is initial and final, based on phase info some are fixed others are not
        ascent, descent = self.phases
        self.set_val("traj.ascent.states:r", ascent.interp('r', [0, 100]))
        self.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        self.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        self.set_val('traj.ascent.states:gam',
                     ascent.interp('gam', [25, 0]), units='deg')

        self.set_val('traj.descent.t_initial', 10.0)
        self.set_val('traj.descent.t_duration', 10.0)

        self.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        self.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        self.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        self.set_val('traj.descent.states:gam',
                     descent.interp('gam', [0, -45]), units='deg')


class CannonballGroup(om.Group):
    def __init__(self):
        super().__init__()


class PreMissionGroup(om.Group):
    def __init__(self):
        super().__init__()
        self.sizingcomp = CannonballSizing()
        self.add_subsystem('sizing_comp', self.sizingcomp,
                           promotes=['*'])


class PostMissionGroup(om.Group):
    pass


class CannonballODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # static params
        self.add_input('m', units='kg')
        self.add_input('S', units='m**2')
        self.add_input('CD', 0.5)

        # time varying inputs
        self.add_input('h', units='m', shape=nn)
        self.add_input('v', units='m/s', shape=nn)
        self.add_input('gam', units='rad', shape=nn)

        # state rates
        self.add_output('v_dot', shape=nn, units='m/s**2',
                        tags=['dymos.state_rate_source:v'])
        self.add_output('gam_dot', shape=nn, units='rad/s',
                        tags=['dymos.state_rate_source:gam'])
        self.add_output('h_dot', shape=nn, units='m/s',
                        tags=['dymos.state_rate_source:h'])
        self.add_output('r_dot', shape=nn, units='m/s',
                        tags=['dymos.state_rate_source:r'])
        self.add_output('ke', shape=nn, units='J')

        # Ask OpenMDAO to compute the partial derivatives using complex-step
        # with a partial coloring algorithm for improved performance, and use
        # a graph coloring algorithm to automatically detect the sparsity pattern.
        self.declare_coloring(wrt='*', method='cs')

        alt_data = USatm1976Data.alt * om.unit_conversion('ft', 'm')[0]
        rho_data = USatm1976Data.rho * om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
        self.rho_interp = interp1d(np.array(alt_data, dtype=complex),
                                   np.array(rho_data, dtype=complex),
                                   kind='linear')

    def compute(self, inputs, outputs):

        gam = inputs['gam']
        v = inputs['v']
        h = inputs['h']
        m = inputs['m']
        S = inputs['S']
        CD = inputs['CD']

        GRAVITY = 9.80665  # m/s**2

        # handle complex-step gracefully from the interpolant
        if np.iscomplexobj(h):
            rho = self.rho_interp(inputs['h'])
        else:
            rho = self.rho_interp(inputs['h']).real

        q = 0.5*rho*inputs['v']**2
        qS = q * S
        D = qS * CD
        cgam = np.cos(gam)
        sgam = np.sin(gam)
        outputs['v_dot'] = - D/m-GRAVITY*sgam
        outputs['gam_dot'] = -(GRAVITY/v)*cgam
        outputs['h_dot'] = v*sgam
        outputs['r_dot'] = v*cgam
        outputs['ke'] = 0.5*m*v**2


class CannonballSizing(om.ExplicitComponent):
    def setup(self):
        self.add_input(name='radius', val=1.0, units='m')
        self.add_input(name='density', val=7870., units='kg/m**3')

        self.add_output(name='mass', shape=(1,), units='kg')
        self.add_output(name='S', shape=(1,), units='m**2')
        self.add_output(name='price', shape=(1,), units='USD')

        self.declare_partials(of='mass', wrt='density')
        self.declare_partials(of='mass', wrt='radius')
        self.declare_partials(of='S', wrt='radius')
        self.declare_partials(of='price', wrt='radius')
        self.declare_partials(of='price', wrt='density')

    def compute(self, inputs, outputs):
        radius = inputs['radius']
        density = inputs['density']
        outputs['mass'] = (4/3.) * density * np.pi * radius ** 3
        outputs['S'] = np.pi * radius ** 2
        outputs['price'] = (4/3.) * density * np.pi * radius ** 3 * 10  # $10 per kg

    def compute_partials(self, inputs, partials):
        radius = inputs['radius']
        density = inputs['density']
        partials['mass', 'density'] = (4/3.) * np.pi * radius ** 3
        partials['mass', 'radius'] = 4. * density * np.pi * radius ** 2
        partials['S', 'radius'] = 2 * np.pi * radius
        partials['price', 'density'] = (4/3.) * np.pi * radius ** 3 * 10
        partials['price', 'radius'] = 4. * density * np.pi * radius ** 2 * 10


# if run as python avcannonball.py n2, it will create an N2
makeN2 = False
if len(sys.argv) > 1:
    if "n2" in sys.argv[1].lower():
        makeN2 = True


def runAvCannonball(kes=[4e3], weights=[1]):
    # handling of multiple KEs
    # if fewer weights present than KEs, use same weight
    if len(kes) > len(weights):
        weights = [1]*len(kes)
    elif len(kes) < len(weights):
        raise Exception("Cannot have more weights than cannons!")
    num_trajs = len(kes)

    probs = []
    super_prob = om.Problem()

    # create sub problems and add them to super in a group
    # group prevents spillage of promoted vars into super
    prefix = "group"  # prefix for each om.Group
    for i, ke in enumerate(kes):
        prob = CannonballProblem()
        prob.add_trajectory(ke_max=ke)
        prob.setDefaults()
        prob.addDesVar()  # doesn't seem to do anything
        prob.connectPreTraj()  # doesn't seem to actually do anything

        group = om.Group()
        group.add_subsystem('pre_mission', prob.pre_mission)
        group.add_subsystem('traj', prob.traj)
        group.add_subsystem('post_mission', prob.post_mission)
        # promoting radius and density keeps it constant for all missions
        super_prob.model.add_subsystem(prefix+f'_{i}', group,
                                       promotes_inputs=['radius', 'density'])
        probs.append(prob)

    # create an execComp with a compound range function to maximize range
    # for all cannons with a weighted function
    ranges = [f"r{i}" for i in range(num_trajs)]  # looks like: [r0, r1, ...]
    # weighted_sum_str looks like: 1*r0+1*r1+...
    weighted_sum_str = "+".join([f"{weight}*{r}" for r, weight in zip(ranges, weights)])
    super_prob.model.add_subsystem('compoundComp', om.ExecComp(
        "compound_range=" + weighted_sum_str),
        promotes=['compound_range', *ranges])

    # controlling radius to affect mass/ballistic coeff
    super_prob.model.add_design_var('radius', lower=0.01, upper=0.10,
                                    ref0=0.01, ref=0.10, units='m')

    for i in range(num_trajs):
        # connect end of trajectory range to compound range input
        super_prob.model.connect(
            prefix+f'_{i}.traj.descent.states:r', ranges[i],
            src_indices=-1)

        # connect pre-mission parameters to trajectory counter parts
        # this connection would happen within each problem's internal function,
        # but since we're extracting the components into a larger problem,
        # these connections have to be made at the super problem level
        super_prob.model.connect(
            prefix+f'_{i}.mass', prefix+f'_{i}.traj.parameters:m')
        super_prob.model.connect(
            prefix+f'_{i}.S', prefix+f'_{i}.traj.parameters:S')
        super_prob.model.connect(
            prefix+f'_{i}.price', prefix+f'_{i}.traj.parameters:price')

    super_prob.model.add_objective('compound_range', scaler=-1)  # maximize range

    super_prob.driver = om.ScipyOptimizeDriver()
    super_prob.driver.options['optimizer'] = 'SLSQP'
    super_prob.driver.declare_coloring()
    super_prob.model.linear_solver = om.DirectSolver()
    super_prob.setup()

    for i, prob in enumerate(probs):
        reference = prefix+f"_{i}"
        super_prob.set_val(reference+'.radius', 0.05, units='m')
        super_prob.set_val(reference+'.density', 7.87, units='g/cm**3')

        super_prob.set_val(reference+".traj.parameters:CD", 0.5)

        super_prob.set_val(reference+".traj.ascent.t_initial", 0.0)
        super_prob.set_val(reference+".traj.ascent.t_duration", 10.0)
        # list is initial and final, based on phase info some are fixed others are not
        ascent, descent = prob.phases
        super_prob.set_val(reference+".traj.ascent.states:r",
                           ascent.interp('r', [0, 100]))
        super_prob.set_val(reference+'.traj.ascent.states:h',
                           ascent.interp('h', [0, 100]))
        super_prob.set_val(reference+'.traj.ascent.states:v',
                           ascent.interp('v', [200, 150]))
        super_prob.set_val(reference+'.traj.ascent.states:gam',
                           ascent.interp('gam', [25, 0]), units='deg')

        super_prob.set_val(reference+'.traj.descent.t_initial', 10.0)
        super_prob.set_val(reference+'.traj.descent.t_duration', 10.0)

        super_prob.set_val(reference+'.traj.descent.states:r',
                           descent.interp('r', [100, 200]))
        super_prob.set_val(reference+'.traj.descent.states:h',
                           descent.interp('h', [100, 0]))
        super_prob.set_val(reference+'.traj.descent.states:v',
                           descent.interp('v', [150, 200]))
        super_prob.set_val(reference+'.traj.descent.states:gam',
                           descent.interp('gam', [0, -45]), units='deg')

    if makeN2:
        om.n2(super_prob, outfile='AvCannonball.html')

    dm.run_problem(super_prob)
    return super_prob, prefix, num_trajs, kes, weights


def printOutput(super_prob, prefix, num_trajs, kes, weights):
    # formatted output
    print("\n\n=================================================")
    print(
        f"Optimized {num_trajs} trajectories with weights: {', '.join(map(str,weights))}")
    rad = super_prob.get_val('radius', units='cm')[0]

    mass0 = super_prob.get_val(prefix+'_0.mass', units='kg')[0]
    price0 = super_prob.get_val(prefix+'_0.price', units='USD')[0]
    area0 = super_prob.get_val(prefix+'_0.S', units='cm**2')[0]

    # mass, price, S are outputs from sizing. These should be common amongst all
    # trajectories, however since they're outputs they cannot be promoted upto
    # super problem without unique names. This loop checks their values with
    # value of the first trajectory to ensure they are the same.
    if num_trajs > 1:
        for i in range(num_trajs-1):
            mass = super_prob.get_val(prefix+f'_{i+1}.mass', units='kg')[0]
            price = super_prob.get_val(prefix+f'_{i+1}.price', units='USD')[0]
            area = super_prob.get_val(prefix+f'_{i+1}.S', units='cm**2')[0]
            if mass != mass0 or price != price0 or area != area0:
                raise Exception(
                    "Masses, Prices, and/or Areas are not equivalent between trajectories.")

    print("\nOptimal Cannonball Description:")
    print(
        f"\tRadius: {rad:.2f} cm, Mass: {mass0:.2f} kg, Price: ${price0:.2f}, Area: {area0:.2f} sqcm")

    print("\nOptimal Trajectory Descriptions:")
    ranges = 0
    for i, ke in enumerate(kes):
        angle = super_prob.get_val(
            prefix+f'_{i}.traj.ascent.timeseries.gam', units='deg')[0, 0]
        max_range = super_prob.get_val(
            prefix+f'_{i}.traj.descent.timeseries.r')[-1, 0]

        print(
            f"\tKE: {ke/1e3:.2f} KJ, Launch Angle: {angle:.2f} deg, Max Range: {max_range:.2f} m")
        ranges += weights[i]*max_range
    print(f"System range: {ranges}")


def plotOutput(super_prob, prefix, num_trajs, kes, weights, show=True):
    if show:
        _, ax = plt.subplots()
    timevals = []
    hvals = []
    rvals = []
    for i in range(num_trajs):
        tv = []
        hv = []
        rv = []
        for phase in ('ascent', 'descent'):
            tv.append(super_prob.get_val(
                prefix+f'_{i}.traj.{phase}.timeseries.time'))
            hv.append(super_prob.get_val(
                prefix+f'_{i}.traj.{phase}.timeseries.h'))
            rv.append(super_prob.get_val(
                prefix+f'_{i}.traj.{phase}.timeseries.r'))

        timevals.append(np.vstack(tv))
        hvals.append(np.vstack(hv))
        rvals.append(np.vstack(rv))
        if show:
            ax.plot(rvals[-1], hvals[-1])

    if show:
        plt.grid()
        plt.legend([f"Weights = {weight}" for weight in weights])
        plt.title(f"Cannonballs With KEs: {kes} J")
        plt.show()
    return timevals, rvals, hvals


if __name__ == '__main__':
    testing_weights = [[3, 2, 2], [2, 2, 3]]
    kes = [4e5, 5e5, 6e5]
    legendlst = []
    rs, hs = [], []
    for i, weighting in enumerate(testing_weights):
        super_prob, prefix, num_trajs, kes, weights = runAvCannonball(
            kes=kes, weights=weighting)
        printOutput(super_prob, prefix, num_trajs, kes, weights)
        tvals, rvals, hvals = plotOutput(
            super_prob, prefix, num_trajs, kes, weights, show=False)

        for r, h, weight in zip(rvals, hvals, weighting):
            rs.append(r)
            hs.append(h)
            legendlst.append(f"Group: {weighting}, Weight: {weight}")

    _, ax = plt.subplots()
    for r, h in zip(rs, hs):
        ax.plot(r, h)
    plt.grid()
    plt.legend(legendlst)
    plt.title(f"Cannonballs With KEs: {kes} J")
    plt.show()


"""
Findings:
- connections between trajectory parameters and pre-mission parameters have to be made at
  super problem level, running the cannonballproblem method to form those connections has seemingly
  no effect, causing divide by zero errors b/c the trajectory value is not set 

- design var has to be set in super problem, cannonballproblem design var is not manipulated by optimizer,
  giving a result with the default value for ball radius

- initial values have to be set at super problem level, running cannonballproblem's initial value function
  throws error that setval cannot run before setup (even though superproblem setup was run). 
"""
