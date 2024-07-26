from dymos.models.atmosphere.atmos_1976 import USatm1976Data
from scipy.interpolate import interp1d
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
        self.model.add_subsystem('pre', self.pre_mission,
                                 promotes_inputs=['*'])
        self.model.add_subsystem('post', self.post_mission)

    def add_trajectory(self, ke_max=1e2):
        # traj = self.model.add_subsystem('traj', dm.Trajectory(), promotes=['*'])
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
        self.model.set_input_defaults('dens', val=7.87, units='g/cm**3')

    def addDesVar(self):
        self.model.add_design_var('radius', lower=0.01, upper=0.10,
                                  ref0=0.01, ref=0.10, units='m')

    def linkPhases(self):
        self.model.connect('sizing_comp.mass', 'traj.parameters:m')
        self.model.connect('sizing_comp.S', 'traj.parameters:S')
        self.model.connect('sizing_comp.price', 'traj.parameters:price')

    def setInitialVals(self):
        self.set_val('radius', 0.05, units='m')
        self.set_val('dens', 7.87, units='g/cm**3')

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
                           promotes_inputs=['*'])


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
        self.add_input(name='dens', val=7870., units='kg/m**3')

        self.add_output(name='mass', shape=(1,), units='kg')
        self.add_output(name='S', shape=(1,), units='m**2')
        self.add_output(name='price', shape=(1,), units='USD')

        self.declare_partials(of='mass', wrt='dens')
        self.declare_partials(of='mass', wrt='radius')
        self.declare_partials(of='S', wrt='radius')
        self.declare_partials(of='price', wrt='radius')
        self.declare_partials(of='price', wrt='dens')

    def compute(self, inputs, outputs):
        radius = inputs['radius']
        dens = inputs['dens']
        outputs['mass'] = (4/3.) * dens * np.pi * radius ** 3
        outputs['S'] = np.pi * radius ** 2
        outputs['price'] = (4/3.) * dens * np.pi * radius ** 3 * 10  # $10 per kg

    def compute_partials(self, inputs, partials):
        radius = inputs['radius']
        dens = inputs['dens']
        partials['mass', 'dens'] = (4/3.) * np.pi * radius ** 3
        partials['mass', 'radius'] = 4. * dens * np.pi * radius ** 2
        partials['S', 'radius'] = 2 * np.pi * radius
        partials['price', 'dens'] = (4/3.) * np.pi * radius ** 3 * 10
        partials['price', 'radius'] = 4. * dens * np.pi * radius ** 2 * 10


# if run as python avcannonball.py n2, it will create an N2
makeN2 = False
if len(sys.argv) > 1:
    if "n2" in sys.argv[1].lower():
        makeN2 = True

# handling of multiple KEs
kes = [4e3]
weights = [1]
# if fewer weights present than KEs, use same weight
if len(kes) > len(weights):
    weights = [1]*len(kes)
elif len(kes) < len(weights):
    raise Exception("Cannot have more weights than cannons!")
num_trajs = len(kes)

probs = []
super_prob = om.Problem()

for i, ke in enumerate(kes):
    prob = CannonballProblem()
    prob.add_trajectory(ke_max=ke)
    prob.setDefaults()

    group = om.Group()
    group.add_subsystem('pre', prob.pre_mission)
    group.add_subsystem('traj', prob.traj)
    group.add_subsystem('post', prob.post_mission)
    super_prob.model.add_subsystem(f'probgroup_{i}', group)
    probs.append(prob)

ranges = [f"r{i}" for i in range(num_trajs)]  # looks like: [r0, r1, ...]
# weighted_sum_str looks like: 1*r0+1*r1+...
weighted_sum_str = "+".join([f"{weight}*{r}" for r, weight in zip(ranges, weights)])
super_prob.model.add_subsystem('compoundComp', om.ExecComp(
    "compound_range=" + weighted_sum_str),
    promotes=['compound_range', *ranges])

for i in range(num_trajs):
    super_prob.model.connect(
        f'probgroup_{i}.traj.descent.states:r', ranges[i],
        src_indices=-1)

super_prob.model.add_objective('compound_range', scaler=-1)  # maximize range

super_prob.driver = om.ScipyOptimizeDriver()
super_prob.driver.options['optimizer'] = 'SLSQP'
super_prob.driver.declare_coloring()
super_prob.model.linear_solver = om.DirectSolver()
super_prob.setup()

for prob in probs:
    prob.setup()
    prob.setInitialVals()

if makeN2:
    om.n2(super_prob, outfile='AvCannonball.html')
dm.run_problem(super_prob)
