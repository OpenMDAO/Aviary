from os import stat
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm
from dymos.models.atmosphere.atmos_1976 import USatm1976Data

"""
Currently has linked sizing and motion ODEs, want to add ability to have 
multiple trajectories (and associated ODEs) that are linked with the same
sizing class. radii/density etc can be defined as 1xn inputs, with n corresponding
to num of missions.

Additionally need to incorporate some output into a compound objective that is optimized,
aside from each mission optimizing for max range. Could for example assign $/kg of material.

Could also add a "fuel" parameter which can then be minimized.
"""
# Working version


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


def createTrajectory(ke_max):
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

    descent.add_objective('r', loc='final', scaler=-1.0)  # negative means to maximize
    for param in ('CD', 'm', 'S', 'price'):
        traj.add_parameter(param, static_target=True)

    # Link Phases (link time and all state variables)
    traj.link_phases(phases=['ascent', 'descent'], vars=['*'])
    # have to set muzzle energy here before setup for sim to run properly
    ascent.add_boundary_constraint('ke', loc='initial',
                                   upper=ke_max, lower=0, ref=100000)
    return traj, ascent, descent


p = om.Problem(model=om.Group())

p.driver = om.ScipyOptimizeDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.declare_coloring()

p.model.add_subsystem('size_comp', CannonballSizing(),
                      promotes_inputs=['radius', 'dens'])
p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
p.model.add_design_var('radius', lower=0.01, upper=0.10,
                       ref0=0.01, ref=0.10, units='m')

kes = [4e5, 5e5]
trajs, ascents, descents = [], [], []
for ke in kes:
    a, b, c = createTrajectory(ke)
    trajs.append(a)
    ascents.append(b)
    descents.append(c)
# traj, ascent, descent = createTrajectory(4e5)
p.model.add_subsystem('traj', trajs[1])
ascent = ascents[1]
descent = descents[1]

# Issue Connections
p.model.connect('size_comp.mass', 'traj.parameters:m')
p.model.connect('size_comp.price', 'traj.parameters:price')
p.model.connect('size_comp.S', 'traj.parameters:S')

# A linear solver at the top level can improve performance.
p.model.linear_solver = om.DirectSolver()
p.setup()

p.set_val('radius', 0.05, units='m')
p.set_val('dens', 7.87, units='g/cm**3')

p.set_val('traj.parameters:CD', 0.5)

p.set_val('traj.ascent.t_initial', 0.0)
p.set_val('traj.ascent.t_duration', 10.0)
# list is initial and final, based on phase info some are fixed others are not
p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

p.set_val('traj.descent.t_initial', 10.0)
p.set_val('traj.descent.t_duration', 10.0)

p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')


dm.run_problem(p, simulate=True)
sol = om.CaseReader('dymos_solution.db').get_case('final')
sim = om.CaseReader('dymos_simulation.db').get_case('final')


def plotstuff():
    rad = p.get_val('radius', units='m')[0]
    print(f'optimal radius: {rad} m ')
    mass = p.get_val('size_comp.mass', units='kg')[0]
    print(f'cannonball mass: {mass} kg ')
    price = p.get_val('size_comp.price', units='USD')[0]
    print(f'cannonball price: ${price:.2f}')
    area = p.get_val('size_comp.S', units='cm**2')[0]
    print(f'cannonball aerodynamic reference area: {area} cm**2 ')
    angle = p.get_val('traj.ascent.timeseries.gam', units='deg')[0, 0]
    print(f'launch angle: {angle} deg')
    max_range = p.get_val('traj.descent.timeseries.r')[-1, 0]
    print(f'maximum range: {max_range} m')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    time_imp = {'ascent': p.get_val('traj.ascent.timeseries.time'),
                'descent': p.get_val('traj.descent.timeseries.time')}
    time_exp = {'ascent': sim.get_val('traj.ascent.timeseries.time'),
                'descent': sim.get_val('traj.descent.timeseries.time')}
    r_imp = {'ascent': p.get_val('traj.ascent.timeseries.r'),
             'descent': p.get_val('traj.descent.timeseries.r')}
    r_exp = {'ascent': sim.get_val('traj.ascent.timeseries.r'),
             'descent': sim.get_val('traj.descent.timeseries.r')}
    h_imp = {'ascent': p.get_val('traj.ascent.timeseries.h'),
             'descent': p.get_val('traj.descent.timeseries.h')}
    h_exp = {'ascent': sim.get_val('traj.ascent.timeseries.h'),
             'descent': sim.get_val('traj.descent.timeseries.h')}

    axes.plot(r_imp['ascent'], h_imp['ascent'], 'bo')
    axes.plot(r_imp['descent'], h_imp['descent'], 'ro')
    axes.plot(r_exp['ascent'], h_exp['ascent'], 'b--')
    axes.plot(r_exp['descent'], h_exp['descent'], 'r--')

    axes.set_xlabel('range (m)')
    axes.set_ylabel('altitude (m)')
    axes.grid(True)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
    states = ['r', 'h', 'v', 'gam']
    for i, state in enumerate(states):
        x_imp = {'ascent': sol.get_val(f'traj.ascent.timeseries.{state}'),
                 'descent': sol.get_val(f'traj.descent.timeseries.{state}')}

        x_exp = {'ascent': sim.get_val(f'traj.ascent.timeseries.{state}'),
                 'descent': sim.get_val(f'traj.descent.timeseries.{state}')}

        axes[i].set_ylabel(state)
        axes[i].grid(True)

        axes[i].plot(time_imp['ascent'], x_imp['ascent'], 'bo')
        axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
        axes[i].plot(time_exp['ascent'], x_exp['ascent'], 'b--')
        axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')

    plt.show()


plotstuff()
