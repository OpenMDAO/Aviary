'''
This is the part of the aero model that uses OAS to compute lift and drag of wing and htail,
as well as aerodynamic conditions and angle of attack.

It's hard to tell what is good here and what is broken without knowing how 
OpenAeroStruct works, so it would be a good idea to investigate further in the docs:

https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/installation.html

I am highly suspicious of the viability of this file and how it is implemented on aero_model.
'''

import numpy as np

import openmdao.api as om

from ambiance import Atmosphere

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.meshing.mesh_generator import generate_mesh

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic

class AeroConditions(om.ExplicitComponent):
    # compute atmospheric conditions, Reynolds number, dynamic pressure
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.ALTITUDE, shape=nn, units='m')
        add_aviary_input(self, Dynamic.Mission.VELOCITY, shape=nn, units='m/s')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='m')

        add_aviary_output(self, Dynamic.Atmosphere.TEMPERATURE, shape=nn, units='K')
        add_aviary_output(self, Dynamic.Atmosphere.KINEMATIC_VISCOSITY, shape=nn, units='m**2/s')
        add_aviary_output(self, Dynamic.Atmosphere.DENSITY, shape=nn, units='kg/m**3')
        add_aviary_output(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')
        
        self.add_output(name='dynamic_viscosity', shape=nn, units='kg/m/s')
        self.add_output(name='re', shape=nn, units='1/m')

        rows_cols = np.arange(nn)
        self.declare_partials('re', Dynamic.Mission.VELOCITY, rows=rows_cols, cols=rows_cols)
        self.declare_partials('re', Aircraft.Wing.ROOT_CHORD)
        self.declare_partials(Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY, rows=rows_cols, cols=rows_cols)

        self.declare_partials(Dynamic.Atmosphere.TEMPERATURE, '*', method='fd')
        self.declare_partials(Dynamic.Atmosphere.DENSITY, '*', method='fd')
        self.declare_partials('dynamic_viscosity', '*', method='fd')
        self.declare_partials(Dynamic.Atmosphere.KINEMATIC_VISCOSITY, '*', method='fd')

    def compute(self, inputs, outputs):
        V = inputs[Dynamic.Mission.VELOCITY]
        h = inputs[Dynamic.Mission.ALTITUDE]
        L = inputs[Aircraft.Wing.ROOT_CHORD]

        nn = self.options['num_nodes']
        T = np.zeros(nn)
        rho = np.zeros(nn)
        mu = np.zeros(nn)
        nu = np.zeros(nn)
        Re = np.zeros(nn)

        for i in range(nn):
            atm = Atmosphere(h[i])
            T[i] = atm.temperature
            rho[i] = atm.density
            mu[i] = atm.dynamic_viscosity
            nu[i] = mu[i] / rho[i]
            Re[i] = V[i] * L / nu[i]

        outputs[Dynamic.Atmosphere.TEMPERATURE] = T
        outputs[Dynamic.Atmosphere.DENSITY] = rho
        outputs['dynamic_viscosity'] = mu
        outputs[Dynamic.Atmosphere.KINEMATIC_VISCOSITY] = nu
        outputs['re'] = Re
        outputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE] = 0.5 * rho * V**2

    def compute_partials(self, inputs, partials):
        V = inputs[Dynamic.Mission.VELOCITY]
        L = inputs[Aircraft.Wing.ROOT_CHORD]
        nu = inputs[Dynamic.Atmosphere.KINEMATIC_VISCOSITY]
        h = inputs[Dynamic.Mission.ALTITUDE]

        nn = self.options['num_nodes']
        rho = np.zeros(nn)

        for i in range(nn):
            atm = Atmosphere(h[i])
            rho[i] = atm.density

        partials['re', Dynamic.Mission.VELOCITY] = L / nu
        partials['re', Aircraft.Wing.ROOT_CHORD] = V / nu
        partials['re', Dynamic.Atmosphere.KINEMATIC_VISCOSITY] = -V * L / nu**2
        
        partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY] = rho * V
        partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.DENSITY] = 0.5 * V**2

class CollectLiftDrag(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        for i in range(nn):
            self.add_input('L_' + str(i), units='N')
            self.add_input('D_' + str(i), units='N')
            self.add_input('CL_' + str(i), units='unitless')
            self.add_input('CD_' + str(i), units='unitless')

        self.add_output(Dynamic.Vehicle.LIFT, shape=(nn,), units='N')
        self.add_output('lifting_surface_drag', shape=(nn,), units='N')
        self.add_output('lifting_surface_CL', shape=(nn,), units='unitless')
        self.add_output('lifting_surface_CD', shape=(nn,), units='unitless')

        for i in range(nn):
            self.declare_partials(Dynamic.Vehicle.LIFT, 'L_' + str(i), rows=[i], cols=[0])
            self.declare_partials('lifting_surface_drag', 'D_' + str(i), rows=[i], cols=[0])
            self.declare_partials('lifting_surface_CL', 'CL_' + str(i), rows=[i], cols=[0])
            self.declare_partials('lifting_surface_CD', 'CD_' + str(i), rows=[i], cols=[0])
            
    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        outputs[Dynamic.Vehicle.LIFT] = np.array([inputs['L_' + str(i)] for i in range(nn)])
        outputs['lifting_surface_drag'] = np.array([inputs['D_' + str(i)] for i in range(nn)])
        outputs['lifting_surface_CL'] = np.array([inputs['CL_' + str(i)] for i in range(nn)])
        outputs['lifting_surface_CD'] = np.array([inputs['CD_' + str(i)] for i in range(nn)])
    
    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        for i in range(nn):
            partials[Dynamic.Vehicle.LIFT, 'L_' + str(i)] = 1.0
            partials['lifting_surface_drag', 'D_' + str(i)] = 1.0
            partials['lifting_surface_CL', 'CL_' + str(i)] = 1.0
            partials['lifting_surface_CD', 'CD_' + str(i)] = 1.0

class BroadcastWing(om.ExplicitComponent):
    # broadcast geometric variables to node in the mesh 
    def setup(self):
        nn = 12 # half of num_y in mesh
        add_aviary_input(self, Aircraft.Wing.INCIDENCE, units='deg')
        self.add_output('broadcast_incidence', val=np.zeros(nn), units='deg')
        
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='m')
        self.add_output('broadcast_wing_chord', val=np.zeros(nn), units='m')

        rows = np.arange(nn)
        cols = np.zeros(nn, int)

        self.declare_partials('broadcast_incidence', Aircraft.Wing.INCIDENCE, rows=rows, cols=cols)
        self.declare_partials('broadcast_wing_chord', Aircraft.Wing.ROOT_CHORD, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['broadcast_incidence'][:] = inputs[Aircraft.Wing.INCIDENCE]
        outputs['broadcast_wing_chord'][:] = inputs[Aircraft.Wing.ROOT_CHORD]

    def compute_partials(self, inputs, partials):
        nn = 12
        partials['broadcast_incidence', Aircraft.Wing.INCIDENCE] = np.ones(nn)
        partials['broadcast_wing_chord', Aircraft.Wing.ROOT_CHORD] = np.ones(nn)

class BroadcastHTailChord(om.ExplicitComponent):
    # broadcast geometric variables to node in the mesh 
    def setup(self):
        nn = 10 # half of num_y from mesh
        add_aviary_input(self, Aircraft.HorizontalTail.ROOT_CHORD, units='m')
        self.add_output('broadcast_htail_chord', val=np.zeros(nn), units='m')
        
        rows = np.arange(nn)
        cols = np.zeros(nn, int)
        
        self.declare_partials('broadcast_htail_chord', Aircraft.HorizontalTail.ROOT_CHORD, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        outputs['broadcast_htail_chord'][:] = inputs[Aircraft.HorizontalTail.ROOT_CHORD]
    
    def compute_partials(self, inputs, partials):
        nn = 10
        partials['broadcast_htail_chord', Aircraft.HorizontalTail.ROOT_CHORD] = np.ones(nn)

class AlphaComp(om.ImplicitComponent):
    # compute AoA using aircraft mass, assuming lift = weight * cos(alpha)
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')
        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')
        self.add_output('alpha', shape=nn, units='deg')

        rows_cols = np.arange(nn)
        self.declare_partials('alpha', Dynamic.Vehicle.LIFT, rows=rows_cols, cols=rows_cols)
        self.declare_partials('alpha', Dynamic.Vehicle.MASS, rows=rows_cols, cols=rows_cols)
        self.declare_partials('alpha', 'alpha', rows=rows_cols, cols=rows_cols)

    def apply_nonlinear(self, inputs, outputs, residuals):
        L = inputs[Dynamic.Vehicle.LIFT]
        m = inputs[Dynamic.Vehicle.MASS]
        g = 9.8 # m/s
        a = np.radians(outputs['alpha'])
        residuals['alpha'] = L - (m * g * np.cos(a))

    def linearize(self, inputs, outputs, partials):
        nn = self.options['num_nodes']
        L = inputs[Dynamic.Vehicle.LIFT]
        m = inputs[Dynamic.Vehicle.MASS]
        g = 9.8 # m/s
        a = np.radians(outputs['alpha'])

        partials['alpha', Dynamic.Vehicle.LIFT] = np.ones(nn)
        partials['alpha', Dynamic.Vehicle.MASS] = -g * np.cos(a)
        partials['alpha', 'alpha'] = m * g * np.sin(a)


class OASAero(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('aviary_inputs')

    def setup(self):
        nn = self.options['num_nodes']
        aviary_inputs = self.options['aviary_inputs']

        self.add_subsystem(
            'aero_conditions',
            AeroConditions(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.ALTITUDE,
                Dynamic.Mission.VELOCITY,
                Aircraft.Wing.ROOT_CHORD],
            promotes_outputs=['re', Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.DENSITY]
        )

        self.add_subsystem(
            'broadcast_wing',
            BroadcastWing(),
            promotes_inputs=[Aircraft.Wing.INCIDENCE, Aircraft.Wing.ROOT_CHORD],
            promotes_outputs=['broadcast_incidence', 'broadcast_wing_chord']
        )

        self.add_subsystem(
            'broadcast_htail_chord',
            BroadcastHTailChord(),
            promotes_inputs=[Aircraft.HorizontalTail.ROOT_CHORD],
            promotes_outputs=['broadcast_htail_chord']
        )

        # WING
        
        mesh_dict = {
            'num_y': 23, # if changing, change in broadcast components too
            'num_x': 7,
            'wing_type': 'rect', 
            'symmetry': True,
            'span': 1, # set to 1, aviary inputs will be scaling factor
            'root_chord': 1,
            'taper': 1,
            'sweep': 1,
            'span_cos_spacing': 1,
            'chord_cos_spacing': 1,
            'num_twist_cp': 1
        }

        wing_mesh = generate_mesh(mesh_dict)

        wing_surface = {
            'name': 'wing',
            'symmetry': True,
            'S_ref_type': 'projected',
            'mesh': wing_mesh,
            'fem_model_type': 'tube',
            't_over_c': aviary_inputs.get_val(Aircraft.Wing.THICKNESS_TO_CHORD),
            'c_max_t': aviary_inputs.get_val(Aircraft.Wing.MAX_THICKNESS_LOCATION),

            'with_viscous': False,
            'with_wave': False,
            'k_lam': 0.15,
            'CL0': 0.1,
            'CD0': 0.015
        }
        
        # HTAIL      

        # location of htail relative to aerodynamic center of wing
        wing_dist = aviary_inputs.get_val(Aircraft.Wing.CENTER_DISTANCE, units='unitless')
        fuselage_length = aviary_inputs.get_val(Aircraft.Fuselage.LENGTH, units='m')
        wing_location = wing_dist * fuselage_length
        htail_dist = fuselage_length - wing_location
        
        mesh_dict = {
            'num_y': 19,
            'num_x': 6,
            'wing_type': 'rect', 
            'symmetry': True,
            'span': 1,
            'root_chord': 1,
            'taper': 1,
            'sweep': 1,
            'span_cos_spacing': 1,
            'chord_cos_spacing': 1,
            'offset': np.array([htail_dist, 0, 0]), # offset from wing in x-direction
        }

        htail_mesh = generate_mesh(mesh_dict)

        htail_surface = {
            'name': 'htail',
            'symmetry': True,
            'S_ref_type': 'projected',
            'mesh': htail_mesh,
            'fem_model_type': 'tube',
            't_over_c': aviary_inputs.get_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD),
            'c_max_t': 0.3, # NACA 00 series

            'with_viscous': False, 
            'with_wave': False,
            'k_lam': 0.15,
            'CL0': 0.0, 
            'CD0': 0.015 
        }
        
        surfaces = [wing_surface, htail_surface]

        prob_vars = om.IndepVarComp()
        # array of zeros for CG (x, y, z)
        prob_vars.add_output('cg', val=np.zeros(3), units='m')
        self.add_subsystem('prob_vars', prob_vars, promotes=[])
       
        self.add_subsystem(
            'alpha_comp',
            AlphaComp(num_nodes=nn),
            promotes_inputs=[Dynamic.Vehicle.LIFT, Dynamic.Vehicle.MASS],
            promotes_outputs=['alpha']
        )

        for surface in surfaces: 
            geom_group = Geometry(surface=surface)
            self.add_subsystem(surface['name'], geom_group)
            
        for i in range(nn):
            point_name = 'aero_point_'+ str(i)
            self.add_subsystem(point_name, AeroPoint(surfaces=surfaces))

            self.promotes(point_name, inputs=[('v', Dynamic.Mission.VELOCITY)], src_indices=[i])
            self.connect('alpha', point_name + 'alpha', src_indices=[i])
            self.connect('re', point_name + '.re', src_indices=[i])
            self.connect('prob_vars.cg', point_name + '.cg')
            self.connect(Dynamic.Atmosphere.DENSITY, point_name + '.rho', src_indices=[i])
            
            self.connect(point_name + '.total_perf.L', 'collect_lift_drag.L_' + str(i))
            self.connect(point_name + '.total_perf.D', 'collect_lift_drag.D_' + str(i))
            self.connect(point_name + '.CL', 'collect_lift_drag.CL_' + str(i))
            self.connect(point_name + '.CD', 'collect_lift_drag.CD_' + str(i))
            
            for surface in surfaces:
                name = surface['name']

                self.connect(name + '.mesh', point_name + '.' + name + '.def_mesh')
                self.connect(name + '.mesh', point_name + '.aero_states.' + name + '_def_mesh')
            
        self.add_subsystem(
            'collect_lift_drag', 
            CollectLiftDrag(num_nodes=nn),
                promotes_outputs=[
                    Dynamic.Vehicle.LIFT,
                    'lifting_surface_drag',
                    'lifting_surface_CL',
                    'lifting_surface_CD'
                ]
        )
        
        self.options['auto_order'] = True

    def configure(self):
        # changing any value in Aviary applies a scaling factor to the existing geometry in the mesh
        # that's why all the values are 1 in the mesh dictionaries
        self.promotes('wing', inputs=[('mesh.stretch.span', Aircraft.Wing.SPAN)])
        self.promotes('htail', inputs=[('mesh.stretch.span', Aircraft.HorizontalTail.SPAN)])

        self.connect('broadcast_wing_chord', 'wing.mesh.scale_x.chord')
        self.connect('broadcast_htail_chord', 'htail.mesh.scale_x.chord')

        self.promotes('wing', inputs=[('mesh.taper.taper', Aircraft.Wing.TAPER_RATIO)])
        self.promotes('htail', inputs=[('mesh.taper.taper', Aircraft.HorizontalTail.TAPER_RATIO)])

        self.promotes('wing', inputs=[('mesh.sweep.sweep', Aircraft.Wing.SWEEP)])
        self.promotes('htail', inputs=[('mesh.sweep.sweep', Aircraft.HorizontalTail.SWEEP)])

        self.connect('broadcast_incidence', 'wing.mesh.rotate.twist')