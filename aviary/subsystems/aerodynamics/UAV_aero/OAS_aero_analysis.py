'''


'''

import numpy as np
import openmdao.api as om

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.meshing.mesh_generator import generate_mesh

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.subsystems.atmosphere.atmosphere import AtmosphereComp
from aviary.variable_info.enums import AtmosphereModel
from aviary.variable_info.variables import Aircraft, Dynamic, Settings

class AeroConditions(om.ExplicitComponent):
    """
    Compute aerodynamic flight-condition quantities needed by OAS
    using atmospheric properties supplied by Aviary.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(
            self,
            Dynamic.Mission.VELOCITY,
            shape=nn,
            units='m/s'
        )

        add_aviary_input(
            self,
            Dynamic.Atmosphere.DENSITY,
            shape=nn,
            units='kg/m**3'
        )

        add_aviary_input(
            self,
            Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
            shape=nn,
            units='Pa*s'
        )

        add_aviary_output(
            self,
            Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
            shape=nn,
            units='m**2/s'
        )

        add_aviary_output(
            self,
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            shape=nn,
            units='N/m**2'
        )

        self.add_output(
            're',
            shape=nn,
            units='1/m'
        )

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):

        V = inputs[Dynamic.Mission.VELOCITY]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        mu = inputs[Dynamic.Atmosphere.DYNAMIC_VISCOSITY]

        nu = mu / rho

        outputs[Dynamic.Atmosphere.KINEMATIC_VISCOSITY] = nu

        # Reynolds number per unit length for OpenAeroStruct
        outputs['re'] = V / nu

        outputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE] = (
            0.5 * rho * V**2
        )

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
#change the alpha comp from a implicit componenet to a slack variable 
class AlphaComp(om.ExplicitComponent):
    # compute AoA using aircraft mass, assuming lift = weight * cos(alpha)
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)
        add_aviary_input(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')
        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')
        self.add_input( 'alpha', val=np.full(nn, 3), units='deg', )

        # This output will be constrained to zero.
        self.add_output( 'lift_balance_residual', val=np.zeros(nn),  units='N', desc='Lift equilibrium residual', )
        self.declare_partials('lift_balance_residual', Dynamic.Vehicle.LIFT, rows=rows_cols, cols=rows_cols)
        self.declare_partials('lift_balance_residual', Dynamic.Vehicle.MASS, rows=rows_cols, cols=rows_cols)
        self.declare_partials('lift_balance_residual', 'alpha', rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs):
        L = inputs[Dynamic.Vehicle.LIFT]
        m = inputs[Dynamic.Vehicle.MASS]
        g = 9.8 # m/s**2
        a = np.radians(inputs['alpha'])
        outputs['lift_balance_residual'] = L - (m * g) #chnaged the lift equation to be equal to weight instead of weight * cos(alpha)

    def compute_partials(self, inputs, partials): #changes the lift equation to be equal to weight instead of weight * cos(alpha)
        nn = self.options['num_nodes']

        g = 9.8

        partials[
            'lift_balance_residual',
            Dynamic.Vehicle.LIFT
        ] = np.ones(nn)

        partials[
            'lift_balance_residual',
            Dynamic.Vehicle.MASS
        ] = -g * np.ones(nn)

        partials[
            'lift_balance_residual',
            'alpha'
        ] = np.zeros(nn)

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
                Dynamic.Mission.VELOCITY,
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
            ],

            promotes_outputs=[
                're',
                Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            ],
        )
        self.add_subsystem(
            'aviary_atmosphere',
            AtmosphereComp(
                num_nodes=nn,
                h_def='geometric',
                **{
                    Settings.ATMOSPHERE_MODEL:
                    AtmosphereModel.MARS_REFERENCE
                },
            ),
            promotes_inputs=[
                Dynamic.Mission.ALTITUDE,
            ],
            promotes_outputs=[
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
            ],
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
            promotes_inputs=[
                Dynamic.Vehicle.LIFT,
                Dynamic.Vehicle.MASS,
                'alpha',
            ],
            promotes_outputs=[
                'lift_balance_residual',
            ],
         )

        for surface in surfaces: 
            geom_group = Geometry(surface=surface)
            self.add_subsystem(surface['name'], geom_group)
            
        for i in range(nn):
            point_name = 'aero_point_'+ str(i)
            self.add_subsystem(point_name, AeroPoint(surfaces=surfaces))

            self.promotes(point_name, inputs=[('v', Dynamic.Mission.VELOCITY)], src_indices=[i])
            self.promotes( point_name, inputs=['alpha'], src_indices=[i], flat_src_indices=True,)
            self.connect('re', f'{point_name}.re', src_indices=[i])
            self.connect('prob_vars.cg', f'{point_name}.cg')
            self.connect(Dynamic.Atmosphere.DENSITY, f'{point_name}.rho', src_indices=[i])
            
            self.connect(f'{point_name}.total_perf.L', f'collect_lift_drag.L_{i}')
            self.connect(f'{point_name}.total_perf.D', f'collect_lift_drag.D_{i}')
            self.connect(f'{point_name}.CL', f'collect_lift_drag.CL_{i}')
            self.connect(f'{point_name}.CD', f'collect_lift_drag.CD_{i}')
                    
            for surface in surfaces:
                name = surface['name']

                self.connect(f'{name}.mesh', f'{point_name}.{name}.def_mesh')
                self.connect(f'{name}.mesh', f'{point_name}.aero_states.{name}_def_mesh')
            
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

        #self.promotes('wing', inputs=[('mesh.taper.taper', Aircraft.Wing.TAPER_RATIO)])
        # self.promotes('htail', inputs=[('mesh.taper.taper', Aircraft.HorizontalTail.TAPER_RATIO)])

        self.promotes('wing', inputs=[('mesh.sweep.sweep', Aircraft.Wing.SWEEP)])
        self.promotes('htail', inputs=[('mesh.sweep.sweep', Aircraft.HorizontalTail.SWEEP)])

        self.connect('broadcast_incidence', 'wing.mesh.rotate.twist')