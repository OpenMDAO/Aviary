import numpy as np
import openmdao.api as om

from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.geometry.geometry_group import Geometry
from openaerostruct.meshing.mesh_generator import generate_mesh

from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.subsystems.atmosphere.atmosphere import AtmosphereComp
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import (
    Aircraft,
    Dynamic,
    Settings,
    Mission,
)


class AeroConditions(om.ExplicitComponent):
    """
    Compute aerodynamic flight-condition quantities needed by OAS
    using atmospheric properties supplied by Aviary.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Mission.VELOCITY, shape=nn, units='m/s')

        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, shape=nn, units='kg/m**3')

        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_VISCOSITY, shape=nn, units='Pa*s')

        add_aviary_output(self, Dynamic.Atmosphere.KINEMATIC_VISCOSITY, shape=nn, units='m**2/s')

        add_aviary_output(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

        self.add_output('re', shape=nn, units='1/m')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.declare_partials(
            Dynamic.Atmosphere.KINEMATIC_VISCOSITY,
            [Dynamic.Atmosphere.DYNAMIC_VISCOSITY, Dynamic.Atmosphere.DENSITY],
            rows=arange,
            cols=arange,
        )
        self.declare_partials('re', '*', rows=arange, cols=arange)
        self.declare_partials(
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            [Dynamic.Mission.VELOCITY, Dynamic.Atmosphere.DENSITY],
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        V = inputs[Dynamic.Mission.VELOCITY]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        mu = inputs[Dynamic.Atmosphere.DYNAMIC_VISCOSITY]

        outputs[Dynamic.Atmosphere.KINEMATIC_VISCOSITY] = mu * rho ** (-1)
        outputs['re'] = V * rho * mu ** (-1)  # Reynolds number per unit length for OpenAeroStruct
        outputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE] = 0.5 * rho * V**2

    def compute_partials(self, inputs, J):
        V = inputs[Dynamic.Mission.VELOCITY]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        mu = inputs[Dynamic.Atmosphere.DYNAMIC_VISCOSITY]

        J[Dynamic.Atmosphere.KINEMATIC_VISCOSITY, Dynamic.Atmosphere.DYNAMIC_VISCOSITY] = rho ** (
            -1
        )
        J[Dynamic.Atmosphere.KINEMATIC_VISCOSITY, Dynamic.Atmosphere.DENSITY] = -mu * rho ** (-2)
        J['re', Dynamic.Mission.VELOCITY] = rho * mu ** (-1)
        J['re', Dynamic.Atmosphere.DENSITY] = V * mu ** (-1)
        J['re', Dynamic.Atmosphere.DYNAMIC_VISCOSITY] = -V * rho * mu ** (-2)
        J[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY] = rho * V
        J[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.DENSITY] = 0.5 * V**2


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

    def setup_partials(self):
        nn = self.options['num_nodes']
        for i in range(nn):
            self.declare_partials(Dynamic.Vehicle.LIFT, 'L_' + str(i), rows=[i], cols=[0], val=1.0)
            self.declare_partials('lifting_surface_drag', 'D_' + str(i), rows=[i], cols=[0], val=1.0)
            self.declare_partials('lifting_surface_CL', 'CL_' + str(i), rows=[i], cols=[0], val=1.0)
            self.declare_partials('lifting_surface_CD', 'CD_' + str(i), rows=[i], cols=[0], val=1.0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']

        outputs[Dynamic.Vehicle.LIFT] = np.array([inputs['L_' + str(i)] for i in range(nn)])
        outputs['lifting_surface_drag'] = np.array([inputs['D_' + str(i)] for i in range(nn)])
        outputs['lifting_surface_CL'] = np.array([inputs['CL_' + str(i)] for i in range(nn)])
        outputs['lifting_surface_CD'] = np.array([inputs['CD_' + str(i)] for i in range(nn)])

class BroadcastWing(om.ExplicitComponent):
    # broadcast geometric variables to node in the mesh
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.INCIDENCE, units='deg')
        self.add_output('broadcast_incidence', val=np.zeros(nn), units='deg')

        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='m')
        self.add_output('broadcast_wing_chord', val=np.zeros(nn), units='m')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)
        self.declare_partials(
            'broadcast_incidence', Aircraft.Wing.INCIDENCE, rows=rows_cols, cols=rows_cols, val=1.0)
        self.declare_partials(
            'broadcast_wing_chord', Aircraft.Wing.ROOT_CHORD, rows=rows_cols, cols=rows_cols, val=1.0)

    def compute(self, inputs, outputs):
        outputs['broadcast_incidence'][:] = inputs[Aircraft.Wing.INCIDENCE]
        outputs['broadcast_wing_chord'][:] = inputs[Aircraft.Wing.ROOT_CHORD]


class BroadcastHTailChord(om.ExplicitComponent):
    # broadcast geometric variables to node in the mesh
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.HorizontalTail.ROOT_CHORD, units='m')
        self.add_output('broadcast_htail_chord', val=np.zeros(nn), units='m')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)
        self.declare_partials(
            'broadcast_htail_chord',
            Aircraft.HorizontalTail.ROOT_CHORD,
            rows=rows_cols,
            cols=rows_cols,
            val=1.0,
        )

    def compute(self, inputs, outputs):
        outputs['broadcast_htail_chord'][:] = inputs[Aircraft.HorizontalTail.ROOT_CHORD]


class LiftBalanceComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        add_aviary_option(self, Mission.GRAVITY, units='m/s**2')

    def setup(self):
        nn = self.options['num_nodes']
        add_aviary_input(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')
        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')

        # This output will be constrained to zero.
        self.add_output(
            'lift_balance_residual',
            val=np.zeros(nn),
            units='N',
            desc='Lift equilibrium residual',
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)
        self.declare_partials(
            'lift_balance_residual', Dynamic.Vehicle.LIFT, rows=rows_cols, cols=rows_cols, val=1.0
        )
        self.declare_partials(
            'lift_balance_residual', Dynamic.Vehicle.MASS, rows=rows_cols, cols=rows_cols
        )

    def compute(self, inputs, outputs):
        L = inputs[Dynamic.Vehicle.LIFT]
        m = inputs[Dynamic.Vehicle.MASS]
        g = self.options[Mission.GRAVITY][0]  # m/s**2
        outputs['lift_balance_residual'] = L - (
            m * g
        )

    def compute_partials(
        self, inputs, partials):
        g = self.options[Mission.GRAVITY][0]  # m/s**2
        partials['lift_balance_residual', Dynamic.Vehicle.MASS] = -g


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
        atmosphere_model = aviary_inputs.get_val(Settings.ATMOSPHERE_MODEL)

        self.add_subsystem(
            'av_atmosphere',
            AtmosphereComp(
                num_nodes=nn,
                h_def='geometric',
                **{Settings.ATMOSPHERE_MODEL: atmosphere_model},
            ),
            promotes_inputs=[Dynamic.Mission.ALTITUDE],
            promotes_outputs=[
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
                'temperature',
                'speed_of_sound',
            ],
        )

        self.add_subsystem(
            'broadcast_wing',
            BroadcastWing(num_nodes=nn),
            promotes_inputs=[Aircraft.Wing.INCIDENCE, Aircraft.Wing.ROOT_CHORD],
            promotes_outputs=['broadcast_incidence', 'broadcast_wing_chord'],
        )

        self.add_subsystem(
            'broadcast_htail_chord',
            BroadcastHTailChord(num_nodes=nn),
            promotes_inputs=[Aircraft.HorizontalTail.ROOT_CHORD],
            promotes_outputs=['broadcast_htail_chord'],
        )

        mesh_dict = {
            'num_y': 23,  # if changing, change in broadcast components too
            'num_x': 7,
            'wing_type': 'rect',
            'symmetry': True,
            'span': 1,  # set to 1, aviary inputs will be scaling factor
            'root_chord': 1,
            'taper': 1,
            'sweep': 1,
            'span_cos_spacing': 1,
            'chord_cos_spacing': 1,
            'num_twist_cp': 1,
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
            'CD0': 0.015,
        }

        # HTAIL: use actual Aviary geometry
        wing_dist = aviary_inputs.get_val(Aircraft.Wing.CENTER_DISTANCE, units='unitless')
        fuselage_length = aviary_inputs.get_val(Aircraft.Fuselage.LENGTH, units='m')
        wing_location = wing_dist * fuselage_length
        htail_dist = fuselage_length - wing_location

        htail_z_offset = 0.10  # m

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
            'offset': np.array([htail_dist, 0, htail_z_offset]),  # offset from wing, x-aft and z-up
        }

        htail_mesh = generate_mesh(mesh_dict)

        htail_surface = {
            'name': 'htail',
            'symmetry': True,
            'S_ref_type': 'projected',
            'mesh': htail_mesh,
            'fem_model_type': 'tube',
            't_over_c': aviary_inputs.get_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD),
            'c_max_t': 0.3,  # NACA 00 series
            'with_viscous': False,
            'with_wave': False,
            'k_lam': 0.15,
            'CL0': 0.0,
            'CD0': 0.015,
        }

        surfaces = [wing_surface, htail_surface]

        prob_vars = om.IndepVarComp()
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
            point_name = 'aero_point_' + str(i)
            self.add_subsystem(point_name, AeroPoint(surfaces=surfaces))

            self.promotes(point_name, inputs=[('v', Dynamic.Mission.VELOCITY)], src_indices=[i])
            self.promotes(
                point_name,
                inputs=['alpha'],
                src_indices=[i],
                flat_src_indices=True,
            )
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
                'lifting_surface_CD',
            ],
        )

        self.options['auto_order'] = True

    def configure(self):
        # changing any value in Aviary applies a scaling factor to the existing geometry in the mesh
        # that's why all the values are 1 in the mesh dictionaries
        self.promotes('wing', inputs=[('mesh.stretch.span', Aircraft.Wing.SPAN)])
        self.promotes('htail', inputs=[('mesh.stretch.span', Aircraft.HorizontalTail.SPAN)])

        self.connect('broadcast_wing_chord', 'wing.mesh.scale_x.chord')
        self.connect('broadcast_htail_chord', 'htail.mesh.scale_x.chord')

        # self.promotes('wing', inputs=[('mesh.taper.taper', Aircraft.Wing.TAPER_RATIO)])
        # self.promotes('htail', inputs=[('mesh.taper.taper', Aircraft.HorizontalTail.TAPER_RATIO)])

        self.promotes('wing', inputs=[('mesh.sweep.sweep', Aircraft.Wing.SWEEP)])
        self.promotes('htail', inputs=[('mesh.sweep.sweep', Aircraft.HorizontalTail.SWEEP)])

        self.connect('broadcast_incidence', 'wing.mesh.rotate.twist')
