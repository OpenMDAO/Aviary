import numpy as np
import openmdao.api as om

from pathlib import Path

from aviary.subsystems.aerodynamics.flops_based.drag import TotalDrag as Drag
from aviary.subsystems.aerodynamics.flops_based.lift import \
    LiftEqualsWeight as CL
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.data_interpolator_builder import build_data_interpolator
from aviary.utils.functions import get_path
from aviary.utils.named_values import NamedValues
from aviary.variable_info.variables import Aircraft, Dynamic


# Map of variable names to allowed headers for data files (only lowercase required,
# spaces are replaced with underscores when data tables are read)
# "Repeated" aliases allows variables with different cases to match with desired
# all-lowercase name
aliases = {Dynamic.Mission.ALTITUDE: ['h', 'alt', 'altitude'],
           Dynamic.Mission.MACH: ['m', 'mach'],
           'lift_coefficient': ['cl', 'coefficient_of_lift', 'lift_coefficient'],
           'lift_dependent_drag_coefficient': ['cdi', 'lift_dependent_drag_coefficient',
                                               'lift-dependent_drag_coefficient'],
           'zero_lift_drag_coefficient': ['cd0', 'zero_lift_drag_coefficient',
                                          'zero-lift_drag_coefficient'],
           }


class TabularAeroGroup(om.Group):
    """
    Define the OpenMDAO system for estimating aerodynamic performance of the vehicle
    by interpolating from a provided drag polar. Separate data tables for lift-dependent
    drag (CDI_data) and zero-lift drag (CD0_data) are required, and can be provided
    either in a .csv data table or an NamedValues object.

    Data is checked for its structure, and attempts to use a structured grid metamodel
    component where possible. The "structured" flag forces the use of the structured
    metamodel components (for both tables).

    The "connect_training_data" flag instructs the metamodel components to look for the
    drag data tables to be connected through OpenMDAO as inputs to this system, for cases
    where you are using another component to generate the drag data. When
    "connect_training_data" is True, anything provided in "CD0_data" and "CDI_data" are
    ignored.
    """

    def initialize(self):
        options = self.options

        options.declare('num_nodes', types=int)

        options.declare('CD0_data', types=(str, Path, NamedValues),
                        desc='Data file or NamedValues object containing zero-lift drag '
                             'coefficient table.')

        options.declare('CDI_data', types=(str, Path, NamedValues),
                        desc='Data file or NamedValues object containing lift-dependent '
                             'drag coefficient table.')

        options.declare('structured', types=bool, default=True,
                        desc='Flag that sets if data is a structured grid.')

        options.declare(
            'connect_training_data',
            types=bool,
            default=False,
            desc='Flag that sets if drag data for interpolation will be '
            'passed via openMDAO connections. If True, provided values '
            'for drag coefficients in data will be ignored.',
        )

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        CDI_table = options['CDI_data']
        CD0_table = options['CD0_data']
        structured = options['structured']
        connect_training_data = options['connect_training_data']

        # if data is from file, read data using alias dict
        if isinstance(CDI_table, str):
            CDI_table = get_path(CDI_table)
        if isinstance(CDI_table, Path):
            CDI_table = read_data_file(CDI_table, aliases=aliases)
        if isinstance(CD0_table, str):
            CD0_table = get_path(CD0_table)
        if isinstance(CD0_table, Path):
            CD0_table = read_data_file(CD0_table, aliases=aliases)

        if connect_training_data or not structured:
            method = 'lagrange3'
        else:
            method = '2D-lagrange3'


        CD0_interp = build_data_interpolator(
            nn,
            interpolator_data=CD0_table,
            interpolator_outputs={'zero_lift_drag_coefficient': 'unitless'},
            method=method,
            structured=structured,
            connect_training_data=connect_training_data,
        )

        CDI_interp = build_data_interpolator(
            nn,
            interpolator_data=CDI_table,
            interpolator_outputs={'lift_dependent_drag_coefficient': 'unitless'},
            method=method,
            structured=structured,
            connect_training_data=connect_training_data,
        )

        # add subsystems
        self.add_subsystem(
            Dynamic.Mission.DYNAMIC_PRESSURE, _DynamicPressure(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.VELOCITY, Dynamic.Mission.DENSITY],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE])

        self.add_subsystem(
            'lift_coefficient', CL(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.MASS,
                             Aircraft.Wing.AREA, Dynamic.Mission.DYNAMIC_PRESSURE],
            promotes_outputs=[('cl', 'lift_coefficient'), Dynamic.Mission.LIFT])

        if connect_training_data:
            extra_promotes = [('zero_lift_drag_coefficient_train',
                               Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR)]
        else:
            extra_promotes = []

        self.add_subsystem('CD0_interp', CD0_interp,
                           promotes_inputs=['*'] + extra_promotes,
                           promotes_outputs=['*'])

        if connect_training_data:
            extra_promotes = [('lift_dependent_drag_coefficient_train',
                               Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR)]
        else:
            extra_promotes = []

        self.add_subsystem('CDI_interp', CDI_interp,
                           promotes_inputs=['*'] + extra_promotes,
                           promotes_outputs=['*'])

        self.add_subsystem(
            Dynamic.Mission.DRAG,
            Drag(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
                Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
                Aircraft.Wing.AREA,
                Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
                ('CDI', 'lift_dependent_drag_coefficient'),
                ('CD0', 'zero_lift_drag_coefficient'),
                Dynamic.Mission.MACH,
                Dynamic.Mission.DYNAMIC_PRESSURE,
            ],
            promotes_outputs=['CD', Dynamic.Mission.DRAG],
        )


class _DynamicPressure(om.ExplicitComponent):
    '''
    Calculate dynamic pressure as a function of velocity and density.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(Dynamic.Mission.VELOCITY, val=np.ones(nn), units='m/s')
        self.add_input(Dynamic.Mission.DENSITY, val=np.ones(nn), units='kg/m**3')

        self.add_output(
            Dynamic.Mission.DYNAMIC_PRESSURE, val=np.ones(nn), units='N/m**2',
            desc='pressure caused by fluid motion')

    def setup_partials(self):
        nn = self.options['num_nodes']

        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Mission.DYNAMIC_PRESSURE, [
                Dynamic.Mission.VELOCITY, Dynamic.Mission.DENSITY],
            rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs):
        TAS = inputs[Dynamic.Mission.VELOCITY]
        rho = inputs[Dynamic.Mission.DENSITY]

        outputs[Dynamic.Mission.DYNAMIC_PRESSURE] = 0.5 * rho * TAS**2

    def compute_partials(self, inputs, partials):
        TAS = inputs[Dynamic.Mission.VELOCITY]
        rho = inputs[Dynamic.Mission.DENSITY]

        partials[Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY] = rho * TAS
        partials[Dynamic.Mission.DYNAMIC_PRESSURE,
                 Dynamic.Mission.DENSITY] = 0.5 * TAS**2
