from pathlib import Path

import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.aerodynamics.flops_based.drag import ScaledCD
from aviary.subsystems.aerodynamics.gasp_based.common import AeroForces, TimeRamp
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.data_interpolator_builder import build_data_interpolator
from aviary.utils.functions import get_path
from aviary.utils.named_values import NamedValues, get_items, get_keys
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

# Map of variable names to allowed headers for data files (only lowercase required,
# spaces are replaced with underscores when data tables are read)
# "Repeated" aliases allows variables with different cases to match with desired
# all-lowercase name
aliases = {
    Dynamic.Mission.ALTITUDE: ['h', 'alt', 'altitude'],
    Dynamic.Atmosphere.MACH: ['m', 'mach'],
    Dynamic.Vehicle.ANGLE_OF_ATTACK: ['alpha', 'angle_of_attack', 'AoA'],
    'flap_deflection': ['flap_deflection'],
    'hob': ['hob'],
    'lift_coefficient': ['cl', 'lift_coefficient'],
    'drag_coefficient': ['cd', 'drag_coefficient'],
    'delta_lift_coefficient': ['delta_cl', 'dcl'],
    'delta_drag_coefficient': ['delta_cd', 'dcd'],
}


class TabularCruiseAero(om.Group):
    """Free-air lift and drag using a table lookup."""

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

        self.options.declare(
            'aero_data',
            types=(str, Path, NamedValues),
            default=None,
            desc='Data file or NamedValues object containing lift and '
            'drag coefficient table as a function of altitude, '
            'Mach, and angle of attack',
        )

        self.options.declare(
            'connect_training_data',
            default=False,
            desc='When True, the aero tables will be passed as OpenMDAO variables',
        )

        self.options.declare(
            'structured',
            types=bool,
            default=True,
            desc='Flag that sets if data is a structured grid',
        )

        self.options.declare(
            'extrapolate', default=True, desc='Flag that sets if drag data can be extrapolated'
        )

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        aero_data = options['aero_data']
        connect_training_data = options['connect_training_data']
        structured = options['structured']
        extrapolate = options['extrapolate']

        # handle aliasing for training data
        extra_promotes = []
        if connect_training_data:
            extra_promotes = [
                ('lift_coefficient_train', Aircraft.Design.LIFT_POLAR),
                ('drag_coefficient_train', Aircraft.Design.DRAG_POLAR),
            ]

        interp_comp = _build_free_aero_interp(
            num_nodes=nn,
            aero_data=aero_data,
            connect_training_data=connect_training_data,
            structured=structured,
            extrapolate=extrapolate,
        )

        self.add_subsystem(
            'free_aero_interp',
            subsys=interp_comp,
            promotes_inputs=[
                Dynamic.Mission.ALTITUDE,
                Dynamic.Atmosphere.MACH,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ]
            + extra_promotes,
            promotes_outputs=[('lift_coefficient', 'CL'), ('drag_coefficient', 'CD_prescaled')],
        )

        #
        self.add_subsystem('simple_CD', ScaledCD(num_nodes=nn), promotes=['*'])

        self.add_subsystem('forces', AeroForces(num_nodes=nn), promotes=['*'])

        self.set_input_defaults(Dynamic.Atmosphere.MACH, np.zeros(nn))


class TabularLowSpeedAero(om.Group):
    """Lift and drag near the ground using a table lookup.

    Includes increments due to ground effects, landing gear, and flaps. Retraction or
    application of landing gear and flaps to vary increments over time are controlled by
    options.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

        self.options.declare(
            'free_aero_data',
            types=(str, Path, NamedValues),
            default=None,
            desc='Data file or NamedValues object containing free aero '
            'lift and drag coefficient table as a function of '
            'altitude, Mach, and angle of attack',
        )
        self.options.declare(
            'flaps_aero_data',
            types=(str, Path, NamedValues),
            default=None,
            desc='Data file or NamedValues object containing flaps aero '
            'lift and drag coefficient table as a function of '
            'altitude, Mach, and angle of attack',
        )
        self.options.declare(
            'ground_aero_data',
            types=(str, Path, NamedValues),
            default=None,
            desc='Data file or NamedValues object containing ground '
            'aero lift and drag coefficient table as a function '
            'of altitude, Mach, and angle of attack',
        )

        self.options.declare(
            'connect_training_data',
            types=bool,
            default=False,
            desc='When True, all aero tables will be passed as OpenMDAO variables',
        )
        self.options.declare(
            'structured',
            types=bool,
            default=True,
            desc='Flag that sets if all data are structured grids',
        )
        self.options.declare(
            'extrapolate',
            types=bool,
            default=False,
            desc='Flag that sets if all drag data can be extrapolated',
        )

        self.options.declare(
            'retract_gear',
            default=True,
            types=bool,
            desc='True to start with gear landing gear down, False for reverse',
        )
        self.options.declare(
            'retract_flaps',
            default=True,
            types=bool,
            desc='True to start with flaps applied, False for reverse',
        )

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        free_aero_data = options['free_aero_data']
        flaps_aero_data = options['flaps_aero_data']
        ground_aero_data = options['ground_aero_data']
        connect_training_data = options['connect_training_data']
        structured = options['structured']
        extrapolate = options['extrapolate']

        # convert altitude to height/span for ground effects
        hob = om.ExecComp(
            'hob = (wing_height + altitude - airport_alt) / wingspan',
            altitude=dict(val=np.zeros(nn), units='ft'),
            airport_alt=dict(val=0, units='ft'),
            wingspan=dict(val=117.8, units='ft'),
            wing_height=dict(val=8.0, units='ft', desc='Wing height above ground at 0 alt'),
            hob=dict(shape=(nn,), desc='Wing height/span', units='unitless'),
            has_diag_partials=True,
        )
        self.add_subsystem(
            'hob',
            hob,
            promotes_inputs=[
                Dynamic.Mission.ALTITUDE,
                'airport_alt',
                ('wingspan', Aircraft.Wing.SPAN),
                ('wing_height', Aircraft.Wing.HEIGHT),
            ],
            promotes_outputs=['hob'],
        )

        free_aero_interp = _build_free_aero_interp(
            nn,
            aero_data=free_aero_data,
            connect_training_data=connect_training_data,
            structured=structured,
            extrapolate=extrapolate,
        )

        # "base" free-air coefficients
        self.add_subsystem(
            'interp_free',
            free_aero_interp,
            promotes_inputs=[
                Dynamic.Mission.ALTITUDE,
                Dynamic.Atmosphere.MACH,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ],
            promotes_outputs=[
                ('lift_coefficient', 'CL_free'),
                ('drag_coefficient', 'CD_free'),
                ('lift_coefficient_max', 'CL_max_free'),
            ],
        )

        flaps_aero_interp = _build_flaps_aero_interp(
            nn,
            aero_data=flaps_aero_data,
            connect_training_data=connect_training_data,
            structured=structured,
            extrapolate=extrapolate,
        )

        # flap drag and lift increment from full flap deflection
        self.add_subsystem(
            'interp_flaps',
            flaps_aero_interp,
            promotes_inputs=[
                ('flap_deflection', 'flap_defl'),
                Dynamic.Atmosphere.MACH,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
            ],
            promotes_outputs=[
                ('delta_lift_coefficient', 'dCL_flaps_full'),
                ('delta_drag_coefficient', 'dCD_flaps_full'),
                ('delta_lift_coefficient_max', 'dCL_max_flaps'),
            ],
        )

        ground_aero_interp = _build_ground_aero_interp(
            nn,
            aero_data=ground_aero_data,
            connect_training_data=connect_training_data,
            structured=structured,
            extrapolate=extrapolate,
        )

        # drag and lift increments from ground effects
        self.add_subsystem(
            'interp_ground',
            ground_aero_interp,
            promotes_inputs=[
                Dynamic.Atmosphere.MACH,
                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                'hob',
            ],
            promotes_outputs=[
                ('delta_lift_coefficient', 'dCL_ground'),
                ('delta_drag_coefficient', 'dCD_ground'),
                ('delta_lift_coefficient_max', 'dCL_max_ground'),
            ],
        )

        # drag increment from landing gear
        self.add_subsystem(
            'gear_drag',
            GearDragIncrement(num_nodes=nn),
            promotes_inputs=['aircraft:*', 'flap_defl', 'mission:*'],
            promotes_outputs=[('dCD', 'dCD_gear_full')],
        )

        # scale gear drag increment over time
        self.add_subsystem(
            'gear_drag_ramp',
            TimeRamp(num_nodes=nn, num_inputs=1, ramp_up=not self.options['retract_gear']),
            promotes_inputs=[
                ('t_init', 't_init_gear'),
                ('duration', 'dt_gear'),
                ('x', 'dCD_gear_full'),
                't_curr',
            ],
            promotes_outputs=[('y', 'dCD_gear')],
        )

        # scale flap drag and lift
        # TODO mux these?
        self.add_subsystem(
            'flaps_drag_ramp',
            TimeRamp(num_nodes=nn, num_inputs=1, ramp_up=not self.options['retract_gear']),
            promotes_inputs=[
                ('t_init', 't_init_flaps'),
                ('duration', 'dt_flaps'),
                ('x', 'dCD_flaps_full'),
                't_curr',
            ],
            promotes_outputs=[('y', 'dCD_flaps')],
        )
        self.add_subsystem(
            'flaps_lift_ramp',
            TimeRamp(num_nodes=nn, num_inputs=1, ramp_up=not self.options['retract_gear']),
            promotes_inputs=[
                ('t_init', 't_init_flaps'),
                ('duration', 'dt_flaps'),
                ('x', 'dCL_flaps_full'),
                't_curr',
            ],
            promotes_outputs=[('y', 'dCL_flaps')],
        )

        # add up CL + increments, CD + increments, CL_max + increments
        adder = om.AddSubtractComp()
        adder.add_equation(
            'CL', vec_size=nn, units='unitless', input_names=['CL_free', 'dCL_ground', 'dCL_flaps']
        )
        adder.add_equation(
            'CD',
            vec_size=nn,
            units='unitless',
            input_names=['CD_free', 'dCD_ground', 'dCD_flaps', 'dCD_gear'],
        )
        adder.add_equation(
            'CL_max',
            vec_size=nn,
            units='unitless',
            input_names=['CL_max_free', 'dCL_max_flaps', 'dCL_max_ground'],
        )
        self.add_subsystem('coef_sum', adder, promotes=['*'])

        # convert coefficients to forces
        self.add_subsystem(
            'forces',
            AeroForces(num_nodes=nn),
            promotes_inputs=[
                'CL',
                'CD',
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            ]
            + ['aircraft:*'],
            promotes_outputs=[Dynamic.Vehicle.LIFT, Dynamic.Vehicle.DRAG],
        )

        if self.options['retract_gear']:
            # takeoff defaults
            self.set_input_defaults('dt_gear', 7)
        else:
            # TODO default gear duration for landing?
            pass

        if self.options['retract_flaps']:
            # takeoff defaults
            self.set_input_defaults('flap_defl', 10 * np.ones(nn))
            self.set_input_defaults('dt_flaps', 3)
        else:
            # landing defaults
            self.set_input_defaults('flap_defl', 40 * np.ones(nn))
            # TODO default flap duration for landing?

        self.set_input_defaults(Dynamic.Mission.ALTITUDE, np.zeros(nn))
        self.set_input_defaults(Dynamic.Atmosphere.MACH, np.zeros(nn))


class GearDragIncrement(om.ExplicitComponent):
    """Gear drag coefficient increment.

    Constant for a given *full* flap deflection.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')

        # note flap_defl should be a constant (scalar) but interpolation breaks with
        # mixed scalar/vector inputs so you can just ``prob.set_val("flap_defl", 10)``
        self.add_input(
            'flap_defl',
            0.0,
            shape=nn,
            units='deg',
            desc='DFLAP: Flap deflection angle. This is the *full* flap '
            'deflection angle for a given flight segment (DFLPLD, '
            'DFLPTO).',
        )

        self.add_output(
            'dCD', units='unitless', shape=nn, desc='Drag coefficient increment for landing gear'
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.declare_partials('dCD', ['flap_defl'], rows=arange, cols=arange)
        self.declare_partials('dCD', [Mission.Design.GROSS_MASS, Aircraft.Wing.AREA])

    def compute(self, inputs, outputs):
        gross_mass_initial, wing_area, flap_defl = inputs.values()
        gross_wt_initial = gross_mass_initial * GRAV_ENGLISH_LBM

        # landing gear flat plate area
        grfe = 0.0033 * gross_wt_initial**0.785
        # landing gear CD increment for zero flap deflection
        grcd = grfe / wing_area

        outputs['dCD'] = grcd * (1 - 0.454545 * flap_defl / 50)

    def compute_partials(self, inputs, J):
        gross_mass_initial, wing_area, flap_defl = inputs.values()
        gross_wt_initial = gross_mass_initial * GRAV_ENGLISH_LBM
        grfe = 0.0033 * gross_wt_initial**0.785
        grcd = grfe / wing_area

        J['dCD', Mission.Design.GROSS_MASS] = (
            (0.0033 * 0.785 * gross_wt_initial ** (0.785 - 1) / wing_area)
            * (1 - 0.454545 * flap_defl / 50)
            * GRAV_ENGLISH_LBM
        )
        J['dCD', Aircraft.Wing.AREA] = -(grfe / wing_area**2) * (1 - 0.454545 * flap_defl / 50)
        J['dCD', 'flap_defl'] = -grcd * 0.454545 / 50


def _build_free_aero_interp(
    num_nodes=0,
    aero_data=None,
    connect_training_data=False,
    method='lagrange2',
    structured=True,
    extrapolate=True,
):
    """Creates interpolation components for cruise aero."""
    # build_data_interpolator normally handles converting to filepath and reading
    # data, but here we need to query the data before building the component
    if isinstance(aero_data, str):
        aero_data = get_path(aero_data)
    if isinstance(aero_data, Path):
        aero_data, _, _ = read_data_file(aero_data, aliases=aliases)

    # aero_data is modified in-place, deepcopy required
    interp_data = aero_data.deepcopy()

    interp_data = _structure_special_grid(interp_data)

    required_inputs = {
        Dynamic.Mission.ALTITUDE,
        Dynamic.Atmosphere.MACH,
        Dynamic.Vehicle.ANGLE_OF_ATTACK,
    }
    required_outputs = {'lift_coefficient', 'drag_coefficient'}

    missing_variables = []
    if not required_inputs <= get_keys(interp_data):
        missing_variables.append([key for key in required_inputs.difference(get_keys(interp_data))])
    if not connect_training_data and not required_outputs <= get_keys(interp_data):
        missing_variables.append(
            [key for key in required_outputs.difference(get_keys(interp_data))]
        )
    if missing_variables:
        raise KeyError(
            f'GASP-based aerodynamics interpolation missing required variables: {missing_variables}'
        )

    if connect_training_data:
        method = 'lagrange2'
    else:
        method = '3D-lagrange2'

    # add the 3d metamodel to the group, promoting all variables
    interp_comp = build_data_interpolator(
        num_nodes=num_nodes,
        interpolator_data=interp_data,
        interpolator_outputs={'lift_coefficient': 'unitless', 'drag_coefficient': 'unitless'},
        method=method,
        structured=structured,
        connect_training_data=connect_training_data,
        extrapolate=extrapolate,
    )

    if connect_training_data:
        return interp_comp
    else:
        group = om.Group()
        group.add_subsystem('free_aero_interp', interp_comp, promotes=['*'])

        # approx CL max as that with highest alpha, assume alpha monotonically increasing
        # free aero CL at max alpha is the same across altitudes but ignore that for now
        cl_max = interp_data.get_val('lift_coefficient', 'unitless')[0, :, -1]
        # add a 1d metamodel for cl_max, promoting all variables
        meta_1d = om.MetaModelStructuredComp(
            method='1D-lagrange2', vec_size=num_nodes, extrapolate=extrapolate
        )
        meta_1d.add_input(
            Dynamic.Atmosphere.MACH,
            0.0,
            units='unitless',
            shape=num_nodes,
            training_data=interp_data.get_val(Dynamic.Atmosphere.MACH, 'unitless'),
        )
        meta_1d.add_output(
            'lift_coefficient_max', units='unitless', shape=num_nodes, training_data=cl_max
        )

        group.add_subsystem('lift_coefficient_max_interp', meta_1d, promotes=['*'])
        return group


def _build_flaps_aero_interp(
    num_nodes=0,
    aero_data=None,
    connect_training_data=False,
    method='slinear',
    structured=True,
    extrapolate=False,
):
    """Creates interpolation components for cruise aero."""
    # TODO linear method default because standard GASP tables have only two flap
    #      deflections - may want to have option for two separate 2D tables instead?

    # build_data_interpolator normally handles converting to filepath and reading
    # data, but here we need to query the data before building the component
    if isinstance(aero_data, str):
        aero_data = get_path(aero_data)
    if isinstance(aero_data, Path):
        aero_data, _, _ = read_data_file(aero_data, aliases=aliases)

    # aero_data is modified in-place, deepcopy required
    interp_data = aero_data.deepcopy()

    interp_data = _structure_special_grid(interp_data)

    required_inputs = {'flap_deflection', Dynamic.Atmosphere.MACH, Dynamic.Vehicle.ANGLE_OF_ATTACK}
    required_outputs = {'delta_lift_coefficient', 'delta_drag_coefficient'}

    missing_variables = []
    if not required_inputs <= get_keys(interp_data):
        missing_variables.extend([key for key in required_inputs.difference(get_keys(interp_data))])
    if not connect_training_data and not required_outputs <= get_keys(interp_data):
        missing_variables.extend(
            [key for key in required_outputs.difference(get_keys(interp_data))]
        )
    if missing_variables:
        raise KeyError(
            f'GASP-based aerodynamics interpolation missing required variables: {missing_variables}'
        )

    dcl = interp_data.get_val('delta_lift_coefficient', 'unitless')
    defl = np.unique(
        interp_data.get_val('flap_deflection', 'deg')
    )  # units don't matter, not using values
    alpha = np.unique(
        interp_data.get_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 'deg')
    )  # units don't matter, not using values
    mach = np.unique(interp_data.get_val(Dynamic.Atmosphere.MACH, 'unitless'))

    dcl_max = np.zeros_like(dcl)
    shape = (defl.size, mach.size, alpha.size)
    dcl_max = np.resize(dcl_max, shape)
    dcl = np.reshape(dcl, shape)
    for i in range(defl.size):
        dcl_max[i, :, :] = np.broadcast_to(dcl[i, :, -1], (alpha.size, mach.size)).T

    interp_data.set_val('delta_lift_coefficient_max', dcl_max.flatten(), 'unitless')

    return build_data_interpolator(
        num_nodes=num_nodes,
        interpolator_data=interp_data,
        interpolator_outputs={
            'delta_lift_coefficient': 'unitless',
            'delta_drag_coefficient': 'unitless',
            'delta_lift_coefficient_max': 'unitless',
        },
        method=method,
        structured=structured,
        connect_training_data=connect_training_data,
        extrapolate=extrapolate,
    )


def _build_ground_aero_interp(
    num_nodes=0,
    aero_data=None,
    connect_training_data=False,
    method='slinear',
    structured=True,
    extrapolate=True,
):
    """Creates interpolation components for cruise aero."""
    # build_data_interpolator normally handles converting to filepath and reading
    # data, but here we need to query the data before building the component
    if isinstance(aero_data, str):
        aero_data = get_path(aero_data)
    if isinstance(aero_data, Path):
        aero_data, _, _ = read_data_file(aero_data, aliases=aliases)

    # aero_data is modified in-place, deepcopy required
    interp_data = aero_data.deepcopy()

    required_inputs = {'hob', Dynamic.Atmosphere.MACH, Dynamic.Vehicle.ANGLE_OF_ATTACK}
    required_outputs = {'delta_lift_coefficient', 'delta_drag_coefficient'}

    missing_variables = []
    if not required_inputs <= get_keys(interp_data):
        missing_variables.append([key for key in required_inputs.difference(get_keys(interp_data))])
    if not connect_training_data and not required_outputs <= get_keys(interp_data):
        missing_variables.append(
            [key for key in required_outputs.difference(get_keys(interp_data))]
        )
    if missing_variables:
        raise KeyError(
            f'GASP-based aerodynamics interpolation missing required variables: {missing_variables}'
        )

    dcl = interp_data.get_val('delta_lift_coefficient', 'unitless')
    alpha = np.unique(
        interp_data.get_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 'deg')
    )  # units don't matter, not using values
    mach = np.unique(interp_data.get_val(Dynamic.Atmosphere.MACH, 'unitless'))
    hob = np.unique(interp_data.get_val('hob', 'unitless'))

    dcl_max = np.zeros_like(dcl)
    shape = (mach.size, hob.size, alpha.size)
    dcl_max = np.resize(dcl_max, shape)
    dcl = np.reshape(dcl, shape)
    for i in range(mach.size):
        dcl_max[i, :, :] = np.broadcast_to(dcl[i, :, -1], (alpha.size, hob.size)).T

    interp_data.set_val('delta_lift_coefficient_max', dcl_max.flatten(), 'unitless')

    # extrapolation fine especially for HOB over max
    return build_data_interpolator(
        num_nodes=num_nodes,
        interpolator_data=interp_data,
        interpolator_outputs={
            'delta_lift_coefficient': 'unitless',
            'delta_drag_coefficient': 'unitless',
            'delta_lift_coefficient_max': 'unitless',
        },
        method=method,
        structured=structured,
        connect_training_data=connect_training_data,
        extrapolate=extrapolate,
    )


def _structure_special_grid(aero_data):
    """
    Structure a GASP-based data table that has a special case with incorrect number
    of alpha points for a structured grid.

    This assumes that the first two values in the data table (or directly provided data)
    are the other independent variables (mach, alt, or hob depending on use case), while
    the third independent variable is alpha
    """
    # store these to later put data back in the NamedValues without knowing what's in it
    data = []
    keys = []
    units = []
    for idx, (key, (val, unit)) in enumerate(get_items(aero_data)):
        if idx < 2:
            data.append(val)
            keys.append(key)
            units.append(unit)
        else:
            break

    x0 = np.unique(data[0])
    x1 = np.unique(data[1])
    # units don't matter, not using values
    aoa = aero_data.get_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, 'deg')

    if data[0].shape[0] > x0.shape[0]:
        # special case alpha - this format saturates alpha at its max
        # get unique alphas at zero alt mach 0, should cover the full range
        mask = (data[0] == x0[0]) & (data[1] == x1[0])
        aoa = np.unique(aoa[mask])

    _, _, aoa = np.meshgrid(x0, x1, aoa)

    # put the aoa data back in the NamedValues object
    aero_data.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, aoa.flatten(), 'deg')

    return aero_data
