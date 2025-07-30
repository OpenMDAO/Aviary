"""
Define subsystem builder for Aviary core aerodynamics.

Classes
-------
AerodynamicsBuilderBase : the interface for an aerodynamics subsystem builder.

CoreAerodynamicsBuilder : the interface for Aviary's core aerodynamics subsystem builder
"""

import warnings

import numpy as np
import openmdao.api as om

# from dymos.utils.misc import _unspecified
from aviary.subsystems.aerodynamics.flops_based.computed_aero_group import ComputedAeroGroup
from aviary.subsystems.aerodynamics.flops_based.design import Design
from aviary.subsystems.aerodynamics.flops_based.tabular_aero_group import TabularAeroGroup
from aviary.subsystems.aerodynamics.flops_based.takeoff_aero_group import TakeoffAeroGroup
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import CruiseAero, LowSpeedAero
from aviary.subsystems.aerodynamics.gasp_based.premission_aero import PreMissionAero
from aviary.subsystems.aerodynamics.gasp_based.table_based import (
    TabularCruiseAero,
    TabularLowSpeedAero,
)
from aviary.subsystems.aerodynamics.solve_alpha_group import SolveAlphaGroup
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.named_values import NamedValues
from aviary.variable_info.enums import LegacyCode, Verbosity
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS

_default_name = 'aerodynamics'


class AerodynamicsBuilderBase(SubsystemBuilderBase):
    """
    Base class of aerodynamics builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None):
        Initializes the AerodynamicsBuilderBase object with a given name.
    mission_inputs(self, **kwargs) -> list:
        Return mission inputs.
    mission_outputs(self, **kwargs) -> list:
        Return mission outputs.
    """

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CoreAerodynamicsBuilder(AerodynamicsBuilderBase):
    """
    Core aerodynamics builder.

    Method
    ------
    build_pre_mission()
        Build pre-mission.
    build_mission()
        Build mission.
    mission_inputs()
        Return mission inputs.
    mission_outputs()
        Return mission outputs.
    get_parameters()
        Return a dictionary of fixed values for the subsystem.
    report()
        Generate the report for Aviary core aerodynamics analysis.
    """

    def __init__(self, name=None, meta_data=None, code_origin=None, tabular=False):
        if name is None:
            name = 'core_aerodynamics'

        if code_origin not in (FLOPS, GASP):
            raise ValueError('Code origin is not one of the following: (FLOPS, GASP)')

        self.code_origin = code_origin
        self.tabular = tabular

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        # pre-mission is not required when exclusively using tabular aero
        if self.tabular:
            return

        code_origin = self.code_origin
        try:
            method = kwargs['method']
        except KeyError:
            method = None

        if method == 'external':
            return None

        if code_origin is GASP:
            return PreMissionAero()

        elif code_origin is FLOPS:
            return Design()

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None

        if method == 'external':
            return None

        aero_group = None

        if self.code_origin is FLOPS:
            if 'solve_alpha' in kwargs:
                if aviary_inputs.get_val(Settings.VERBOSITY) >= Verbosity.BRIEF:
                    warnings.warn(
                        "The 'solve_alpha' flag has been set, but is not used for FLOPS-based "
                        'aerodynamics.'
                    )

            if method is None:
                aero_group = ComputedAeroGroup(num_nodes=num_nodes)

            elif method == 'computed':
                aero_group = ComputedAeroGroup(num_nodes=num_nodes, **kwargs)

            elif method == 'low_speed':
                aero_group = TakeoffAeroGroup(
                    num_nodes=num_nodes, aviary_options=aviary_inputs, **kwargs
                )

            elif method == 'tabular':
                aero_group = TabularAeroGroup(
                    num_nodes=num_nodes,
                    CD0_data=kwargs.pop('CD0_data'),
                    CDI_data=kwargs.pop('CDI_data'),
                    **kwargs,
                )

            else:
                raise ValueError(
                    'FLOPS-based aero method is not one of the following: (computed, '
                    'low_speed, tabular)'
                )

        elif self.code_origin is GASP:
            try:
                solve_alpha = kwargs.pop('solve_alpha')
            except KeyError:
                solve_alpha = False

            if method is None:
                aero_group = CruiseAero(num_nodes=num_nodes)

            elif method == 'cruise':
                aero_group = CruiseAero(num_nodes=num_nodes, **kwargs)

            elif method == 'tabular_cruise':
                # if 'aero_data' in kwargs:
                aero_group = TabularCruiseAero(
                    num_nodes=num_nodes,
                    aero_data=kwargs.pop('aero_data'),
                    **kwargs,
                )

            elif method == 'low_speed':
                aero_group = LowSpeedAero(num_nodes=num_nodes, **kwargs)

            elif method == 'tabular_low_speed':
                data_tables = [
                    key in kwargs
                    for key in ['free_aero_data', 'free_flaps_data', 'free_ground_data']
                ]

                if all(data_tables):
                    aero_group = TabularLowSpeedAero(num_nodes=num_nodes, **kwargs)
                # raise error if only some data types are provided (at this point we know
                # not all are present, now need to see if any were provided at all)
                elif any(data_tables):
                    var_msg = set(['free_aero_data', 'free_flaps_data', 'free_ground_data']) - set(
                        data_tables
                    )
                    raise UserWarning(
                        f'Low-speed tabular aerodynamics also requires {var_msg} but '
                        'these data sets were not provided.'
                    )
                else:
                    raise UserWarning(
                        'Low-speed tabular aerodynamics requires 3 data tables: free_aero_data, '
                        'free_flaps_data, and free_ground_data.'
                    )
            else:
                raise ValueError(
                    'GASP-based aero method is not one of the following: (cruise, '
                    'tabular_cruise, low_speed, tabular_low_speed)'
                )

            if solve_alpha:
                # build a group to house the aero method plus the AoA balance comp
                aero_supergroup = om.Group()
                aero_supergroup.add_subsystem(f'{method}_aero', aero_group, promotes=['*'])

                aero_supergroup.add_subsystem(
                    'solve_alpha_group',
                    SolveAlphaGroup(num_nodes=num_nodes),
                    promotes=['*'],
                )

                aero_supergroup.linear_solver = om.DirectSolver()
                newton = aero_supergroup.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
                newton.options['iprint'] = 2
                newton.options['atol'] = 1e-9
                newton.options['rtol'] = 1e-12

                # return the supergroup instead of the individual aero method group
                aero_group = aero_supergroup

        return aero_group

    # TODO DragPolar comp is unfinished and currently does nothing
    # def build_post_mission(self, aviary_inputs, phase_info, phase_mission_bus_lengths, **kwargs):
    #     aero_group = DragPolar(aviary_options=aviary_inputs),

    #     return aero_group

    def mission_inputs(self, **kwargs):
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None

        promotes = ['*']

        if self.code_origin is FLOPS:
            # FLOPS default is 'computed'
            if method is None:
                method = 'computed'

            if method == 'computed':
                promotes = [
                    Dynamic.Atmosphere.STATIC_PRESSURE,
                    Dynamic.Atmosphere.MACH,
                    Dynamic.Atmosphere.TEMPERATURE,
                    Dynamic.Vehicle.MASS,
                    'aircraft:*',
                    'mission:*',
                ]

            elif method == 'low_speed':
                promotes = [
                    Dynamic.Vehicle.ANGLE_OF_ATTACK,
                    Dynamic.Mission.ALTITUDE,
                    Dynamic.Mission.FLIGHT_PATH_ANGLE,
                    Mission.Takeoff.DRAG_COEFFICIENT_MIN,
                    Aircraft.Wing.ASPECT_RATIO,
                    Aircraft.Wing.HEIGHT,
                    Aircraft.Wing.SPAN,
                    Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                    Aircraft.Wing.AREA,
                ]

            elif method == 'tabular':
                promotes = [
                    Dynamic.Mission.ALTITUDE,
                    Dynamic.Atmosphere.MACH,
                    Dynamic.Vehicle.MASS,
                    Dynamic.Mission.VELOCITY,
                    Dynamic.Atmosphere.DENSITY,
                    'aircraft:*',
                ]

            else:
                raise ValueError(
                    'FLOPS-based aero method is not one of the following: '
                    '(computed, low_speed, tabular)'
                )

        elif self.code_origin is GASP:
            # GASP default is 'cruise'
            if method is None:
                method = 'cruise'

            if method in ('low_speed', 'tabular_low_speed'):
                promotes = [
                    '*',
                    ('airport_alt', Mission.Takeoff.AIRPORT_ALTITUDE),
                    ('CL_max_flaps', Mission.Takeoff.LIFT_COEFFICIENT_MAX),
                    (
                        'dCL_flaps_model',
                        Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
                    ),
                    (
                        'dCD_flaps_model',
                        Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
                    ),
                    ('flap_defl', Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF),
                ]

            elif method in ('cruise', 'tabular_cruise'):
                if 'output_alpha' in kwargs:
                    if kwargs['output_alpha']:
                        promotes = ['*', ('lift_req', 'weight')]

            else:
                raise ValueError(
                    'GASP-based aero method is not one of the following: (cruise, '
                    'tabular_cruise, low_speed, tabular_low_speed)'
                )

        return promotes

    def mission_outputs(self, **kwargs):
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None
        promotes = ['*']

        if self.code_origin is FLOPS:
            promotes = [Dynamic.Vehicle.DRAG, Dynamic.Vehicle.LIFT]

        elif self.code_origin is GASP:
            # GASP default is 'cruise'
            if method is None:
                method = 'cruise'

            if method in ('low_speed', 'tabular_low_speed'):
                promotes = [
                    Dynamic.Vehicle.DRAG,
                    Dynamic.Vehicle.LIFT,
                    'CL',
                    'CD',
                    'flap_factor',
                    'gear_factor',
                ]

            elif method in ('cruise', 'tabular_cruise'):
                if method == 'tabular_cruise':
                    promotes = [Dynamic.Vehicle.DRAG, Dynamic.Vehicle.LIFT]
                else:
                    if 'output_alpha' in kwargs:
                        if kwargs['output_alpha']:
                            promotes = [
                                Dynamic.Vehicle.DRAG,
                                Dynamic.Vehicle.LIFT,
                                Dynamic.Vehicle.ANGLE_OF_ATTACK,
                            ]
                    else:
                        promotes = [
                            Dynamic.Vehicle.DRAG,
                            Dynamic.Vehicle.LIFT,
                            'CL_max',
                        ]

            else:
                raise ValueError(
                    'GASP-based aero method is not one of the following: (cruise, '
                    'tabular_cruise, low_speed, tabular_low_speed)'
                )

        return promotes

    def get_parameters(self, aviary_inputs=None, **kwargs):
        """
        Return a dictionary of fixed values for the subsystem.

        Optional, used if subsystems have fixed values.

        Used in the phase builders (e.g. cruise_phase.py) when other parameters are
        added to the phase.

        This is distinct from `get_design_vars` in a nuanced way. Design variables
        are variables that are optimized by the problem that are not at the phase level.
        An example would be something that occurs in the pre-mission level of the
        problem.
        Parameters are fixed values that are held constant throughout a phase, but if
        `opt=True`, they are able to change during the optimization.

        Parameters
        ----------
        aviary_info : dict
            The AviaryValues for the aircraft problem.

        Returns
        -------
        fixed_values : dict
            A dictionary where the keys are the names of the fixed variables
            and the values are dictionaries with the following keys:

            - 'value': float or array
                The fixed value for the variable.
            - 'units': str
                The units for the fixed value (optional).
            - any additional keyword arguments required by OpenMDAO for the fixed
              variable.
        """
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None

        if method == 'external':
            return super().get_parameters()

        num_engine_type = len(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES))
        params = {}
        aero_options = kwargs

        if self.code_origin is FLOPS:
            # FLOPS default is 'computed'
            if method is None:
                method = 'computed'
            if aero_options != {}:
                # Only some methods have connectable training inputs.
                if method == 'tabular':
                    CD0_data = aero_options['CD0_data']

                    if isinstance(CD0_data, NamedValues):
                        altitude = CD0_data.get_item('altitude')[0]
                        mach = CD0_data.get_item('mach')[0]

                        n1 = altitude.size
                        n2 = mach.size
                        n1u = np.unique(altitude).size

                        if n1 > n1u:
                            # Data is free-format instead of pre-formatted.
                            n1 = n1u
                            n2 = np.unique(mach).size

                        shape = (n1, n2)

                        if (
                            aviary_inputs is not None
                            and Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR in aviary_inputs
                        ):
                            opts = {
                                'val': aviary_inputs.get_val(
                                    Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR
                                ),
                                'static_target': True,
                            }
                        else:
                            opts = {'shape': shape, 'static_target': True}

                        params[Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR] = opts

                    CDI_data = aero_options['CDI_data']

                    if isinstance(CDI_data, NamedValues):
                        mach = CDI_data.get_item('mach')[0]
                        cl = CDI_data.get_item('lift_coefficient')[0]

                        n1 = mach.size
                        n2 = cl.size
                        n1u = np.unique(mach).size

                        if n1 > n1u:
                            # Data is free-format instead of pre-formatted.
                            n1 = n1u
                            n2 = np.unique(cl).size

                        shape = (n1, n2)

                        if (
                            aviary_inputs is not None
                            and Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR in aviary_inputs
                        ):
                            opts = {
                                'val': aviary_inputs.get_val(
                                    Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR
                                ),
                                'static_target': True,
                            }
                        else:
                            opts = {'shape': shape, 'static_target': True}

                        params[Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR] = opts

            if method == 'computed':
                for var in COMPUTED_CORE_INPUTS:
                    meta = _MetaData[var]

                    val = meta['default_value']
                    if val is None:
                        val = 0.0  # _unspecified
                    units = meta['units']

                    if var in aviary_inputs:
                        try:
                            val = aviary_inputs.get_val(var, units)
                        except TypeError:
                            val = aviary_inputs.get_val(var)

                    params[var] = {'val': val, 'units': units, 'static_target': True}

                for var in ENGINE_SIZED_INPUTS:
                    meta = _MetaData[var]
                    val = meta['default_value']
                    if val is None:
                        val = [0.0]  # _unspecified
                    units = meta['units']
                    params[var] = {
                        'val': val,
                        'units': units,
                        'shape': (num_engine_type,),
                        'static_target': True,
                    }

            elif method == 'tabular':
                for var in TABULAR_CORE_INPUTS:
                    meta = _MetaData[var]

                    val = meta['default_value']
                    if val is None:
                        val = 0.0  # _unspecified
                    units = meta['units']

                    if var in aviary_inputs:
                        try:
                            val = aviary_inputs.get_val(var, units)
                        except TypeError:
                            val = aviary_inputs.get_val(var)

                    params[var] = {'val': val, 'units': units, 'static_target': True}

            elif method == 'low_speed':
                for var in LOW_SPEED_CORE_INPUTS:
                    meta = _MetaData[var]

                    val = meta['default_value']
                    if val is None:
                        val = 0.0  # _unspecified
                    units = meta['units']

                    if var in aviary_inputs:
                        try:
                            val = aviary_inputs.get_val(var, units)
                        except TypeError:
                            val = aviary_inputs.get_val(var)

                    params[var] = {'val': val, 'units': units, 'static_target': True}

        # GASP aero
        else:
            if method is None:
                # GASP default is 'cruise'
                method = 'cruise'
            try:
                solve_alpha = kwargs.pop('solve_alpha')
            except KeyError:
                solve_alpha = False

            if solve_alpha and 'tabular' in method:
                aero_data = kwargs['aero_data']

                if isinstance(aero_data, NamedValues):
                    altitude = aero_data.get_item('altitude')[0]
                    mach = aero_data.get_item('mach')[0]
                    angle_of_attack = aero_data.get_item('angle_of_attack')[0]

                    n1 = altitude.size
                    n2 = mach.size
                    n3 = angle_of_attack.size
                    n1u = np.unique(altitude).size

                    if n1 > n1u:
                        # Data is free-format instead of pre-formatted.
                        n1 = n1u
                        n2 = np.unique(mach).size
                        n3 = np.unique(angle_of_attack).size

                    shape = (n1, n2, n3)

                    if aviary_inputs is not None and Aircraft.Design.LIFT_POLAR in aviary_inputs:
                        lift_opts = {
                            'val': aviary_inputs.get_val(Aircraft.Design.LIFT_POLAR),
                            'static_target': True,
                        }
                    else:
                        lift_opts = {'shape': shape, 'static_target': True}

                    if aviary_inputs is not None and Aircraft.Design.DRAG_POLAR in aviary_inputs:
                        drag_opts = {
                            'val': aviary_inputs.get_val(Aircraft.Design.DRAG_POLAR),
                            'static_target': True,
                        }
                    else:
                        drag_opts = {'shape': shape, 'static_target': True}

                    params[Aircraft.Design.LIFT_POLAR] = lift_opts
                    params[Aircraft.Design.DRAG_POLAR] = drag_opts

            all_vars = set()
            if method == 'low_speed':
                all_vars = set(AERO_2DOF_INPUTS + AERO_LS_2DOF_INPUTS)
            elif method == 'cruise':
                all_vars = set(AERO_2DOF_INPUTS + AERO_CLEAN_2DOF_INPUTS)
            elif method == 'tabular_low_speed':
                all_vars = AERO_2DOF_TABULAR_LS_INPUTS
            elif method == 'tabular_cruise':
                all_vars = TABULAR_CORE_INPUTS
            else:
                raise ValueError(
                    'GASP-based aero method is not one of the following: (cruise, '
                    'tabular_cruise, low_speed, tabular_low_speed)'
                )

            for var in all_vars:
                # TODO only checking core metadata here!!
                meta = _MetaData[var]

                val = meta['default_value']
                if val is None:
                    val = 0.0  # _unspecified
                units = meta['units']

                if var in aviary_inputs:
                    try:
                        val = aviary_inputs.get_val(var, units)
                    except TypeError:
                        val = aviary_inputs.get_val(var)

                params[var] = {'val': val, 'units': units, 'static_target': True}

        return params

    def get_pre_mission_bus_variables(self, aviary_inputs=None):
        if self.code_origin is GASP and not self.tabular:
            return {
                'interference_independent_of_shielded_area': {
                    'mission_name': ['interference_independent_of_shielded_area'],
                    # "post_mission_name": ['interference_independent_of_shielded_area'],
                    'units': 'unitless',
                },
                'drag_loss_due_to_shielded_wing_area': {
                    'mission_name': ['drag_loss_due_to_shielded_wing_area'],
                    # "post_mission_name": ['drag_loss_due_to_shielded_wing_area'],
                    'units': 'unitless',
                },
            }
        else:
            return {}

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core aerodynamics analysis.

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        # TODO drag polar plot should go here!
        if self.code_origin is FLOPS:
            # FLOPS aero report goes here
            return
        elif self.code_origin is GASP:
            # GASP aero report goes here
            return


# Parameters for drag computation.
COMPUTED_CORE_INPUTS = [
    Aircraft.Design.BASE_AREA,
    Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
    Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
    Aircraft.Fuselage.CROSS_SECTION,
    Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
    Aircraft.Fuselage.FINENESS,
    Aircraft.Fuselage.LAMINAR_FLOW_LOWER,
    Aircraft.Fuselage.LAMINAR_FLOW_UPPER,
    Aircraft.Fuselage.LENGTH_TO_DIAMETER,
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
    Aircraft.HorizontalTail.FINENESS,
    Aircraft.HorizontalTail.LAMINAR_FLOW_LOWER,
    Aircraft.HorizontalTail.LAMINAR_FLOW_UPPER,
    Aircraft.HorizontalTail.WETTED_AREA,
    Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
    Aircraft.VerticalTail.FINENESS,
    Aircraft.VerticalTail.LAMINAR_FLOW_LOWER,
    Aircraft.VerticalTail.LAMINAR_FLOW_UPPER,
    Aircraft.VerticalTail.WETTED_AREA,
    Aircraft.Wing.AREA,
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.CHARACTERISTIC_LENGTH,
    Aircraft.Wing.FINENESS,
    Aircraft.Wing.LAMINAR_FLOW_LOWER,
    Aircraft.Wing.LAMINAR_FLOW_UPPER,
    Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
    Aircraft.Wing.SPAN_EFFICIENCY_FACTOR,
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD,
    Aircraft.Wing.WETTED_AREA,
    # Mission.Summary.GROSS_MASS,
    Mission.Design.LIFT_COEFFICIENT,
    Mission.Design.MACH,
]

TABULAR_CORE_INPUTS = [
    Aircraft.Wing.AREA,
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
]

# Parameters for low speed aero.
LOW_SPEED_CORE_INPUTS = [
    Aircraft.Wing.AREA,
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.HEIGHT,
    Aircraft.Wing.SPAN,
    Mission.Takeoff.DRAG_COEFFICIENT_MIN,
]

# These parameters are sized by number of engine models.
ENGINE_SIZED_INPUTS = [
    Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
    Aircraft.Nacelle.FINENESS,
    Aircraft.Nacelle.LAMINAR_FLOW_LOWER,
    Aircraft.Nacelle.LAMINAR_FLOW_UPPER,
    Aircraft.Nacelle.WETTED_AREA,
]

AERO_2DOF_INPUTS = [
    Aircraft.Design.CG_DELTA,
    Aircraft.Design.DRAG_COEFFICIENT_INCREMENT,  # drag increment?
    Aircraft.Design.STATIC_MARGIN,
    Aircraft.Fuselage.AVG_DIAMETER,
    Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT,
    Aircraft.Fuselage.FORM_FACTOR,
    Aircraft.Fuselage.LENGTH,
    Aircraft.Fuselage.WETTED_AREA,
    Aircraft.HorizontalTail.AREA,
    Aircraft.HorizontalTail.AVERAGE_CHORD,
    Aircraft.HorizontalTail.FORM_FACTOR,
    Aircraft.HorizontalTail.MOMENT_RATIO,
    Aircraft.HorizontalTail.SPAN,
    Aircraft.HorizontalTail.SWEEP,
    Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION,
    Aircraft.Nacelle.AVG_LENGTH,
    Aircraft.Nacelle.FORM_FACTOR,
    Aircraft.Nacelle.SURFACE_AREA,
    Aircraft.Strut.AREA_RATIO,
    Aircraft.Strut.CHORD,
    Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR,
    Aircraft.VerticalTail.AREA,
    Aircraft.VerticalTail.AVERAGE_CHORD,
    Aircraft.VerticalTail.FORM_FACTOR,
    Aircraft.VerticalTail.SPAN,
    Aircraft.Wing.AVERAGE_CHORD,
    Aircraft.Wing.AREA,
    Aircraft.Wing.ASPECT_RATIO,
    Aircraft.Wing.FORM_FACTOR,
    Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR,
    Aircraft.Wing.MAX_THICKNESS_LOCATION,
    Aircraft.Wing.MIN_PRESSURE_LOCATION,
    Aircraft.Wing.SPAN,
    Aircraft.Wing.SWEEP,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
    Aircraft.Wing.VERTICAL_MOUNT_LOCATION,
    Aircraft.Wing.ZERO_LIFT_ANGLE,
]

AERO_2DOF_TABULAR_LS_INPUTS = [Aircraft.Wing.SPAN, Aircraft.Wing.HEIGHT]

AERO_LS_2DOF_INPUTS = [
    Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT,
    Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT,
    Mission.Takeoff.LIFT_COEFFICIENT_MAX,
    Aircraft.Wing.HEIGHT,
    Aircraft.Wing.FLAP_CHORD_RATIO,
    Mission.Design.GROSS_MASS,
]

AERO_CLEAN_2DOF_INPUTS = [
    Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT,  # super drag shift?
    Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP,
    Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
    Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
    Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
]
