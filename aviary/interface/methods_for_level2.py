import csv
import warnings
from packaging import version
import inspect
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np

import dymos as dm
from dymos.utils.misc import _unspecified

import openmdao.api as om
from openmdao.utils.units import convert_units
from openmdao.utils.units import valid_units

from aviary.constants import GRAV_ENGLISH_LBM, RHO_SEA_LEVEL_ENGLISH
from aviary.mission.flops_based.phases.build_landing import Landing
from aviary.mission.flops_based.phases.build_takeoff import Takeoff
from aviary.mission.flops_based.phases.climb_phase import Climb
from aviary.mission.flops_based.phases.cruise_phase import Cruise
from aviary.mission.flops_based.phases.descent_phase import Descent
from aviary.mission.flops_based.phases.simple_energy_phase import EnergyPhase
from aviary.mission.gasp_based.ode.groundroll_ode import GroundrollODE
from aviary.mission.gasp_based.ode.params import ParamPort
from aviary.mission.gasp_based.ode.unsteady_solved.unsteady_solved_ode import \
    UnsteadySolvedODE
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.mission.gasp_based.phases.time_integration_phases import SGMCruise
from aviary.mission.gasp_based.phases.accel_phase import get_accel
from aviary.mission.gasp_based.phases.ascent_phase import get_ascent
from aviary.mission.gasp_based.phases.climb_phase import get_climb
from aviary.mission.gasp_based.phases.desc_phase import get_descent
from aviary.mission.gasp_based.phases.groundroll_phase import get_groundroll
from aviary.mission.gasp_based.phases.landing_group import LandingSegment
from aviary.mission.gasp_based.phases.rotation_phase import get_rotation
from aviary.mission.gasp_based.phases.taxi_group import TaxiSegment
from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.mission.gasp_based.polynomial_fit import PolynomialFit
from aviary.subsystems.premission import CorePreMission
from aviary.mission.gasp_based.ode.breguet_cruise_ode import BreguetCruiseODESolution
from aviary.utils.functions import set_aviary_initial_values, Null, create_opts2vals, add_opts2vals, promote_aircraft_and_mission_vars
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.preprocessors import preprocess_crewpayload
from aviary.interface.utils.check_phase_info import check_phase_info
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.functions import setup_trajectory_params, override_aviary_vars
from aviary.variable_info.variables import Aircraft, Mission, Dynamic
from aviary.variable_info.enums import AnalysisScheme, ProblemType, SpeedType, AlphaModes
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables_in import VariablesIn

from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.merge_variable_metadata import merge_meta_data
from aviary.variable_info.variables import Aircraft, Mission, Dynamic

from aviary.interface.default_phase_info.gasp_fiti import create_gasp_based_ascent_phases, create_gasp_based_descent_phases
from aviary.mission.gasp_based.idle_descent_estimation import descent_range_and_fuel
from aviary.mission.flops_based.phases.phase_utils import get_initial


def wrapped_convert_units(val_unit_tuple, new_units):
    """
    Wrapper for OpenMDAO's convert_units function.

    Parameters
    ----------
    val_unit_tuple : tuple
        Tuple of the form (value, units) where value is a float and units is a
        string.
    new_units : string
        New units to convert to.

    Returns
    -------
    float
        Value converted to new units.
    """
    value, units = val_unit_tuple

    # can't convert units on None; return None
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        return [convert_units(v, units, new_units) for v in value]
    else:
        return convert_units(value, units, new_units)


class PreMissionGroup(om.Group):
    def configure(self):
        external_outputs = promote_aircraft_and_mission_vars(self)

        statics = self.core_subsystems
        override_aviary_vars(statics, statics.options["aviary_options"],
                             external_overrides=external_outputs,
                             manual_overrides=statics.manual_overrides)


class PostMissionGroup(om.Group):
    def configure(self):
        promote_aircraft_and_mission_vars(self)


class AviaryProblem(om.Problem):
    """
    Main class for instantiating, formulating, and solving Aviary problems.

    On a basic level, this problem object is all the conventional user needs
    to interact with. Looking at the three "levels" of use cases, from simplest
    to most complicated, we have:

    Level 1: users interact with Aviary through input files (.csv or .yaml, TBD)
    Level 2: users interact with Aviary through a Python interface
    Level 3: users can modify Aviary's workings through Python and OpenMDAO

    This Problem object is simply a specialized OpenMDAO Problem that has
    additional methods to help users create and solve Aviary problems.
    """

    def __init__(self, phase_info, mission_method, mass_method, analysis_scheme=AnalysisScheme.COLLOCATION, **kwargs):
        super().__init__(**kwargs)

        self.timestamp = datetime.now()

        self.model = om.Group()
        self.pre_mission = PreMissionGroup()
        self.post_mission = PostMissionGroup()

        self.aviary_inputs = None

        phase_info = deepcopy(phase_info)

        for phase_name in phase_info:
            if 'external_subsystems' not in phase_info[phase_name]:
                phase_info[phase_name]['external_subsystems'] = []

        self.phase_info = phase_info
        self.traj = None

        self.mission_method = mission_method
        self.mass_method = mass_method
        self.analysis_scheme = analysis_scheme

        # set up core subsystems
        if self.mission_method == "solved":
            everything_else_origin = "GASP"
        elif self.mission_method == "simple":
            everything_else_origin = "FLOPS"
        else:
            everything_else_origin = self.mission_method

        prop = CorePropulsionBuilder('core_propulsion')
        mass = CoreMassBuilder('core_mass', code_origin=mass_method)
        aero = CoreAerodynamicsBuilder(
            'core_aerodynamics', code_origin=everything_else_origin)

        # TODO These values are currently hardcoded, in future should come from user
        both_geom = False
        code_origin_to_prioritize = None

        # which geometry methods should be used, or both?
        geom_code_origin = None
        if (everything_else_origin == 'FLOPS') and (mass_method == 'FLOPS'):
            geom_code_origin = 'FLOPS'
        elif (everything_else_origin == 'GASP') and (mass_method == 'GASP'):
            geom_code_origin = 'GASP'
        else:
            both_geom = True

        # which geometry method gets prioritized in case of conflicting outputs
        if not code_origin_to_prioritize:
            if everything_else_origin == 'GASP':
                code_origin_to_prioritize = 'GASP'
            elif everything_else_origin == 'FLOPS':
                code_origin_to_prioritize = 'FLOPS'

        geom = CoreGeometryBuilder('core_geometry',
                                   code_origin=geom_code_origin,
                                   use_both_geometries=both_geom,
                                   code_origin_to_prioritize=code_origin_to_prioritize)

        self.core_subsystems = {'propulsion': prop,
                                'geometry': geom,
                                'mass': mass,
                                'aerodynamics': aero}

        if 'pre_mission' in self.phase_info:
            self.pre_mission_info = self.phase_info.pop('pre_mission')
        else:
            self.pre_mission_info = {'include_takeoff': True,
                                     'external_subsystems': []}

        if 'post_mission' in self.phase_info:
            self.post_mission_info = self.phase_info.pop('post_mission')
        else:
            self.post_mission_info = {'include_landing': True,
                                      'external_subsystems': [],
                                      'constrain_range': False}

    def load_inputs(self, input_filename, engine_builder=None):
        """
        This method loads the aviary_values inputs and options that the
        user specifies. They could specify files to load and values to
        replace here as well.

        This method is not strictly necessary; a user could also supply
        an AviaryValues object of their own.
        """
        self.engine_builder = engine_builder
        self.aviary_inputs, self.initial_guesses = create_vehicle(input_filename)

        aviary_inputs = self.aviary_inputs

        if self.mission_method == "GASP":
            aviary_inputs.set_val(Mission.Summary.CRUISE_MASS_FINAL,
                                  val=self.initial_guesses['cruise_mass_final'], units='lbm')
            aviary_inputs.set_val(Mission.Summary.GROSS_MASS,
                                  val=self.initial_guesses['actual_takeoff_mass'], units='lbm')

            self.cruise_mass_final = aviary_inputs.get_val(
                Mission.Summary.CRUISE_MASS_FINAL, units='lbm')
            self.target_range = aviary_inputs.get_val(
                Mission.Design.RANGE, units='NM')
            self.cruise_mach = aviary_inputs.get_val(Mission.Design.MACH)

        # TODO optionally accept which subsystems to load from phase_info
        subsystems = self.core_subsystems
        default_mission_subsystems = [
            subsystems['aerodynamics'], subsystems['propulsion']]
        self.ode_args = dict(aviary_options=aviary_inputs,
                             core_subsystems=default_mission_subsystems)

        if engine_builder is None:
            engine = EngineDeck(options=aviary_inputs)
        else:
            engine = engine_builder

        preprocess_propulsion(aviary_inputs, [engine])

        self._update_metadata_from_subsystems()

        self.aviary_inputs = aviary_inputs
        return aviary_inputs

    def _update_metadata_from_subsystems(self):
        self.meta_data = BaseMetaData.copy()

        variables_to_pop = []

        # loop through phase_info and external subsystems
        for phase_name in self.phase_info:
            external_subsystems = self._get_all_subsystems(
                self.phase_info[phase_name]['external_subsystems'])
            for subsystem in external_subsystems:
                meta_data = subsystem.meta_data.copy()

                state_info = subsystem.get_states()
                for state in state_info:
                    variables_to_pop.append(state)
                    variables_to_pop.append(state_info[state]['rate_source'])

                arg_spec = inspect.getfullargspec(subsystem.get_controls)
                if 'phase_name' in arg_spec.args:
                    control_dicts = subsystem.get_controls(
                        phase_name=phase_name)
                else:
                    control_dicts = subsystem.get_controls()

                for control_name, control_dict in control_dicts.items():
                    variables_to_pop.append(control_name)

                for output in subsystem.get_outputs():
                    variables_to_pop.append(output)

                for parameter in subsystem.get_parameters():
                    variables_to_pop.append(parameter)

                self.meta_data = merge_meta_data([self.meta_data, meta_data])

        variables_to_pop = list(set(variables_to_pop))

        for variable in variables_to_pop:
            if variable in self.meta_data:
                self.meta_data.pop(variable)

    def check_inputs(self):
        """
        This method checks the user-supplied input values for any potential problems.
        These problems include variable names that are not recognized in Aviary,
        conflicting options or values, or units mismatching.
        """
        check_phase_info(self.phase_info, self.mission_method)

        for phase_name in self.phase_info:
            for external_subsystem in self.phase_info[phase_name]['external_subsystems']:
                self.aviary_inputs = external_subsystem.preprocess_inputs(
                    self.aviary_inputs)

        # TODO find preprocessors a permanent home
        # PREPROCESSORS #
        # Fill in anything missing in the options with computed defaults.
        preprocess_crewpayload(self.aviary_inputs)

    def add_pre_mission_systems(self):
        """
        Add pre-mission systems to the Aviary problem. These systems are executed before the mission
        and are also known as the "pre_mission" group.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds various subsystems
        to the aircraft model. For the `FLOPS` mission model, a takeoff phase is added using the Takeoff class
        with the number of engines and airport altitude specified. For the `GASP` mission model, three subsystems
        are added: a TaxiSegment subsystem, an ExecComp to calculate the time to initiate gear and flaps,
        and an ExecComp to calculate the speed at which to initiate rotation. All subsystems are promoted with
        aircraft and mission inputs and outputs as appropriate.

        A user can override this method with their own pre-mission systems as desired.
        """
        pre_mission = self.pre_mission
        self.model.add_subsystem('pre_mission', pre_mission,
                                 promotes_inputs=['aircraft:*', 'mission:*'],
                                 promotes_outputs=['aircraft:*', 'mission:*'],)

        if 'linear_solver' in self.pre_mission_info:
            pre_mission.linear_solver = self.pre_mission_info['linear_solver']

        if 'nonlinear_solver' in self.pre_mission_info:
            pre_mission.nonlinear_solver = self.pre_mission_info['nonlinear_solver']

        self._add_premission_external_subsystems()

        subsystems = self.core_subsystems
        default_subsystems = [subsystems['propulsion'],
                              subsystems['geometry'],
                              subsystems['aerodynamics'],
                              subsystems['mass'],]

        pre_mission.add_subsystem(
            'core_subsystems',
            CorePreMission(
                aviary_options=self.aviary_inputs,
                subsystems=default_subsystems,
                process_overrides=False,
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        if not self.pre_mission_info['include_takeoff']:
            return

        # Check for 'GASP' mission method
        if self.mission_method == "GASP":
            self._add_gasp_takeoff_systems()

        # Check for 'FLOPS' mission method
        elif self.mission_method == "FLOPS" or self.mission_method == "simple":
            self._add_flops_takeoff_systems()

    def _add_flops_takeoff_systems(self):
        # Initialize takeoff options
        takeoff_options = Takeoff(
            airport_altitude=0.,  # ft
            num_engines=self.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)
        )

        # Build and add takeoff subsystem
        takeoff = takeoff_options.build_phase(False)
        self.model.add_subsystem(
            'takeoff', takeoff, promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'])

    def _add_gasp_takeoff_systems(self):
        # Create options to values
        OptionsToValues = create_opts2vals(
            [Aircraft.CrewPayload.NUM_PASSENGERS,
                Mission.Design.CRUISE_ALTITUDE, ])
        add_opts2vals(self.model, OptionsToValues, self.aviary_inputs)

        # Add thrust-to-weight ratio subsystem
        self.model.add_subsystem(
            'tw_ratio',
            om.ExecComp(
                f'TW_ratio = Fn_SLS / (takeoff_mass * {GRAV_ENGLISH_LBM})',
                TW_ratio={'units': "unitless"},
                Fn_SLS={'units': 'lbf'},
                takeoff_mass={'units': 'lbm'},
            ),
            promotes_inputs=[('Fn_SLS', Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST),
                             ('takeoff_mass', Mission.Summary.GROSS_MASS)],
            promotes_outputs=[('TW_ratio', Aircraft.Design.THRUST_TO_WEIGHT_RATIO)],
        )

        self.cruise_alt = self.aviary_inputs.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')

        # Add taxi subsystem
        self.model.add_subsystem(
            "taxi", TaxiSegment(**(self.ode_args)),
            promotes_inputs=['aircraft:*', 'mission:*'],
        )

        if self.analysis_scheme is AnalysisScheme.COLLOCATION:
            # Add event transformation subsystem
            self.model.add_subsystem(
                "event_xform",
                om.ExecComp(
                    ["t_init_gear=m*tau_gear+b", "t_init_flaps=m*tau_flaps+b"], units="s"
                ),
                promotes_inputs=[
                    "tau_gear",
                    "tau_flaps",
                    ("m", Mission.Takeoff.ASCENT_DURATION),
                    ("b", Mission.Takeoff.ASCENT_T_INTIIAL),
                ],
                promotes_outputs=["t_init_gear", "t_init_flaps"],
            )

        # Calculate speed at which to initiate rotation
        self.model.add_subsystem(
            "vrot",
            om.ExecComp(
                "Vrot = ((2 * mass * g) / (rho * wing_area * CLmax))**0.5 + dV1 + dVR",
                Vrot={"units": "ft/s"},
                mass={"units": "lbm"},
                CLmax={"units": "unitless"},
                g={"units": "lbf/lbm", "val": GRAV_ENGLISH_LBM},
                rho={"units": "slug/ft**3", "val": RHO_SEA_LEVEL_ENGLISH},
                wing_area={"units": "ft**2"},
                dV1={
                    "units": "ft/s",
                    "desc": "Increment of engine failure decision speed above stall",
                },
                dVR={
                    "units": "ft/s",
                    "desc": "Increment of takeoff rotation speed above engine failure "
                    "decision speed",
                },
            ),
            promotes_inputs=[
                ("wing_area", Aircraft.Wing.AREA),
                ("dV1", Mission.Takeoff.DECISION_SPEED_INCREMENT),
                ("dVR", Mission.Takeoff.ROTATION_SPEED_INCREMENT),
                ("CLmax", Mission.Takeoff.LIFT_COEFFICIENT_MAX),
            ],
            promotes_outputs=[('Vrot', Mission.Takeoff.ROTATION_VELOCITY)]
        )

    def _add_premission_external_subsystems(self):
        """
        This private method adds each external subsystem to the pre-mission subsystem and
        a mass component that captures external subsystem masses for use in mass buildups.

        Firstly, the method iterates through all external subsystems in the pre-mission
        information. For each subsystem, it builds the pre-mission instance of the
        subsystem.

        Secondly, the method collects the mass names of the added subsystems. This
        expression is then used to define an ExecComp (a component that evaluates a
        simple equation given input values).

        The method promotes the input and output of this ExecComp to the top level of the
        pre-mission object, allowing this calculated subsystem mass to be accessed
        directly from the pre-mission object.
        """

        mass_names = []
        # Loop through all the phases in this subsystem.
        for external_subsystem in self.pre_mission_info['external_subsystems']:
            # Get all the subsystem builders for this phase.
            subsystem_premission = external_subsystem.build_pre_mission(
                self.aviary_inputs)

            if subsystem_premission is not None:
                self.pre_mission.add_subsystem(external_subsystem.name,
                                               subsystem_premission)

                mass_names.extend(external_subsystem.get_mass_names())

        if mass_names:
            formatted_names = []
            for name in mass_names:
                formatted_name = name.replace(':', '_')
                formatted_names.append(formatted_name)

            # Define the expression for computing the sum of masses
            expr = 'subsystem_mass = ' + ' + '.join(formatted_names)

            promotes_inputs_list = [(formatted_name, original_name)
                                    for formatted_name, original_name in zip(formatted_names, mass_names)]

            # Create the ExecComp
            self.pre_mission.add_subsystem('external_comp_sum', om.ExecComp(expr, units='kg'),
                                           promotes_inputs=promotes_inputs_list,
                                           promotes_outputs=[
                ('subsystem_mass', Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS)])

    def _get_gasp_phase(self, phase_name):
        # Get the phase options for the specified phase name
        phase_options = self.phase_info[phase_name]

        if 'cruise' in phase_name:
            phase = dm.AnalyticPhase(
                ode_class=BreguetCruiseODESolution,
                ode_init_kwargs=self.ode_args,
                num_nodes=5,
            )

            # Time here is really the independent variable through which we are integrating.
            # In the case of the Breguet Range ODE, it's mass.
            # We rely on mass being monotonically non-increasing across the phase.
            phase.set_time_options(
                name='mass',
                fix_initial=False,
                fix_duration=False,
                units="lbm",
                targets="mass",
                initial_bounds=(10.e3, 500_000),
                initial_ref=100_000,
                duration_bounds=(-50000, -10),
                duration_ref=50000,
            )

            phase.add_parameter(Dynamic.Mission.ALTITUDE, opt=False,
                                val=self.phase_info['climb2']['final_alt'][0], units=self.phase_info['climb2']['final_alt'][1])
            phase.add_parameter(Dynamic.Mission.MACH, opt=False,
                                val=self.phase_info['climb2']['mach_cruise'])
            phase.add_parameter("initial_distance", opt=False, val=0.0,
                                units="NM", static_target=True)
            phase.add_parameter("initial_time", opt=False, val=0.0,
                                units="s", static_target=True)

            phase.add_timeseries_output("time", units="s")

        else:
            # Create a Radau transcription scheme object with the specified num_segments and order
            transcription = dm.Radau(
                num_segments=phase_options['num_segments'],
                order=phase_options['order'],
                compressed=True,
                solve_segments=False)

            # Create a dictionary of phase functions
            phase_functions = {
                'groundroll': get_groundroll,
                'rotation': get_rotation,
                'ascent': get_ascent,
                'accel': get_accel
            }

            # Set the phase function based on the phase name
            if 'climb' in phase_name:
                phase_functions[phase_name] = get_climb
            elif 'desc' in phase_name:
                phase_functions[phase_name] = get_descent

            # Get the phase function corresponding to the phase name
            phase_func = phase_functions.get(phase_name)

            # Calculate the phase by calling the phase function
            # with the transcription object and remaining phase options
            trimmed_phase_options = {k: v for k, v in phase_options.items(
            ) if k not in ['num_segments', 'order', 'initial_guesses', 'throttle_setting', 'external_subsystems']}

            # define expected units for each phase option
            expected_units = {
                'alt': 'ft',
                'mass': 'lbm',
                'distance': 'ft',
                'time': 's',
                'duration': 's',
                'initial': 's',
                'EAS': 'kn',
                'TAS': 'kn',
                'angle': 'deg',
                'pitch': 'deg',
                'normal': 'lbf',
                'final_alt': 'ft',
                'required_available_climb_rate': 'ft/min',
            }

            if phase_name in ['accel', 'climb1', 'climb2', 'desc1', 'desc2']:
                expected_units['distance'] = 'NM'

            # loop through all trimmed_phase_options and call wrapped_convert_units with the correct expected units
            for key, value in trimmed_phase_options.items():
                for expected_key, expected_unit in expected_units.items():
                    if key.startswith(expected_key):
                        trimmed_phase_options[key] = wrapped_convert_units(
                            value, expected_unit)

            phase = phase_func(
                ode_args=self.ode_args,
                transcription=transcription,
                **trimmed_phase_options)

            phase.add_control(
                Dynamic.Mission.THROTTLE, targets=Dynamic.Mission.THROTTLE, units='unitless',
                opt=False, lower=0.0, upper=1.0
            )

        phase.timeseries_options['use_prefix'] = True

        return phase

    def _add_groundroll_eq_constraint(self, phase):
        """
        Add an equality constraint to the problem to ensure that the TAS at the end of the
        groundroll phase is equal to the rotation velocity at the start of the rotation phase.
        """
        self.model.add_subsystem(
            "groundroll_boundary",
            om.EQConstraintComp(
                "TAS",
                eq_units="ft/s",
                normalize=True,
                add_constraint=True,
            ),
        )
        self.model.connect(Mission.Takeoff.ROTATION_VELOCITY,
                           "groundroll_boundary.rhs:TAS")
        self.model.connect(
            "traj.groundroll.states:TAS",
            "groundroll_boundary.lhs:TAS",
            src_indices=[-1],
            flat_src_indices=True,
        )

        ascent_tx = phase.options["transcription"]
        ascent_num_nodes = ascent_tx.grid_data.num_nodes
        self.model.add_subsystem(
            "h_fit",
            PolynomialFit(N_cp=ascent_num_nodes),
            promotes_inputs=["t_init_gear", "t_init_flaps"],
        )

    def _get_flops_phase(self, phase_name, phase_idx):
        phase_options = self.phase_info[phase_name]

        fix_duration = phase_options['user_options'].pop('fix_duration')

        # TODO optionally accept which subsystems to load from phase_info
        subsystems = self.core_subsystems
        default_mission_subsystems = [
            subsystems['aerodynamics'], subsystems['propulsion']]

        if self.mission_method == "FLOPS" or self.mission_method == "simple":
            if self.mission_method == "simple":
                climb_builder = EnergyPhase
                cruise_builder = EnergyPhase
                descent_builder = EnergyPhase
            else:
                climb_builder = Climb
                cruise_builder = Cruise
                descent_builder = Descent

            if phase_name == 'climb':
                phase_object = climb_builder.from_phase_info(
                    phase_name, phase_options, default_mission_subsystems, meta_data=self.meta_data)

            elif phase_name == 'cruise':
                phase_object = cruise_builder.from_phase_info(
                    phase_name, phase_options, default_mission_subsystems, meta_data=self.meta_data)

            elif phase_name == 'descent':
                phase_object = descent_builder.from_phase_info(
                    phase_name, phase_options, default_mission_subsystems, meta_data=self.meta_data)
            else:
                phase_object = EnergyPhase.from_phase_info(
                    phase_name, phase_options, default_mission_subsystems, meta_data=self.meta_data)

            phase = phase_object.build_phase(aviary_options=self.aviary_inputs)

            # TODO: add logic to filter which phases get which controls.
            # right now all phases get all controls added from every subsystem.
            # for example, we might only want ELECTRIC_SHAFT_POWER applied during the climb phase.
            all_subsystems = self._get_all_subsystems(
                phase_options['external_subsystems'])

            # loop through all_subsystems and call `get_controls` on each subsystem
            for subsystem in all_subsystems:
                # add the controls from the subsystems to each phase
                arg_spec = inspect.getfullargspec(subsystem.get_controls)
                if 'phase_name' in arg_spec.args:
                    control_dicts = subsystem.get_controls(
                        phase_name=phase_name)
                else:
                    control_dicts = subsystem.get_controls()
                for control_name, control_dict in control_dicts.items():
                    phase.add_control(control_name, **control_dict)

            user_options = AviaryValues(phase_options.get('user_options', ()))

            fix_initial = user_options.get_val("fix_initial")
            if "fix_initial_time" in user_options:
                fix_initial_time = user_options.get_val("fix_initial_time")
            else:
                fix_initial_time = get_initial(fix_initial, "time", True)

            input_initial = False
            if self.mission_method == "simple":
                user_options.set_val('initial_ref', 10., 'min')
                duration_bounds = user_options.get_val("duration_bounds", 'min')
                user_options.set_val(
                    'duration_ref', (duration_bounds[0] + duration_bounds[1]) / 2., 'min')
                if phase_idx > 0:
                    input_initial = True

            if fix_initial_time or input_initial:
                phase.set_time_options(
                    fix_initial=fix_initial_time, fix_duration=fix_duration, units='s',
                    duration_bounds=user_options.get_val("duration_bounds", 's'),
                    duration_ref=user_options.get_val("duration_ref", 's'),
                )
            elif phase_name == 'descent':  # TODO: generalize this logic for all phases
                phase.set_time_options(
                    fix_initial=False, fix_duration=False, units='s',
                    duration_bounds=user_options.get_val("duration_bounds", 's'),
                    duration_ref=user_options.get_val("duration_ref", 's'),
                    initial_bounds=user_options.get_val("initial_bounds", 's'),
                    initial_ref=user_options.get_val("initial_ref", 's'),
                )
            else:  # TODO: figure out how to handle this now that fix_initial is dict
                phase.set_time_options(
                    fix_initial=fix_initial, fix_duration=fix_duration, units='s',
                    duration_bounds=user_options.get_val("duration_bounds", 's'),
                    duration_ref=user_options.get_val("duration_ref", 's'),
                    initial_bounds=user_options.get_val("initial_bounds", 's'),
                    initial_ref=user_options.get_val("initial_ref", 's'),
                )

            phase.timeseries_options['use_prefix'] = True

            return phase

    def _get_solved_phase(self, phase_name):
        phase_options = self.phase_info[phase_name]

        takeoff_mass = self.aviary_inputs.get_val(
            Mission.Design.GROSS_MASS, units='lbm')
        climb_mach = 0.8
        solve_segments = False

        if phase_name == "groundroll":
            groundroll_trans = dm.Radau(
                num_segments=phase_options['num_segments'], order=3, compressed=True, solve_segments="forward",
            )

            phase = dm.Phase(
                ode_class=GroundrollODE,
                ode_init_kwargs=self.ode_args,
                transcription=groundroll_trans,
            )

            phase.set_time_options(fix_initial=True, fix_duration=False,
                                   units="kn", name="TAS",
                                   duration_bounds=wrapped_convert_units(
                                       phase_options['duration_bounds'], 'kn'),
                                   duration_ref=wrapped_convert_units(
                                       phase_options['duration_ref'], 'kn'),
                                   initial_ref=wrapped_convert_units(
                                       phase_options['initial_ref'], 'kn'),
                                   )

            phase.set_state_options("time", rate_source="dt_dv", units="s",
                                    fix_initial=True, fix_final=False, ref=1., defect_ref=1.)

            phase.set_state_options("mass", rate_source="dmass_dv",
                                    fix_initial=True, fix_final=False, lower=1, upper=195_000, ref=takeoff_mass, defect_ref=takeoff_mass)

            phase.set_state_options(Dynamic.Mission.RANGE, rate_source="over_a",
                                    fix_initial=True, fix_final=False, lower=0, upper=2000., ref=1.e2, defect_ref=1.e2)

            phase.add_parameter("t_init_gear", units="s",
                                static_target=True, opt=False, val=32.3)
            phase.add_parameter("t_init_flaps", units="s",
                                static_target=True, opt=False, val=44.0)
            phase.add_parameter("wing_area", units="ft**2",
                                static_target=True, opt=False, val=1370)

        else:

            trans = dm.Radau(num_segments=phase_options['num_segments'], order=3,
                             compressed=True, solve_segments=solve_segments)

            ode_args = dict(
                input_speed_type=phase_options['input_speed_type'],
                clean=phase_options['clean'],
                ground_roll=phase_options['ground_roll'],
                aviary_options=self.aviary_inputs,
                core_subsystems=self.ode_args['core_subsystems'],
                include_param_comp=True,
                balance_throttle=False
            )

            phase = dm.Phase(
                ode_class=UnsteadySolvedODE,
                ode_init_kwargs=ode_args,
                transcription=trans,
            )

            phase.add_parameter(
                Dynamic.Mission.THROTTLE,
                opt=False,
                units="unitless",
                val=phase_options['throttle_setting'],
                static_target=False)

            phase.set_time_options(fix_initial=False, fix_duration=False,
                                   units="range_units", name=Dynamic.Mission.RANGE,
                                   duration_bounds=wrapped_convert_units(
                                       phase_options['duration_bounds'], "range_units"),
                                   duration_ref=wrapped_convert_units(
                                       phase_options['duration_ref'], "range_units"),
                                   initial_bounds=wrapped_convert_units(
                                       phase_options['initial_bounds'], "range_units"),
                                   initial_ref=wrapped_convert_units(
                                       phase_options['initial_ref'], "range_units"),
                                   )

            if phase_name == "cruise" or phase_name == "descent":
                time_ref = 1.e4
            else:
                time_ref = 100.

            phase.set_state_options("time", rate_source="dt_dr", targets=['t_curr'] if 'retract' in phase_name else [],
                                    fix_initial=False, fix_final=False, ref=time_ref, defect_ref=time_ref * 1.e2)

            phase.set_state_options("mass", rate_source="dmass_dr",
                                    fix_initial=False, fix_final=False, ref=170.e3, defect_ref=170.e5,
                                    val=170.e3, units='lbm', lower=10.e3)

            phase.add_parameter("wing_area", units="ft**2",
                                static_target=True, opt=False, val=1370)

            if 'climb_at_constant_TAS' in phase_name or 'ascent' in phase_name:
                phase.add_parameter(
                    "t_init_gear", units="s", static_target=True, opt=False, val=100)
                phase.add_parameter(
                    "t_init_flaps", units="s", static_target=True, opt=False, val=100)

            if 'rotation' in phase_name:
                phase.add_polynomial_control("TAS",
                                             order=phase_options['control_order'],
                                             units="kn", val=200.0,
                                             opt=phase_options['opt'], lower=1, upper=500, ref=250)

                phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                             order=phase_options['control_order'],
                                             fix_initial=False,
                                             rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                             opt=False, upper=40.e3, ref=30.e3, lower=-1.)

                phase.add_polynomial_control("alpha",
                                             order=phase_options['control_order'],
                                             lower=-4, upper=15,
                                             units='deg', ref=10.,
                                             val=[0., 5.],
                                             opt=phase_options['opt'])
            else:
                if 'constant_EAS' in phase_name:
                    fixed_EAS = self.phase_info[phase_name]['fixed_EAS']
                    phase.add_parameter("EAS", units="kn", val=fixed_EAS)
                elif 'constant_mach' in phase_name:
                    phase.add_parameter(
                        Dynamic.Mission.MACH,
                        units="unitless",
                        val=climb_mach)
                elif 'cruise' in phase_name:
                    self.cruise_mach = self.aviary_inputs.get_val(Mission.Design.MACH)
                    phase.add_parameter(
                        Dynamic.Mission.MACH, units="unitless", val=self.cruise_mach)
                else:
                    phase.add_polynomial_control("TAS",
                                                 order=phase_options['control_order'],
                                                 fix_initial=False,
                                                 units="kn", val=200.0,
                                                 opt=True, lower=1, upper=500, ref=250)

                phase.add_polynomial_control(Dynamic.Mission.ALTITUDE,
                                             order=phase_options['control_order'],
                                             units="ft", val=0.,
                                             fix_initial=False,
                                             rate_targets=['dh_dr'], rate2_targets=['d2h_dr2'],
                                             opt=phase_options['opt'], upper=40.e3, ref=30.e3, lower=-1.)

        phase.timeseries_options['use_prefix'] = True

        return phase

    def add_phases(self, phase_info_parameterization=None):
        """
        Add the mission phases to the problem trajectory based on the user-specified
        phase_info dictionary.

        Parameters
        ----------
        phase_info_parameterization (function, optional): A function that takes in the phase_info dictionary
            and aviary_inputs and returns modified aviary_inputs. Defaults to None.

        Returns
        -------
        traj: The Dymos Trajectory object containing the added mission phases.
        """
        if phase_info_parameterization is not None:
            self.phase_info = phase_info_parameterization(self.phase_info,
                                                          self.aviary_inputs)

        phase_info = self.phase_info

        phases = list(phase_info.keys())

        if self.analysis_scheme is AnalysisScheme.COLLOCATION:
            traj = self.model.add_subsystem('traj', dm.Trajectory())

        elif self.analysis_scheme is AnalysisScheme.SHOOTING:
            initial_mass = self.aviary_inputs.get_val(Mission.Summary.GROSS_MASS, 'lbm')

            ascent_phases = create_gasp_based_ascent_phases(
                self.ode_args,
                cruise_alt=self.cruise_alt,
                cruise_mach=self.cruise_mach)

            descent_phases = create_gasp_based_descent_phases(
                self.ode_args,
                cruise_alt=self.cruise_alt,
                cruise_mach=self.cruise_mach)

            descent_estimation = descent_range_and_fuel(
                phases=descent_phases,
                initial_mass=initial_mass,
                cruise_alt=self.cruise_alt,
                cruise_mach=self.cruise_mach)

            estimated_descent_range = descent_estimation['refined_guess']['distance_flown']
            end_of_cruise_range = self.target_range - estimated_descent_range

            estimated_descent_fuel = descent_estimation['refined_guess']['fuel_burned']

            cruise_kwargs = dict(
                input_speed_type=SpeedType.MACH,
                input_speed_units="unitless",
                ode_args=self.ode_args,
                alpha_mode=AlphaModes.REQUIRED_LIFT,
                simupy_args=dict(
                    DEBUG=True,
                    blocked_state_names=['engine.nox', 'nox',
                                         'TAS', Dynamic.Mission.FLIGHT_PATH_ANGLE],
                ),
            )
            cruise_vals = {
                'mach': {'val': self.cruise_mach, 'units': cruise_kwargs['input_speed_units']},
                'descent_fuel': {'val': estimated_descent_fuel, 'units': 'lbm'},
            }

            phases = {
                **ascent_phases,
                'cruise': {
                    'ode': SGMCruise(**cruise_kwargs),
                    'vals_to_set': cruise_vals,
                },
                **descent_phases,
            }
            full_traj = FlexibleTraj(
                Phases=phases,
                traj_final_state_output=[
                    Dynamic.Mission.MASS,
                    Dynamic.Mission.DISTANCE,
                ],
                traj_initial_state_input=[
                    Dynamic.Mission.MASS,
                    Dynamic.Mission.DISTANCE,
                    Dynamic.Mission.ALTITUDE,
                ],
                traj_event_trigger_input=[
                    # specify ODE, output_name, with units that SimuPyProblem expects
                    # assume event function is of form ODE.output_name - value
                    # third key is event_idx associated with input
                    (phases['groundroll']['ode'], "TAS", 0,),
                    (phases['climb3']['ode'], Dynamic.Mission.ALTITUDE, 0,),
                    (phases['cruise']['ode'], Dynamic.Mission.MASS, 0,),
                ],
            )
            traj = self.model.add_subsystem('traj', full_traj)
            return traj

        def add_subsystem_timeseries_outputs(phase, phase_name):
            phase_options = self.phase_info[phase_name]
            all_subsystems = self._get_all_subsystems(
                phase_options['external_subsystems'])
            for subsystem in all_subsystems:
                timeseries_to_add = subsystem.get_outputs()
                for timeseries in timeseries_to_add:
                    phase.add_timeseries_output(timeseries)

        if self.mission_method == "GASP":
            if self.analysis_scheme is AnalysisScheme.COLLOCATION:
                for idx, phase_name in enumerate(phases):
                    phase = traj.add_phase(phase_name, self._get_gasp_phase(phase_name))
                    add_subsystem_timeseries_outputs(phase, phase_name)

                    if phase_name == 'ascent':
                        self._add_groundroll_eq_constraint(phase)

        elif self.mission_method == "FLOPS" or self.mission_method == "simple":
            for phase_idx, phase_name in enumerate(phases):
                phase = traj.add_phase(
                    phase_name, self._get_flops_phase(phase_name, phase_idx))
                add_subsystem_timeseries_outputs(phase, phase_name)

            # loop through phase_info and external subsystems
            external_parameters = {}
            for phase_name in self.phase_info:
                external_parameters[phase_name] = {}
                all_subsystems = self._get_all_subsystems(
                    self.phase_info[phase_name]['external_subsystems'])
                for subsystem in all_subsystems:
                    parameter_dict = subsystem.get_parameters()
                    for parameter in parameter_dict:
                        external_parameters[phase_name][parameter] = parameter_dict[parameter]

            traj = setup_trajectory_params(
                self.model, traj, self.aviary_inputs, phases, meta_data=self.meta_data, external_parameters=external_parameters)

        elif self.mission_method == "solved":
            target_range = self.aviary_inputs.get_val(
                Mission.Design.RANGE, units='nmi')
            takeoff_mass = self.aviary_inputs.get_val(
                Mission.Design.GROSS_MASS, units='lbm')
            climb_mach = 0.8

            for idx, phase_name in enumerate(phases):
                phase = traj.add_phase(phase_name, self._get_solved_phase(phase_name))
                add_subsystem_timeseries_outputs(phase, phase_name)

                if phase_name in phases[1:3]:
                    phase.add_path_constraint(
                        "fuselage_pitch", upper=15., units='deg', ref=15)
                if phase_name == "rotation":
                    phase.add_boundary_constraint(
                        "TAS", loc="final", upper=200., units="kn", ref=200.)
                    phase.add_boundary_constraint(
                        "normal_force", loc="final", equals=0., units="lbf", ref=10000.0)
                elif phase_name == "ascent_to_gear_retract":
                    phase.add_path_constraint("load_factor", lower=0.0, upper=1.10)
                elif phase_name == "ascent_to_flap_retract":
                    phase.add_path_constraint("load_factor", lower=0.0, upper=1.10)
                elif phase_name == "ascent":
                    phase.add_path_constraint(
                        "EAS", upper=250., units="kn", ref=250.)
                elif phase_name == 'climb_at_constant_TAS':
                    phase.add_boundary_constraint(
                        "EAS", loc="final", equals=250., units="kn", ref=250.)
                elif phase_name == "climb_at_constant_EAS":
                    pass
                elif phase_name == "climb_at_constant_EAS_to_mach":
                    phase.add_boundary_constraint(
                        Dynamic.Mission.MACH, loc="final", equals=climb_mach, units="unitless")
                elif phase_name == "climb_at_constant_mach":
                    pass
                elif phase_name == "descent":
                    phase.add_boundary_constraint(
                        Dynamic.Mission.RANGE,
                        loc="final",
                        equals=target_range,
                        units="NM",
                        ref=1.e3)
                    phase.add_boundary_constraint(
                        Dynamic.Mission.ALTITUDE,
                        loc="final",
                        equals=10.e3,
                        units="ft",
                        ref=10e3)
                    phase.add_boundary_constraint(
                        "TAS", loc="final", equals=250., units="kn", ref=250.)

                phase.add_timeseries_output(
                    Dynamic.Mission.THRUST_TOTAL, units="lbf")
                phase.add_timeseries_output("thrust_req", units="lbf")
                phase.add_timeseries_output("normal_force")
                phase.add_timeseries_output(Dynamic.Mission.MACH)
                phase.add_timeseries_output("EAS", units="kn")
                phase.add_timeseries_output("TAS", units="kn")
                phase.add_timeseries_output(Dynamic.Mission.LIFT)
                phase.add_timeseries_output("CL")
                phase.add_timeseries_output("CD")
                phase.add_timeseries_output("time")
                phase.add_timeseries_output("mass")
                phase.add_timeseries_output(Dynamic.Mission.ALTITUDE)
                phase.add_timeseries_output("gear_factor")
                phase.add_timeseries_output("flap_factor")
                phase.add_timeseries_output("alpha")
                phase.add_timeseries_output(
                    "fuselage_pitch", output_name="theta", units="deg")

                if 'rotation' in phase_name:
                    phase.add_parameter("t_init_gear", units="s",
                                        static_target=True, opt=False, val=100)
                    phase.add_parameter("t_init_flaps", units="s",
                                        static_target=True, opt=False, val=100)

        self.traj = traj

        return traj

    def add_post_mission_systems(self, include_landing=True):
        """
        Add post-mission systems to the aircraft model. This is akin to the statics group
        or the "premission_systems", but occurs after the mission in the execution order.

        Depending on the mission model specified (`FLOPS` or `GASP`), this method adds various subsystems
        to the aircraft model. For the `FLOPS` mission model, a landing phase is added using the Landing class
        with the wing area and lift coefficient specified, and a takeoff constraints ExecComp is added to enforce
        mass, range, velocity, and altitude continuity between the takeoff and climb phases. The landing subsystem
        is promoted with aircraft and mission inputs and outputs as appropriate, while the takeoff constraints ExecComp
        is only promoted with mission inputs and outputs.

        For the `GASP` mission model, four subsystems are added: a LandingSegment subsystem, an ExecComp to calculate
        the reserve fuel required, an ExecComp to calculate the overall fuel burn, and three ExecComps to calculate
        various mission objectives and constraints. All subsystems are promoted with aircraft and mission inputs and
        outputs as appropriate.

        A user can override this with their own postmission systems.
        """

        if include_landing and self.post_mission_info['include_landing']:
            if self.mission_method == "FLOPS" or self.mission_method == "simple":
                self._add_flops_landing_systems()
            elif self.mission_method == "GASP":
                self._add_gasp_landing_systems()

        self.model.add_subsystem('post_mission', self.post_mission,
                                 promotes_inputs=['*'],
                                 promotes_outputs=['*'])

        # Loop through all the phases in this subsystem.
        for external_subsystem in self.post_mission_info['external_subsystems']:
            # Get all the subsystem builders for this phase.
            subsystem_postmission = external_subsystem.build_post_mission(
                self.aviary_inputs)

            if subsystem_postmission is not None:
                self.post_mission.add_subsystem(external_subsystem.name,
                                                subsystem_postmission)

        if self.mission_method == "FLOPS" or self.mission_method == "simple":
            phases = list(self.phase_info.keys())
            ecomp = om.ExecComp('fuel_burned = initial_mass - mass_final',
                                initial_mass={'units': 'lbm'},
                                mass_final={'units': 'lbm'},
                                fuel_burned={'units': 'lbm'})

            self.post_mission.add_subsystem('fuel_burn', ecomp,
                                            promotes_outputs=['fuel_burned'])

            self.model.connect(f"traj.{phases[0]}.timeseries.states:mass",
                               "fuel_burn.initial_mass", src_indices=[0])
            self.model.connect(f"traj.{phases[-1]}.states:mass",
                               "fuel_burn.mass_final", src_indices=[-1])

            self._add_fuel_reserve_component()

            # TODO: need to add some sort of check that this value is less than the fuel capacity
            # TODO: the overall_fuel variable is the burned fuel plus the reserve, but should
            # also include the unused fuel, and the hierarchy variable name should be more clear
            ecomp = om.ExecComp('overall_fuel = fuel_burned + fuel_reserve',
                                fuel_burned={'units': 'lbm', 'shape': 1},
                                fuel_reserve={'units': 'lbm', 'shape': 1},
                                overall_fuel={'units': 'lbm'})
            self.post_mission.add_subsystem(
                'fuel_calc', ecomp,
                promotes_inputs=[
                    'fuel_burned',
                    ("fuel_reserve", Mission.Design.RESERVE_FUEL),
                ],
                promotes_outputs=[('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS)])

            if 'constrain_range' in self.post_mission_info:
                if self.post_mission_info['constrain_range']:
                    target_range = self.post_mission_info['target_range']
                    self.post_mission.add_subsystem(
                        "range_constraint",
                        om.ExecComp(
                            "range_resid = target_range - actual_range",
                            range_resid={'units': 'nmi'},
                            actual_range={'units': 'nmi'},
                            target_range={
                                'val': target_range[0], 'units': target_range[1]},
                        ),
                        promotes_outputs=[
                            ("range_resid", Mission.Constraints.RANGE_RESIDUAL)],
                    )

                    self.model.connect(f"traj.{phases[-1]}.timeseries.states:range",
                                       "range_constraint.actual_range", src_indices=[-1])
                    self.model.add_constraint(
                        Mission.Constraints.RANGE_RESIDUAL, equals=0.0, ref=1.e2)

        ecomp = om.ExecComp(
            'mass_resid = operating_empty_mass + overall_fuel + payload_mass -'
            ' initial_mass',
            operating_empty_mass={'units': 'lbm'},
            overall_fuel={'units': 'lbm'},
            payload_mass={'units': 'lbm'},
            initial_mass={'units': 'lbm'},
            mass_resid={'units': 'lbm'})

        if self.mass_method == "GASP":
            payload_mass_src = Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS
        else:
            payload_mass_src = Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS

        self.post_mission.add_subsystem(
            'mass_constraint', ecomp,
            promotes_inputs=[
                ('operating_empty_mass', Aircraft.Design.OPERATING_MASS),
                ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
                ('payload_mass', payload_mass_src),
                ('initial_mass', Mission.Design.GROSS_MASS)],
            promotes_outputs=[("mass_resid", Mission.Constraints.MASS_RESIDUAL)])

        if self.mission_method != "solved":
            self.post_mission.add_constraint(
                Mission.Constraints.MASS_RESIDUAL, equals=0.0, ref=1.e5)

    def _link_phases_helper_with_options(self, phases, option_name, var, **kwargs):
        phases_to_link = []
        for idx, phase_name in enumerate(self.phase_info):
            if self.phase_info[phase_name]['user_options'][option_name][0]:
                # get the name of the previous phase
                if idx > 0:
                    prev_phase_name = phases[idx - 1]
                    phases_to_link.append(prev_phase_name)
                phases_to_link.append(phase_name)
                if idx < len(phases) - 1:
                    next_phase_name = phases[idx + 1]
                    phases_to_link.append(next_phase_name)
        if len(phases_to_link) > 1:
            phases_to_link = list(dict.fromkeys(phases))
            self.traj.link_phases(phases=phases_to_link, vars=[var], **kwargs)

    def link_phases(self):
        """
        Link phases together after they've been added.

        Based on which phases the user has selected, we might need
        special logic to do the Dymos linkages correctly. Some of those
        connections for the simple GASP and FLOPS mission are shown here.
        """
        self._add_bus_variables_and_connect()

        phases = list(self.phase_info.keys())

        if len(phases) <= 1:
            return

        # In summary, the following code loops over all phases in self.phase_info, gets
        # the linked variables from each external subsystem in each phase, and stores
        # the lists of linked variables in lists_to_link. It then gets a list of
        # unique variable names from lists_to_link and loops over them, creating
        # a list of phase names for each variable and linking the phases
        # using self.traj.link_phases().

        lists_to_link = []
        for idx, phase_name in enumerate(self.phase_info):
            lists_to_link.append([])
            for external_subsystem in self.phase_info[phase_name]['external_subsystems']:
                lists_to_link[idx].extend(external_subsystem.get_linked_variables())

        # get unique variable names from lists_to_link
        unique_vars = list(set([var for sublist in lists_to_link for var in sublist]))

        # loop over unique variable names
        for var in unique_vars:
            phases_to_link = []
            for idx, phase_name in enumerate(self.phase_info):
                if var in lists_to_link[idx]:
                    phases_to_link.append(phase_name)

            if len(phases_to_link) > 1:  # TODO: hack
                self.traj.link_phases(phases=phases_to_link, vars=[var], connected=True)

        if self.mission_method == "solved":
            def add_linkage_constraint(self, phase_a, phase_b, var_b):
                self.traj.add_linkage_constraint(phase_a=phase_a,
                                                 phase_b=phase_b,
                                                 var_a='time',
                                                 var_b=var_b,
                                                 loc_a='final',
                                                 loc_b='initial',
                                                 connected=True)

            add_linkage_constraint(self, 'ascent_to_gear_retract',
                                   'ascent_to_flap_retract', 't_init_gear')
            add_linkage_constraint(self, 'ascent_to_gear_retract',
                                   'ascent', 't_init_gear')
            add_linkage_constraint(self, 'ascent_to_gear_retract',
                                   'climb_at_constant_TAS', 't_init_gear')
            add_linkage_constraint(self, 'ascent_to_flap_retract',
                                   'ascent', 't_init_flaps')
            add_linkage_constraint(self, 'ascent_to_flap_retract',
                                   'climb_at_constant_TAS', 't_init_flaps')

            self.traj.link_phases(
                phases[6:], vars=[Dynamic.Mission.ALTITUDE], ref=10.e3)
            self.traj.link_phases(phases, vars=['time'], ref=100.)
            self.traj.link_phases(phases, vars=['mass'], ref=10.e3)
            self.traj.link_phases(
                phases, vars=[Dynamic.Mission.RANGE], units='m', ref=10.e3)
            self.traj.link_phases(phases[:7], vars=['TAS'], units='kn', ref=200.)

        elif self.mission_method == "simple":
            self.traj.link_phases(
                phases, ["time", Dynamic.Mission.MASS, Dynamic.Mission.RANGE], connected=True)

            self._link_phases_helper_with_options(
                phases, 'optimize_altitude', Dynamic.Mission.ALTITUDE, ref=1.e4)
            self._link_phases_helper_with_options(
                phases, 'optimize_mach', Dynamic.Mission.MACH)

        elif self.mission_method == "FLOPS":
            self.traj.link_phases(
                phases, ["time", Dynamic.Mission.ALTITUDE,
                         Dynamic.Mission.MASS, Dynamic.Mission.RANGE], connected=False, ref=1.e4)
            self.traj.link_phases(
                phases, [Dynamic.Mission.VELOCITY], connected=False, ref=250.)

        elif self.mission_method == "GASP":
            if self.analysis_scheme is AnalysisScheme.COLLOCATION:
                self.traj.link_phases(["groundroll", "rotation", "ascent"], [
                    "time", "TAS", "mass", "distance"], connected=True)
                self.traj.link_phases(
                    ["rotation", "ascent"], ["alpha"], connected=False,
                    ref=5e1,
                )
                self.traj.add_linkage_constraint(
                    "ascent",
                    "accel",
                    "distance",
                    "distance",
                    "final",
                    "initial",
                    connected=False,
                    units="NM",
                    ref=5000.,
                )  # we use this because units from the two phases do not match up
                self.traj.link_phases(
                    phases=[
                        "ascent", "accel"], vars=[
                        "time", "mass", "TAS"], connected=True)
                self.traj.link_phases(
                    phases=["accel", "climb1", "climb2"],
                    vars=["time", Dynamic.Mission.ALTITUDE, "mass", "distance"],
                    connected=True,
                )

                self.traj.link_phases(
                    phases=["desc1", "desc2"],
                    vars=["time", "mass", "distance"],
                    connected=True,
                )

                # add all params and promote them to self.model level
                ParamPort.promote_params(
                    self.model,
                    trajs=["traj"],
                    phases=[
                        ["groundroll", "rotation", "ascent",
                            "accel", "climb1", "climb2"],
                        ["desc1", "desc2"],
                    ],
                )

                def add_linkage_constraint(self, phase_a, phase_b, var_a, var_b, connected,
                                           ref=None):
                    self.traj.add_linkage_constraint(
                        phase_a=phase_a,
                        phase_b=phase_b,
                        var_a=var_a,
                        var_b=var_b,
                        loc_a='final',
                        loc_b='initial',
                        connected=connected,
                        ref=ref
                    )

                add_linkage_constraint(self, 'climb2', 'cruise',
                                       'time', 'initial_time', True)
                add_linkage_constraint(self, 'climb2', 'cruise',
                                       'distance', 'initial_distance', True)
                add_linkage_constraint(
                    self, 'climb2', 'cruise', Dynamic.Mission.ALTITUDE, Dynamic.Mission.ALTITUDE, True)
                add_linkage_constraint(self, 'climb2', 'cruise', 'mass',
                                       'mass', False, ref=1.0e5)
                add_linkage_constraint(self, 'cruise', 'desc1', 'time', 'time', True)
                add_linkage_constraint(self, 'cruise', 'desc1',
                                       'distance', 'distance', True)
                add_linkage_constraint(
                    self, 'cruise', 'desc1', Dynamic.Mission.ALTITUDE, Dynamic.Mission.ALTITUDE, True)
                add_linkage_constraint(self, 'cruise', 'desc1', 'mass', 'mass', True)

                self.model.promotes(
                    "traj",
                    inputs=[
                        ("ascent.parameters:t_init_gear", "t_init_gear"),
                        ("ascent.parameters:t_init_flaps", "t_init_flaps"),
                        ("ascent.t_initial", Mission.Takeoff.ASCENT_T_INTIIAL),
                        ("ascent.t_duration", Mission.Takeoff.ASCENT_DURATION),
                    ],
                )

                # imitate input_initial for taxi -> groundroll
                eq = self.model.add_subsystem(
                    "link_taxi_groundroll", om.EQConstraintComp())
                eq.add_eq_output("mass", eq_units="lbm", normalize=False,
                                 ref=10000., add_constraint=True)
                self.model.connect("taxi.mass", "link_taxi_groundroll.rhs:mass")
                self.model.connect(
                    "traj.groundroll.states:mass",
                    "link_taxi_groundroll.lhs:mass",
                    src_indices=[0],
                    flat_src_indices=True,
                )

                self.model.connect("traj.ascent.timeseries.time", "h_fit.time_cp")
                self.model.connect(
                    "traj.ascent.timeseries.states:altitude", "h_fit.h_cp")

                self.model.connect(
                    "traj.desc2.timeseries.states:mass",
                    "landing.mass",
                    src_indices=[-1],
                    flat_src_indices=True,
                )

                connect_map = {
                    "traj.desc2.timeseries.states:distance": Mission.Summary.RANGE,
                    "traj.desc2.states:mass": Mission.Landing.TOUCHDOWN_MASS,
                }
            else:
                connect_map = {
                    "taxi.mass": "traj.mass_initial",
                    Mission.Takeoff.ROTATION_VELOCITY: "traj.SGMGroundroll_TAS_trigger",
                    "traj.distance_final": Mission.Summary.RANGE,
                    "traj.mass_final": Mission.Landing.TOUCHDOWN_MASS,
                    "traj.mass_final": "landing.mass",
                }

            # promote all ParamPort inputs for analytic segments as well
            param_list = list(ParamPort.param_data)
            self.model.promotes("taxi", inputs=param_list)
            self.model.promotes("landing", inputs=param_list)
            if self.analysis_scheme is AnalysisScheme.SHOOTING:
                self.model.promotes("traj", inputs=param_list)
                self.model.list_inputs()
                # self.model.promotes("traj", inputs=['ascent.ODE_group.eoms.'+Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE])

            self.model.connect("taxi.mass", "vrot.mass")

            def connect_with_common_params(self, source, target):
                self.model.connect(
                    source,
                    target,
                    src_indices=[-1],
                    flat_src_indices=True,
                )

            for source, target in connect_map.items():
                connect_with_common_params(self, source, target)

            self.model.set_input_defaults(Mission.Takeoff.ASCENT_DURATION, val=30.0)

    def add_driver(self, optimizer=None, use_coloring=None, max_iter=50, debug_print=False):
        """
        Add an optimization driver to the Aviary problem.

        Depending on the provided optimizer, the method instantiates the relevant driver (ScipyOptimizeDriver or
        pyOptSparseDriver) and sets the optimizer options. Options for 'SNOPT', 'IPOPT', and 'SLSQP' are
        specified. The method also allows for the declaration of coloring and setting debug print options.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer to use. It can be "SLSQP", "SNOPT", "IPOPT" or others supported by OpenMDAO.
            If "SLSQP", it will instantiate a ScipyOptimizeDriver, else it will instantiate a pyOptSparseDriver.

        use_coloring : bool, optional
            If True (default), the driver will declare coloring, which can speed up derivative computations.

        max_iter : int, optional
            The maximum number of iterations allowed for the optimization process. Default is 50. This option is
            applicable to "SNOPT", "IPOPT", and "SLSQP" optimizers.

        debug_print : bool or list, optional
            If True, default debug print options ['desvars','ln_cons','nl_cons','objs'] will be set. If a list is
            provided, it will be used as the debug print options.

        Returns
        -------
        None
        """
        # Set defaults for optimizer and use_coloring based on analysis scheme
        if optimizer is None:
            optimizer = 'IPOPT' if self.analysis_scheme is AnalysisScheme.SHOOTING else 'SNOPT'
        if use_coloring is None:
            use_coloring = False if self.analysis_scheme is AnalysisScheme.SHOOTING else True

        # check if optimizer is SLSQP
        if optimizer == "SLSQP":
            driver = self.driver = om.ScipyOptimizeDriver()
        else:
            driver = self.driver = om.pyOptSparseDriver()

        driver.options["optimizer"] = optimizer
        if use_coloring:
            driver.declare_coloring()

        if driver.options["optimizer"] == "SNOPT":
            driver.opt_settings["Major iterations limit"] = max_iter
            driver.opt_settings["Major optimality tolerance"] = 1e-4
            driver.opt_settings["Major feasibility tolerance"] = 1e-7
            driver.opt_settings["iSumm"] = 6
        elif driver.options["optimizer"] == "IPOPT":
            driver.opt_settings['tol'] = 1.0E-6
            driver.opt_settings['mu_init'] = 1e-5
            driver.opt_settings['max_iter'] = max_iter
            driver.opt_settings['print_level'] = 5
            # for faster convergence
            driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
            driver.opt_settings['mu_strategy'] = 'monotone'
        elif driver.options["optimizer"] == "SLSQP":
            driver.options["tol"] = 1e-9
            driver.options["maxiter"] = max_iter
            driver.options["disp"] = True

        if debug_print:
            if isinstance(debug_print, list):
                driver.options['debug_print'] = debug_print
            else:
                driver.options['debug_print'] = ['desvars', 'ln_cons', 'nl_cons', 'objs']

    def add_design_variables(self):
        """
        Adds design variables to the Aviary problem.

        Depending on the mission model and problem type, different design variables and constraints are added.

        If using the FLOPS model, a design variable is added for the gross mass of the aircraft, with a lower bound of 100,000 lbm and an upper bound of 200,000 lbm.

        If using the GASP model, the following design variables are added depending on the mission type:
            - the initial thrust-to-weight ratio of the aircraft during ascent
            - the duration of the ascent phase
            - the time constant for the landing gear actuation
            - the time constant for the flaps actuation

        In addition, two constraints are added for the GASP model:
            - the initial altitude of the aircraft with gear extended is constrained to be 50 ft
            - the initial altitude of the aircraft with flaps extended is constrained to be 400 ft

        If solving a sizing problem, a design variable is added for the gross mass of the aircraft, and another for the gross mass of the aircraft computed during the mission. A constraint is also added to ensure that the residual range is zero.

        If solving an alternate problem, only a design variable for the gross mass of the aircraft computed during the mission is added. A constraint is also added to ensure that the residual range is zero.

        In all cases, a design variable is added for the final cruise mass of the aircraft, with no upper bound, and a residual mass constraint is added to ensure that the mass balances.

        """
        # add the engine builder `get_design_vars` dict to a collected dict from the external subsystems

        # TODO : maybe in the most general case we need to handle DVs in the mission and post-mission as well.
        # for right now we just handle pre_mission
        all_subsystems = self._get_all_subsystems()

        # loop through all_subsystems and call `get_design_vars` on each subsystem
        for subsystem in all_subsystems:
            dv_dict = subsystem.get_design_vars()
            for dv_name, dv_dict in dv_dict.items():
                self.model.add_design_var(dv_name, **dv_dict)

        if self.mission_method == "FLOPS" or self.mission_method == "simple":
            optimize_mass = self.pre_mission_info.get('optimize_mass')
            if optimize_mass:
                self.model.add_design_var(Mission.Design.GROSS_MASS, units='lbm',
                                          lower=100.e3, upper=200.e3, ref=135.e3)

        elif self.mission_method == "GASP":
            if self.analysis_scheme is AnalysisScheme.COLLOCATION:
                # problem formulation to make the trajectory work
                self.model.add_design_var(Mission.Takeoff.ASCENT_T_INTIIAL,
                                          lower=0, upper=100, ref=30.0)
                self.model.add_design_var(Mission.Takeoff.ASCENT_DURATION,
                                          lower=1, upper=1000, ref=10.)
                self.model.add_design_var("tau_gear", lower=0.01,
                                          upper=1.0, units="s", ref=1)
                self.model.add_design_var("tau_flaps", lower=0.01,
                                          upper=1.0, units="s", ref=1)
                self.model.add_constraint(
                    "h_fit.h_init_gear", equals=50.0, units="ft", ref=50.0)
                self.model.add_constraint("h_fit.h_init_flaps",
                                          equals=400.0, units="ft", ref=400.0)

            self.problem_type = self.aviary_inputs.get_val('problem_type')

            # vehicle sizing problem
            # size the vehicle (via design GTOW) to meet a target range using all fuel capacity
            if self.problem_type is ProblemType.SIZING:
                self.model.add_design_var(
                    Mission.Design.GROSS_MASS,
                    lower=10.,
                    upper=400.e3,
                    units="lbm",
                    ref=175_000,
                )
                self.model.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=10.,
                    upper=400.e3,
                    units="lbm",
                    ref=175_000,
                )

                self.model.add_constraint(
                    Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10
                )
                self.model.add_subsystem(
                    "gtow_constraint",
                    om.EQConstraintComp(
                        "GTOW",
                        eq_units="lbm",
                        normalize=True,
                        add_constraint=True,
                    ),
                    promotes_inputs=[
                        ("lhs:GTOW", Mission.Design.GROSS_MASS),
                        ("rhs:GTOW", Mission.Summary.GROSS_MASS),
                    ],
                )

            # target range problem
            # fixed vehicle (design GTOW) but variable actual GTOW for off-design mission range
            elif self.problem_type is ProblemType.ALTERNATE:
                self.model.add_design_var(
                    Mission.Summary.GROSS_MASS,
                    lower=0,
                    upper=None,
                    units="lbm",
                    ref=175_000,
                )

                self.model.add_constraint(
                    Mission.Constraints.RANGE_RESIDUAL, equals=0, ref=10,
                )
            elif self.problem_type is ProblemType.FALLOUT:
                print('No design variables for Fallout missions')

    def add_objective(self, objective_type=None, ref=None):
        """
        Add the objective function based on the given objective_type and ref.

        NOTE: the ref value should be positive for values you're trying
        to minimize and negative for values you're trying to maximize.
        Please check and double-check that your ref value makes sense
        for the objective you're using.

        Parameters
        ----------
        objective_type : str
            The type of objective to add. Options are 'mass', 'hybrid_objective', 'fuel_burned', and 'fuel'.
        ref : float
            The reference value for the objective. If None, a default value will be used based on the objective type. Please see the
            `default_ref_values` dict for these default values.

        Raises
        ------
            ValueError: If an invalid problem type is provided.

        """
        # Dictionary for default reference values
        default_ref_values = {
            'mass': -5e4,
            'hybrid_objective': -5e4,
            'fuel_burned': 1e4,
            'fuel': 1e4
        }

        # Check if an objective type is specified
        if objective_type is not None:
            ref = ref if ref is not None else default_ref_values.get(objective_type, 1)

            if objective_type == 'mass':
                last_phase = list(self.traj._phases.items())[-1][1]
                last_phase.add_objective(
                    Dynamic.Mission.MASS, loc='final', ref=ref)
            elif objective_type == "hybrid_objective":
                self._add_hybrid_objective(self.phase_info)
                self.model.add_objective("obj_comp.obj")
            elif objective_type == "fuel_burned":
                self.model.add_objective("fuel_burned", ref=ref)
            elif objective_type == "fuel":
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)

        # If 'mission_method' is 'FLOPS', add a 'fuel_burned' objective
        elif self.mission_method == "FLOPS" or self.mission_method == "simple":
            ref = ref if ref is not None else default_ref_values.get("fuel_burned", 1)
            self.model.add_objective("fuel_burned", ref=ref)

        else:  # If no 'objective_type' is specified, we handle based on 'problem_type'
            # If 'ref' is not specified, assign a default value
            ref = ref if ref is not None else 1

            if self.problem_type is ProblemType.SIZING:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)
            elif self.problem_type is ProblemType.ALTERNATE:
                self.model.add_objective(Mission.Objectives.FUEL, ref=ref)
            elif self.problem_type is ProblemType.FALLOUT:
                self.model.add_objective(Mission.Objectives.RANGE, ref=ref)
            else:
                raise ValueError(f'{self.problem_type} is not a valid problem type.')

    def _add_bus_variables_and_connect(self):
        all_subsystems = self._get_all_subsystems()

        base_phases = list(self.phase_info.keys())

        for external_subsystem in all_subsystems:
            bus_variables = external_subsystem.get_bus_variables()
            if bus_variables is not None:
                for bus_variable in bus_variables:
                    mission_variable_name = bus_variables[bus_variable]['mission_name']

                    # check if mission_variable_name is a list
                    if not isinstance(mission_variable_name, list):
                        mission_variable_name = [mission_variable_name]

                    # loop over the mission_variable_name list and add each variable to the trajectory
                    for mission_var_name in mission_variable_name:
                        if 'mission_name' in bus_variables[bus_variable]:
                            if mission_var_name not in self.meta_data:
                                # base_units = self.model.get_io_metadata(includes=f'pre_mission.{external_subsystem.name}.{bus_variable}')[f'pre_mission.{external_subsystem.name}.{bus_variable}']['units']
                                base_units = bus_variables[bus_variable]['units']

                                shape = bus_variables[bus_variable].get(
                                    'shape', _unspecified)

                                targets = mission_var_name
                                if '.' in mission_var_name:
                                    # Support for non-hiearchy variables as parameters.
                                    mission_var_name = mission_var_name.split('.')[-1]

                                if 'phases' in bus_variables[bus_variable]:
                                    # Support for connecting bus variables into a subset of
                                    # phases.
                                    phases = bus_variables[bus_variable]['phases']

                                    for phase_name in phases:
                                        phase = getattr(self.traj.phases, phase_name)

                                        phase.add_parameter(mission_var_name, opt=False, static_target=True,
                                                            units=base_units, shape=shape, targets=targets)

                                        self.model.connect(f'pre_mission.{bus_variable}',
                                                           f'traj.{phase_name}.parameters:{mission_var_name}')

                                else:
                                    phases = base_phases

                                    self.traj.add_parameter(mission_var_name, opt=False, static_target=True,
                                                            units=base_units, shape=shape, targets={
                                                                phase_name: [mission_var_name] for phase_name in phases})

                                    self.model.connect(
                                        f'pre_mission.{bus_variable}', f'traj.parameters:'+mission_var_name)

                        if 'post_mission_name' in bus_variables[bus_variable]:
                            self.model.connect(f'pre_mission.{external_subsystem.name}.{bus_variable}',
                                               f'post_mission.{external_subsystem.name}.{bus_variables[bus_variable]["post_mission_name"]}')

    def setup(self, **kwargs):
        """
        Lightly wrappd setup() method for the problem.

        Allows us to do pre- and post-setup changes, like adding
        calls to `set_input_defaults` and do some simple `set_vals`
        if needed.
        """
        # Adding a trailing component that contains all inputs so that set_input_defaults
        # doesn't raise any errors.
        self.model.add_subsystem(
            'input_sink',
            VariablesIn(aviary_options=self.aviary_inputs,
                        meta_data=self.meta_data),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        # suppress warnings:
        # "input variable '...' promoted using '*' was already promoted using 'aircraft:*'
        with warnings.catch_warnings():

            # Set initial default values for all LEAPS aircraft variables.
            set_aviary_initial_values(
                self.model, self.aviary_inputs, meta_data=self.meta_data)

            warnings.simplefilter("ignore", om.PromotionWarning)
            super().setup(**kwargs)

    def set_initial_guesses(self):
        """
        Call `set_val` on the trajectory for states and controls to seed
        the problem with reasonable initial guesses. This is especially
        important for collocation methods.

        This method first identifies all phases in the trajectory then
        loops over each phase. If the mission method is "solved", solved guesses
        are added to the problem for each phase as those are handled differently
        than other mission types. If not solved, specific initial guesses
        are added depending on the phase and mission method. Cruise is treated
        as a special phase for GASP-based missions because it is an AnalyticPhase
        in Dymos. For this phase, we handle the initial guesses first separately
        and continue to the next phase after that. For other phases, we set the initial
        guesses for states and controls according to the information available
        in the 'initial_guesses' attribute of the phase.
        """
        # Grab the trajectory object from the model
        if self.analysis_scheme is AnalysisScheme.SHOOTING:
            if self.problem_type is ProblemType.SIZING:
                self.set_val(Mission.Summary.GROSS_MASS,
                             self.get_val(Mission.Design.GROSS_MASS))

            self.set_val("traj.SGMClimb_"+Dynamic.Mission.ALTITUDE +
                         "_trigger", val=self.cruise_alt, units="ft")

            return

        traj = self.model.traj

        # Determine which phases to loop over, fetching them from the trajectory
        phase_items = traj._phases.items()

        # Loop over each phase and set initial guesses for the state and control variables
        for idx, (phase_name, phase) in enumerate(phase_items):
            if self.mission_method == "solved":
                # If so, add solved guesses to the problem
                self._add_solved_guesses(idx, phase_name, phase)
            else:
                # If not, fetch the initial guesses specific to the phase
                guesses = self.phase_info[phase_name]['initial_guesses']

                if 'cruise' in phase_name and self.mission_method == "GASP":
                    for guess_key, guess_data in guesses.items():
                        val, units = guess_data

                        if 'mass' == guess_key:
                            # Set initial and duration mass for the analytic cruise phase.
                            # Note we are integrating over mass, not time for this phase.
                            self.set_val(f'traj.{phase_name}.t_initial',
                                         val[0], units=units)
                            self.set_val(f'traj.{phase_name}.t_duration',
                                         val[1], units=units)
                        else:
                            # Otherwise, set the value of the parameter in the trajectory phase
                            self.set_val(f'traj.{phase_name}.parameters:{guess_key}',
                                         val, units=units)

                    continue

                # If not cruise and GASP, add subsystem guesses
                self._add_subsystem_guesses(phase_name, phase)

                # Set initial guesses for states and controls for each phase
                self._add_guesses(phase_name, phase, guesses)

    def _process_guess_var(self, val, key, phase):
        """
        Process the guess variable, which can either be a float or an array of floats.

        This method is responsible for interpolating initial guesses when the user
        provides a list or array of values rather than a single float. It interpolates
        the guess values across the phase's domain for a given variable, be it a control
        or a state variable. The interpolation is performed between -1 and 1 (representing
        the normalized phase time domain), using the numpy linspace function.

        The result of this method is a single value or an array of interpolated values
        that can be used to seed the optimization problem with initial guesses.

        Parameters
        ----------
        val : float or list/array of floats
            The initial guess value(s) for a particular variable.
        key : str
            The key identifying the variable for which the initial guess is provided.
        phase : Phase
            The phase for which the variable is being set.

        Returns
        -------
        val : float or array of floats
            The processed guess value(s) to be used in the optimization problem.
        """

        # Check if val is not a single float
        if not isinstance(val, float):
            # If val is an array of values
            if len(val) > 1:
                # Get the shape of the val array
                shape = np.shape(val)

                # Generate an array of evenly spaced values between -1 and 1,
                # reshaping to match the shape of the val array
                xs = np.linspace(-1, 1, num=np.prod(shape)).reshape(shape)

                # Check if the key indicates a control or state variable
                if "controls:" in key or "states:" in key:
                    # If so, strip the first part of the key to match the variable name in phase
                    stripped_key = ":".join(key.split(":")[1:])

                    # Interpolate the initial guess values across the phase's domain
                    val = phase.interp(stripped_key, xs=xs, ys=val)
                else:
                    # If not a control or state variable, interpolate the initial guess values directly
                    val = phase.interp(key, xs=xs, ys=val)

        # Return the processed guess value(s)
        return val

    def _add_subsystem_guesses(self, phase_name, phase):
        """
        Adds the initial guesses for each subsystem of a given phase to the problem.

        This method first fetches all subsystems associated with the given phase.
        It then loops over each subsystem and fetches its initial guesses. For each
        guess, it identifies whether the guess corresponds to a state or a control
        variable and then processes the guess variable. After this, the initial
        guess is set in the problem using the `set_val` method.

        Parameters
        ----------
        phase_name : str
            The name of the phase for which the subsystem guesses are being added.
        phase : Phase
            The phase object for which the subsystem guesses are being added.
        """

        # Get all subsystems associated with the phase
        all_subsystems = self._get_all_subsystems(
            self.phase_info[phase_name]['external_subsystems'])

        # Loop over each subsystem
        for subsystem in all_subsystems:
            # Fetch the initial guesses for the subsystem
            initial_guesses = subsystem.get_initial_guesses()

            # Loop over each guess
            for key, val in initial_guesses.items():
                # Identify the type of the guess (state or control)
                type = val.pop('type')
                if 'state' in type:
                    path_string = 'states'
                elif 'control' in type:
                    path_string = 'controls'

                # Process the guess variable (handles array interpolation)
                val['val'] = self._process_guess_var(val['val'], key, phase)

                # Set the initial guess in the problem
                self.set_val(f'traj.{phase_name}.{path_string}:{key}', **val)

    def _add_guesses(self, phase_name, phase, guesses):
        """
        Adds the initial guesses for each variable of a given phase to the problem.

        This method sets the initial guesses for time, control, state, and problem-specific
        variables for a given phase. If using the GASP model, it also handles some special
        cases that are not covered in the `phase_info` object. These include initial guesses
        for mass, time, and distance, which are determined based on the phase name and other
        mission-related variables.

        Parameters
        ----------
        phase_name : str
            The name of the phase for which the guesses are being added.
        phase : Phase
            The phase object for which the guesses are being added.
        guesses : dict
            A dictionary containing the initial guesses for the phase.
        """

        # If using the GASP model, set initial guesses for the rotation mass and flight duration
        if self.mission_method == "GASP":
            rotation_mass = self.initial_guesses['rotation_mass']
            flight_duration = self.initial_guesses['flight_duration']

        if self.mission_method == "simple":
            control_keys = ["mach", "altitude"]
            state_keys = ["mass", "range"]
        else:
            control_keys = ["velocity_rate", "throttle"]
            state_keys = ["altitude", "velocity", "mass",
                          "range", "TAS", "distance", "flight_path_angle", "alpha"]
            if self.mission_method == "GASP" and phase_name == 'ascent':
                # Alpha is a control for ascent.
                control_keys.append('alpha')

        prob_keys = ["tau_gear", "tau_flaps"]

        # for the simple mission method, use the provided initial and final mach and altitude values from phase_info
        if self.mission_method == "simple":
            initial_altitude = self.phase_info[phase_name]['user_options']['initial_altitude']
            final_altitude = self.phase_info[phase_name]['user_options']['final_altitude']
            initial_mach = self.phase_info[phase_name]['user_options']['initial_mach']
            final_mach = self.phase_info[phase_name]['user_options']['final_mach']

            # add a check for the altitude units, raise an error if they are not consistent
            if initial_altitude[1] != final_altitude[1]:
                raise ValueError(
                    f"Initial and final altitude units for {phase_name} are not consistent.")
            guesses["mach"] = ([initial_mach[0], final_mach[0]], "unitless")
            guesses["altitude"] = (
                [initial_altitude[0], final_altitude[0]], initial_altitude[1])

        for guess_key, guess_data in guesses.items():
            val, units = guess_data

            # Set initial guess for time variables
            if 'times' == guess_key:
                self.set_val(f'traj.{phase_name}.t_initial',
                             val[0], units=units)
                self.set_val(f'traj.{phase_name}.t_duration',
                             val[1], units=units)

            else:
                # Set initial guess for control variables
                if guess_key in control_keys:
                    try:
                        self.set_val(f'traj.{phase_name}.controls:{guess_key}', self._process_guess_var(
                            val, guess_key, phase), units=units)
                    except KeyError:
                        try:
                            self.set_val(f'traj.{phase_name}.polynomial_controls:{guess_key}', self._process_guess_var(
                                val, guess_key, phase), units=units)
                        except KeyError:
                            self.set_val(f'traj.{phase_name}.bspline_controls:{guess_key}', self._process_guess_var(
                                val, guess_key, phase), units=units)

                # Set initial guess for state variables
                elif guess_key in state_keys:
                    self.set_val(f'traj.{phase_name}.states:{guess_key}', self._process_guess_var(
                        val, guess_key, phase), units=units)
                elif guess_key in prob_keys:
                    self.set_val(guess_key, val, units=units)
                elif ":" in guess_key:
                    self.set_val(f'traj.{phase_name}.{guess_key}', self._process_guess_var(
                        val, guess_key, phase), units=units)
                else:
                    # raise error if the guess key is not recognized
                    raise ValueError(
                        f"Initial guess key {guess_key} in {phase_name} is not recognized.")

        # We need some special logic for these following variables because GASP computes
        # initial guesses using some knowledge of the mission duration and other variables
        # that are only available after calling `create_vehicle`. Thus these initial guess
        # values are not included in the `phase_info` object.
        if 'mass' not in guesses:
            if self.mission_method == "GASP":
                # Determine a mass guess depending on the phase name
                if phase_name in ["groundroll", "rotation", "ascent", "accel", "climb1"]:
                    mass_guess = rotation_mass
                elif phase_name == "climb2":
                    mass_guess = 0.99 * rotation_mass
                elif "desc" in phase_name:
                    mass_guess = 0.9 * self.cruise_mass_final
            else:
                mass_guess = self.aviary_inputs.get_val(
                    Mission.Design.GROSS_MASS, units='lbm')
            # Set the mass guess as the initial value for the mass state variable
            self.set_val(f'traj.{phase_name}.states:mass',
                         mass_guess, units='lbm')

        if 'times' not in guesses:
            # Determine initial time and duration guesses depending on the phase name
            if 'desc1' == phase_name:
                t_initial = flight_duration*.9
                t_duration = flight_duration*.04
            elif 'desc2' in phase_name:
                t_initial = flight_duration*.94
                t_duration = 5000
            # Set the time guesses as the initial values for the time-related trajectory variables
            self.set_val(f"traj.{phase_name}.t_initial",
                         t_initial, units='s')
            self.set_val(f"traj.{phase_name}.t_duration",
                         t_duration, units='s')

        if self.mission_method == "GASP":
            if 'distance' not in guesses:
                # Determine initial distance guesses depending on the phase name
                if 'desc1' == phase_name:
                    ys = [self.target_range*.97, self.target_range*.99]
                elif 'desc2' in phase_name:
                    ys = [self.target_range*.99, self.target_range]
                # Set the distance guesses as the initial values for the distance state variable
                self.set_val(
                    f"traj.{phase_name}.states:distance", phase.interp(
                        "distance", ys=ys)
                )

    def _add_solved_guesses(self, idx, phase_name, phase):
        target_range = self.aviary_inputs.get_val(
            Mission.Design.RANGE, units='nmi')
        takeoff_mass = self.aviary_inputs.get_val(
            Mission.Design.GROSS_MASS, units='lbm')
        cruise_alt = self.aviary_inputs.get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')

        range_guesses = np.array([  # in meters
            0.,  # 'groundroll',
            1000.,  # 'rotation',
            1500.,  # 'ascent_to_gear_retract',
            2000.,  # 'ascent_to_flap_retract',
            2500.,  # 'ascent',
            3500.,  # 'climb_at_constant_TAS',
            4500.,  # 'climb_at_constant_EAS',
            32000.,  # 'climb_at_constant_EAS_to_mach',
            300.e3,  # 'climb_at_constant_mach',
            600.e3,  # 'cruise',
            1700. * target_range,  # 'descent'
            1700. * target_range + 200000.,
        ])
        mass_guesses = np.array([
            1.0,
            0.999,
            0.999,
            0.999,
            0.998,
            0.998,
            0.998,
            0.990,
            0.969,
            0.951,
            0.873,
            0.866,
        ]) * takeoff_mass
        alt_guesses = [
            0.0,
            0.0,
            0.0,
            50.0,
            400.0,
            500.0,
            500.0,
            10000.,
            32857.,
            cruise_alt,
            cruise_alt,
            10000.,
        ]
        TAS_guesses = np.array([
            0.0,
            150.,
            200.,
            200.,
            200.,
            225.,
            251.,
            290.,
            465. * self.cruise_mach / 0.8,
            458. * self.cruise_mach / 0.8,
            483. * self.cruise_mach / 0.8,
            250.,
        ])

        mean_TAS = (TAS_guesses[1:] + TAS_guesses[:-1]) / 2. / 1.94
        range_durations = range_guesses[1:] - range_guesses[:-1]
        time_guesses = np.hstack((0., np.cumsum(range_durations / mean_TAS)))

        if phase_name != "groundroll":
            range_initial = range_guesses[idx]
            self.set_val(f"traj.{phase_name}.t_initial",
                         range_initial, units='range_units')
            self.set_val(f"traj.{phase_name}.t_duration",
                         range_guesses[idx+1] - range_initial, units='range_units')

            self.set_val(
                f"traj.{phase_name}.polynomial_controls:altitude",
                phase.interp(Dynamic.Mission.ALTITUDE, [
                    alt_guesses[idx], alt_guesses[idx + 1]]),
                units="ft",
            )

            if "constant_EAS" in phase_name or "constant_mach" in phase_name or "cruise" in phase_name:
                pass
            else:
                self.set_val(
                    f"traj.{phase_name}.polynomial_controls:TAS",
                    phase.interp(
                        "TAS", [TAS_guesses[idx], TAS_guesses[idx+1]]),
                    units="kn",
                )
        else:
            self.set_val(f"traj.{phase_name}.t_initial", 0., units='kn')
            self.set_val(f"traj.{phase_name}.t_duration", 100., units='kn')

        self.set_val(
            f"traj.{phase_name}.states:mass",
            phase.interp("mass", [mass_guesses[idx], mass_guesses[idx+1]]),
            units="lbm",
        )

        self.set_val(
            f"traj.{phase_name}.states:time",
            phase.interp("time", [time_guesses[idx], time_guesses[idx+1]]),
            units="s",
        )

        self.final_setup()

    def run_aviary_problem(self,
                           record_filename="aviary_history.db",
                           optimization_history_filename=None,
                           restart_filename=None, suppress_solver_print=True, run_driver=True, simulate=False, make_plots=True):
        """
        This function actually runs the Aviary problem, which could be a simulation, optimization, or a driver execution, depending on the arguments provided.

        Parameters
        ----------
        record_filename : str, optional
            The name of the database file where the solutions are to be recorded. The default is "aviary_history.db".
        optimization_history_filename : str, None
            The name of the database file where the driver iterations are to be recorded. The default is None.
        restart_filename : str, optional
            The name of the file that contains previously computed solutions which are to be used as starting points for this run. If it is None (default), no restart file will be used.
        suppress_solver_print : bool, optional
            If True (default), all solvers' print statements will be suppressed. Useful for deeply nested models with multiple solvers so the print statements don't overwhelm the output.
        run_driver : bool, optional
            If True (default), the driver (aka optimizer) will be executed. If False, the problem will be run through one pass -- equivalent to OpenMDAO's `run_model` behavior.
        simulate : bool, optional
            If True, an explicit Dymos simulation will be performed. The default is False.
        make_plots : bool, optional
            If True (default), Dymos html plots will be generated as part of the output.
        """

        if self.aviary_inputs.get_val('debug_mode'):
            self.final_setup()
            with open('input_list.txt', 'w') as outfile:
                self.model.list_inputs(out_stream=outfile)

        if suppress_solver_print:
            self.set_solver_print(level=0)

        if optimization_history_filename:
            recorder = om.SqliteRecorder(optimization_history_filename)
            self.driver.add_recorder(recorder)

        # and run mission, and dynamics
        if run_driver:
            failed = dm.run_problem(self, run_driver=run_driver, simulate=simulate, make_plots=make_plots,
                                    solution_record_file=record_filename, restart=restart_filename)
        else:
            # prevent UserWarning that is displayed when an event is triggered
            warnings.filterwarnings('ignore', category=UserWarning)
            failed = self.run_model()
            warnings.filterwarnings('default', category=UserWarning)

        if self.aviary_inputs.get_val('debug_mode'):
            with open('output_list.txt', 'w') as outfile:
                self.model.list_outputs(out_stream=outfile)

        return failed

    def _add_hybrid_objective(self, phase_info):
        phases = list(phase_info.keys())
        takeoff_mass = self.aviary_inputs.get_val(
            Mission.Design.GROSS_MASS, units='lbm')

        obj_comp = om.ExecComp(f"obj = -final_mass / {takeoff_mass} + final_time / 5.",
                               final_mass={"units": "lbm"},
                               final_time={"units": "h"})
        self.model.add_subsystem("obj_comp", obj_comp)

        final_phase_name = phases[-1]
        self.model.connect(f"traj.{final_phase_name}.timeseries.mass",
                           "obj_comp.final_mass", src_indices=[-1])
        self.model.connect(f"traj.{final_phase_name}.timeseries.time",
                           "obj_comp.final_time", src_indices=[-1])

    def _add_vrotate_comp(self):
        self.model.add_subsystem("vrot_comp", VRotateComp())
        self.model.connect('traj.groundroll.states:mass',
                           'vrot_comp.mass', src_indices=om.slicer[0, ...])

        vrot_eq_comp = self.model.add_subsystem("vrot_eq_comp", om.EQConstraintComp())
        vrot_eq_comp.add_eq_output("v_rotate_error", eq_units="kn",
                                   lhs_name="v_rot_computed", rhs_name="groundroll_v_final", add_constraint=True)

        self.model.connect('vrot_comp.Vrot', 'vrot_eq_comp.v_rot_computed')
        self.model.connect('traj.groundroll.timeseries.TAS',
                           'vrot_eq_comp.groundroll_v_final', src_indices=om.slicer[-1, ...])

    def _save_to_csv_file(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['name', 'value', 'units']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for name, value_units in sorted(self.aviary_inputs):
                if 'engine_models' in name:
                    continue
                value, units = value_units
                writer.writerow({'name': name, 'value': value, 'units': units})

    def _get_all_subsystems(self, external_subsystems=None):
        all_subsystems = []
        if external_subsystems is None:
            all_subsystems.extend(self.pre_mission_info['external_subsystems'])
        else:
            all_subsystems.extend(external_subsystems)

        if self.engine_builder is not None:
            all_subsystems.append(self.engine_builder)

        return all_subsystems

    def _add_flops_landing_systems(self):
        landing_options = Landing(
            ref_wing_area=self.aviary_inputs.get_val(
                Aircraft.Wing.AREA, units='ft**2'),
            Cl_max_ldg=self.aviary_inputs.get_val(
                Mission.Landing.LIFT_COEFFICIENT_MAX)  # no units
        )

        landing = landing_options.build_phase(
            False,
        )
        self.model.add_subsystem(
            'landing', landing, promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'])

        connect_takeoff_to_climb = not self.phase_info['climb']['user_options'].get(
            'add_initial_mass_constraint', True)[0]

        if connect_takeoff_to_climb:
            self.model.connect(Mission.Takeoff.FINAL_MASS,
                               'traj.climb.initial_states:mass')
            self.model.connect(Mission.Takeoff.GROUND_DISTANCE,
                               'traj.climb.initial_states:range')
            if self.mission_method == "FLOPS":
                self.model.connect(Mission.Takeoff.FINAL_VELOCITY,
                                   'traj.climb.initial_states:velocity')
                self.model.connect(Mission.Takeoff.FINAL_ALTITUDE,
                                   'traj.climb.initial_states:altitude')
            else:
                pass
                # TODO: connect this correctly
                # mass is the most important to connect but these others should
                # be connected as well
                # self.model.connect(Mission.Takeoff.FINAL_VELOCITY,
                #                 'traj.climb.initial_states:mach')
                # self.model.connect(Mission.Takeoff.FINAL_ALTITUDE,
                #                 'traj.climb.controls:altitude')

        self.model.connect('traj.descent.states:mass',
                           Mission.Landing.TOUCHDOWN_MASS, src_indices=[-1])
        # TODO: approach velocity should likely be connected
        if self.mission_method == "FLOPS":
            self.model.connect('traj.descent.states:altitude', Mission.Landing.INITIAL_ALTITUDE,
                               src_indices=[-1])
        else:
            self.model.connect('traj.descent.control_values:altitude', Mission.Landing.INITIAL_ALTITUDE,
                               src_indices=[0])

    def _add_gasp_landing_systems(self):
        self.model.add_subsystem(
            "landing",
            LandingSegment(
                **(self.ode_args)),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['mission:*'],
        )

        self._add_fuel_reserve_component()

        self.model.add_subsystem(
            "fuel_burn",
            om.ExecComp(
                "overall_fuel = (1 + fuel_margin/100)*(takeoff_mass - final_mass) + reserve_fuel",
                takeoff_mass={"units": "lbm"},
                final_mass={"units": "lbm"},
                fuel_margin={"units": "unitless"},
                reserve_fuel={"units": "lbm"},
                overall_fuel={"units": "lbm"},
            ),
            promotes_inputs=[
                ("takeoff_mass", Mission.Summary.GROSS_MASS),
                ("fuel_margin", Aircraft.Fuel.FUEL_MARGIN),
                ("final_mass", Mission.Landing.TOUCHDOWN_MASS),
                ("reserve_fuel", Mission.Design.RESERVE_FUEL),
            ],
            promotes_outputs=[("overall_fuel", Mission.Summary.TOTAL_FUEL_MASS)],
        )

        self.model.add_subsystem(
            "fuel_obj",
            om.ExecComp(
                "reg_objective = overall_fuel/10000 + ascent_duration/30.",
                reg_objective={"val": 0.0, "units": "unitless"},
                ascent_duration={"units": "s", "shape": 1},
                overall_fuel={"units": "lbm"},
            ),
            promotes_inputs=[
                ("ascent_duration", Mission.Takeoff.ASCENT_DURATION),
                ("overall_fuel", Mission.Summary.TOTAL_FUEL_MASS),
            ],
            promotes_outputs=[("reg_objective", Mission.Objectives.FUEL)],
        )

        self.model.add_subsystem(
            "range_obj",
            om.ExecComp(
                "reg_objective = -actual_range/1000 + ascent_duration/30.",
                reg_objective={"val": 0.0, "units": "unitless"},
                ascent_duration={"units": "s", "shape": 1},
                actual_range={
                    "val": self.target_range, "units": "NM"},
            ),
            promotes_inputs=[
                ("actual_range", Mission.Summary.RANGE),
                ("ascent_duration", Mission.Takeoff.ASCENT_DURATION),
            ],
            promotes_outputs=[("reg_objective", Mission.Objectives.RANGE)],
        )

        self.model.add_subsystem(
            "range_constraint",
            om.ExecComp(
                "range_resid = target_range - actual_range",
                target_range={"val": self.target_range, "units": "NM"},
                actual_range={"val": self.target_range - 25, "units": "NM"},
                range_resid={"val": 30, "units": "NM"},
            ),
            promotes_inputs=[
                ("actual_range", Mission.Summary.RANGE),
                ("target_range", Mission.Design.RANGE),
            ],
            promotes_outputs=[
                ("range_resid", Mission.Constraints.RANGE_RESIDUAL)],
        )

    def _add_fuel_reserve_component(self, reserves_name=Mission.Design.RESERVE_FUEL):
        reserves_val = self.aviary_inputs.get_val(Aircraft.Design.RESERVES)
        if reserves_val <= 0:
            reserves_val = -reserves_val
            self.model.add_subsystem(
                "reserves_calc",
                om.ExecComp(
                    f"reserve_fuel = {reserves_val}*(takeoff_mass - final_mass)",
                    takeoff_mass={"units": "lbm"},
                    final_mass={"units": "lbm"},
                    reserve_fuel={"units": "lbm"}
                ),
                promotes_inputs=[
                    ("takeoff_mass", Mission.Summary.GROSS_MASS),
                    ("final_mass", Mission.Landing.TOUCHDOWN_MASS),
                ],
                promotes_outputs=[("reserve_fuel", reserves_name)],
            )
        elif reserves_val > 10:
            self.model.add_subsystem(
                "reserves_calc",
                om.ExecComp(
                    f"reserve_fuel = {reserves_val}",
                    reserve_fuel={"val": reserves_val, "units": "lbm"}
                ),
                promotes_outputs=[("reserve_fuel", reserves_name)],
            )
        else:
            raise ValueError('"aircraft:design:reserves" is not valid between 0 and 10.')
