"""
Define utilities for building phases.

Classes
-------
PhaseBuilderBase : the interface for a phase builder
"""

from abc import ABC
from collections import namedtuple

import dymos as dm
import openmdao.api as om
import numpy as np

from aviary.mission.flops_based.ode.takeoff_ode import TakeoffODE
from aviary.mission.flops_based.phases.balanced_field_trajectory import BalancedFieldPhaseBuilder
from aviary.mission.initial_guess_builders import InitialGuess
from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic
from aviary.variable_info.functions import setup_trajectory_params
from aviary.api import Aircraft, Dynamic, Mission

from aviary.mission.phase_builder_base import PhaseBuilderBase


_require_new_initial_guesses_meta_data_class_attr_ = namedtuple(
    '_require_new_initial_guesses_meta_data_class_attr_', ()
)


class BalancedFieldTrajectoryBuilder(ABC):
    """
    Define the interface for a balanced field trajectory builder.

    Attributes
    ----------
    name : str ('_unknown phase_')
        object label

    core_subsystems : (None)
        list of SubsystemBuilderBase objects that will be added to the phase ODE

    user_options : OptionsDictionary (<empty>)
        state/path constraint values and flags

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

    ode_class : type (None)
        advanced: the type of system defining the ODE

    transcription : "Dymos transcription object" (None)
        advanced: an object providing the transcription technique of the
        optimal control problem

    subsystem_options : dict (None)
        dictionary of parameters to be passed to the subsystem builders

    default_name : str
        class attribute: derived type customization point; the default value
        for name

    default_ode_class : type
        class attribute: derived type customization point; the default value
        for ode_class used by build_phase

    default_options_class : type
        class attribute: derived type customization point; the default class
        containing the phase options options_dictionary

    is_analytic_phase : bool (False)
        class attribute: derived type customization point; if True, build_phase
        will return an AnalyticPhase instead of a Phase

    num_nodes : int (5)
        class attribute: derived type customization point; the default value
        for num_nodes used by build_phase, only for AnalyticPhases

    Methods
    -------
    build_phase
    make_default_transcription
    """

    __slots__ = (
        'name',
        'core_subsystems',
        'external_subsystems',
        'subsystem_options',
        'user_options',
        'initial_guesses',
        'ode_class',
        'transcription',
        'is_analytic_phase',
        'num_nodes',
        'meta_data',
        '_traj',
        '_phases',
    )

    MappedPhase = namedtuple('MappedPhase', ('phase', 'phase_builder'))

    _initial_guesses_meta_data_ = _require_new_initial_guesses_meta_data_class_attr_()

    default_name = '_unknown phase_'

    default_ode_class = TakeoffODE
    default_options_class = om.OptionsDictionary

    default_meta_data = _MetaData
    # endregion : derived type customization points

    def __init__(
        self,
        name=None,
        core_subsystems=None,
        external_subsystems=None,
        user_options=None,
        initial_guesses=None,
        subsystem_options=None,
        num_nodes=10,
        meta_data=None,
        traj=None,
    ):
        if name is None:
            name = self.default_name

        self.name = name

        if core_subsystems is None:
            core_subsystems = []
        if external_subsystems is None:
            external_subsystems = []

        self.core_subsystems = core_subsystems
        self.external_subsystems = external_subsystems

        if subsystem_options is None:
            subsystem_options = {}

        self.subsystem_options = subsystem_options

        self.user_options = self.default_options_class(user_options)

        if initial_guesses is None:
            initial_guesses = AviaryValues()

        self.initial_guesses = initial_guesses
        self.validate_initial_guesses()

        self.ode_class = self.default_ode_class
        self.num_nodes = num_nodes

        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

        self._traj = traj
        self._phases = {}

    def build_trajectory(
        self, *, aviary_options: AviaryValues, model: om.Group = None, traj: dm.Trajectory = None
    ) -> dm.Trajectory:
        """
        Return a new trajectory for detailed takeoff analysis.

        Call only after assigning phase builders for required phases.

        Parameters
        ----------
        aviary_options : AviaryValues
            collection of Aircraft/Mission specific options

        model : openmdao.api.Group (None)
            the model handling trajectory parameter setup; if `None`, trajectory
            parameter setup will not be handled

        traj : dymos.Trajectory (None)
            the trajectory to update; if `None`, a new trajetory will be updated and
            returned

        Returns
        -------
        the updated trajectory; if the specified trajectory is `None`, a new trajectory
        will be updated and returned

        Notes
        -----
        Do not modify this object or any of its referenced data between the call to
        `build_trajectory()` and the call to `apply_initial_guesses()`, or the behavior
        is undefined, no diagnostic required.
        """
        # ode_class = self.ode_class

        # transcription = dm.PicardShooting(nodes_per_seg=10)

        if traj is None:
            self._traj = traj = dm.Trajectory(parallel_phases=False)
            model.add_subsystem('traj', self._traj)

        if aviary_options is None:
            aviary_options = AviaryValues()

        # kwargs = self._extra_ode_init_kwargs()

        # kwargs = {'aviary_options': aviary_options, **kwargs}

        # # if subsystem_options is not None:
        # #     kwargs['subsystem_options'] = subsystem_options

        # kwargs['core_subsystems'] = self.core_subsystems
        # kwargs['external_subsystems'] = self.external_subsystems

        # Add all phase builders here.

        common_user_options = AviaryValues()
        common_user_options.set_val('max_duration', val=100.0, units='s')
        common_user_options.set_val('time_duration_ref', val=60.0, units='s')
        common_user_options.set_val('distance_max', val=10000.0, units='ft')
        common_user_options.set_val('max_velocity', val=200.0, units='kn')

        tobl_nl_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6
        )
        tobl_nl_solver.linesearch = om.BoundsEnforceLS()

        common_user_options.set_val('nonlinear_solver', val=tobl_nl_solver)
        common_user_options.set_val('linear_solver', val=om.DirectSolver())

        common_initial_guesses = AviaryValues()

        common_initial_guesses.set_val('time', [0.0, 30.0], 's')
        common_initial_guesses.set_val('distance', [0.0, 4100.0], 'ft')
        common_initial_guesses.set_val('velocity', [0.01, 150.0], 'kn')
        common_initial_guesses.set_val('throttle', 1.0)
        common_initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        gross_mass_units = 'lbm'
        gross_mass = aviary_options.get_val(Mission.Design.GROSS_MASS, gross_mass_units)
        common_initial_guesses.set_val('mass', gross_mass, gross_mass_units)

        #
        # First phase: Brake release to engine failure
        #
        # TODO: would like to be able to do this -  user_options=balanced_field_user_options | {'terminal_condition': 'VEF'}
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'VEF')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('dV1', 3.0, 'kn')
        initial_guesses.set_val('dVEF', 1.0, 'kn')
        takeoff_brake_release_to_engine_failure_builder = BalancedFieldPhaseBuilder(
            'takeoff_brake_release_to_engine_failure',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )
        self._add_phase(takeoff_brake_release_to_engine_failure_builder, aviary_options)

        #
        # Second Phase: Engine Failure to Decision Speed
        #
        # TODO: dymos PicardShooting shouldn't require initial guesses if connecting an initial state value
        #       In that case just assume the state value holds through the phase as an initial guess.
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'V1')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [30.0, 31.0], 's')
        initial_guesses.set_val('distance', [4100.0, 4500.0], 'ft')
        initial_guesses.set_val('velocity', [150.0, 151], 'kn')
        initial_guesses.set_val('throttle', 0.5)  # TODO: Don't do this hack, decrement num engines
        initial_guesses.set_val('angle_of_attack', 0.0, 'deg')
        initial_guesses.set_val('dV1', 3.0, 'kn')

        takeoff_engine_failure_to_v1_builder = BalancedFieldPhaseBuilder(
            'takeoff_engine_failure_to_v1',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_engine_failure_to_v1_builder, aviary_options)

        #
        # Third Phase: Decision Speed to Rejected Takeoff Stop
        #
        # TODO: dymos PicardShooting shouldn't require initial guesses if connecting an initial state value
        #       In that case just assume the state value holds through the phase as an initial guess.
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'STOP')
        user_options.set_val('friction_key', Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT)
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [31.0, 20.0], 's')
        initial_guesses.set_val('distance', [4500.0, 6500.0], 'ft')
        initial_guesses.set_val('velocity', [151.0, 5.0], 'kn')
        initial_guesses.set_val('throttle', 0.0)
        initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        takeoff_v1_to_roll_stop = BalancedFieldPhaseBuilder(
            'takeoff_v1_to_roll_stop',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_v1_to_roll_stop, aviary_options)

        #
        # Fourth Phase: Decision Speed to Rotation Speed
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'VR')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [36.0, 1.0], 's')
        initial_guesses.set_val('distance', [4500.0, 5000.0], 'ft')
        initial_guesses.set_val('velocity', [151.0, 155.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val('angle_of_attack', 0.0, 'deg')

        takeoff_v1_to_vr = BalancedFieldPhaseBuilder(
            'takeoff_v1_to_vr',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_v1_to_vr, aviary_options)

        #
        # Fifth Phase: Rotate to liftoff
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'LIFTOFF')
        user_options.set_val('pitch_control', 'alpha_rate_fixed')
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [37.0, 7.0], 's')
        initial_guesses.set_val('distance', [5000.0, 5500.0], 'ft')
        initial_guesses.set_val('velocity', [155.0, 160.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [0.0, 14.0], 'deg')
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE, 3.0, 'deg/s')

        takeoff_vr_to_liftoff = BalancedFieldPhaseBuilder(
            'takeoff_vr_to_liftoff',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_vr_to_liftoff, aviary_options)

        #
        # Sixth Phase: Liftoff to V2
        #
        user_options = common_user_options.deepcopy()
        user_options.set_val('terminal_condition', 'OBSTACLE')
        user_options.set_val('pitch_control', 'alpha_rate_fixed')
        user_options.set_val('climbing', True)
        initial_guesses = common_initial_guesses.deepcopy()
        initial_guesses.set_val('time', [37.0, 15.0], 's')
        initial_guesses.set_val('distance', [5000.0, 5500.0], 'ft')
        initial_guesses.set_val('velocity', [155.0, 160.0], 'kn')
        initial_guesses.set_val('throttle', 0.5)
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, [10.0, 10.0], 'deg')
        initial_guesses.set_val(Dynamic.Mission.FLIGHT_PATH_ANGLE, [0.0, 0.5], 'deg')
        initial_guesses.set_val(Dynamic.Mission.ALTITUDE, [0.0, 0.5], 'ft')
        initial_guesses.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK_RATE, 1.0, 'deg/s')

        takeoff_liftoff_to_v2 = BalancedFieldPhaseBuilder(
            'takeoff_liftoff_to_v2',
            core_subsystems=self.core_subsystems,
            subsystem_options=self.subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        self._add_phase(takeoff_liftoff_to_v2, aviary_options)

        bal_comp = self._traj.add_subsystem('balance_comp', om.BalanceComp())

        # The first balance sets dVEF such that the elapsed time from engine failure to V1 is one second.
        bal_comp.add_balance('dVEF', lhs_name='t_react', rhs_val=1.0, eq_units='s', units='kn')
        self._traj.connect(
            'balance_comp.dVEF', 'takeoff_brake_release_to_engine_failure.parameters:dVEF'
        )
        self._traj.connect('takeoff_engine_failure_to_v1.t_duration', 'balance_comp.t_react')

        # The second balance sets dV1 such that the distance at the end of the abort roll and the distance at obstacle clearance are the same
        bal_comp.add_balance(
            'dV1',
            lhs_name='distance_abort',
            rhs_name='distance_obstacle',
            eq_units='ft',
            units='kn',
        )
        self._traj.connect(
            'balance_comp.dV1',
            [
                'takeoff_brake_release_to_engine_failure.parameters:dV1',
                'takeoff_engine_failure_to_v1.parameters:dV1',
            ],
        )
        self._traj.connect(
            'takeoff_liftoff_to_v2.final_states:distance', 'balance_comp.distance_obstacle'
        )
        self._traj.connect(
            'takeoff_v1_to_roll_stop.final_states:distance', 'balance_comp.distance_abort'
        )

        self._traj.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=100, iprint=2, atol=1.0e-6, rtol=1.0e-6
        )
        self._traj.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        self._traj.linear_solver = om.DirectSolver()

        # Press to Takeoff Sequence
        self._traj.link_phases(
            [
                'takeoff_brake_release_to_engine_failure',
                'takeoff_engine_failure_to_v1',
                'takeoff_v1_to_vr',
                'takeoff_vr_to_liftoff',
                'takeoff_liftoff_to_v2',
            ],
            vars=['time', 'distance', 'velocity', 'mass'],
            connected=True,
        )

        self._traj.link_phases(
            ['takeoff_v1_to_vr', 'takeoff_vr_to_liftoff', 'takeoff_liftoff_to_v2'],
            vars=[Dynamic.Vehicle.ANGLE_OF_ATTACK],
            connected=True,
        )

        # Abort Sequence
        self._traj.link_phases(
            ['takeoff_engine_failure_to_v1', 'takeoff_v1_to_roll_stop'],
            vars=['time', 'distance', 'velocity', 'mass'],
            connected=True,
        )

        if model is not None:
            phase_names = list(self._phases.keys())

            # This is a level 3 method that uses the default subsystems.
            # We need to create parameters for just the inputs we have.
            # They mostly come from the low-speed aero subsystem.

            # aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, LegacyCode('FLOPS'))

            # for cs in self.core_subsystems:
            #     print(cs.name)
            #     print(cs.get_parameters(**kwargs))
            # exit(0)

            kwargs = {'method': 'low_speed'}

            # TODO: Why is get_parameters different for different subsystems?
            # Do with without blindly indexing.
            aero_builder = self.core_subsystems[0]

            params = aero_builder.get_parameters(aviary_options, **kwargs)

            # takeoff introduces this one.
            params[Mission.Takeoff.LIFT_COEFFICIENT_MAX] = {
                'shape': (1,),
                'static_target': True,
            }

            ext_params = {}
            for phase in self._phases.keys():
                ext_params[phase] = params

            setup_trajectory_params(
                model, self._traj, aviary_options, phase_names, external_parameters=ext_params
            )

        return self._traj

    def validate_initial_guesses(self):
        """
        Raise TypeError if an unsupported initial guess is found.

        Users can call this method when updating initial guesses after initialization.
        """
        initial_guesses = self.initial_guesses

        # if not initial_guesses:
        #     return  # acceptable

        # meta_data = self._initial_guesses_meta_data_

        # for key in get_keys(initial_guesses):
        #     if key not in meta_data:
        #         raise TypeError(
        #             f'{self.__class__.__name__}: {self.name}: unsupported initial guess: {key}'
        #         )

    def apply_initial_guesses(self, prob: om.Problem, traj_name):  # , phase: dm.Phase):
        """Apply any stored initial guesses; return a list of guesses not applied."""
        not_applied = {}
        phase_builder: PhaseBuilderBase = None

        for phase, phase_builder in self._phases.values():
            tmp = phase_builder.apply_initial_guesses(prob, traj_name, phase)

            if tmp:
                not_applied[phase_builder.name] = tmp

        return not_applied

    def _extra_ode_init_kwargs(self):
        """Return extra kwargs required for initializing the ODE."""
        return {}

    def to_phase_info(self):
        """
        Return the stored settings as phase info.

        Returns
        -------
        tuple
            name : str
                object label
            phase_info : dict
                stored settings
        """
        subsystem_options = self.subsystem_options  # TODO: aero info?
        user_options = self.user_options.to_phase_info()
        initial_guesses = dict(self.initial_guesses)

        # TODO some of these may be purely programming API hooks, rather than for use
        # with phase info
        # - ode_class
        # - transcription
        # - external_subsystems
        # - meta_data

        phase_info = dict(
            subsystem_options=subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
        )

        return (self.name, phase_info)

    @classmethod
    def from_phase_info(
        cls, name, phase_info: dict, core_subsystems=None, meta_data=None, transcription=None
    ):
        """
        Return a new phase builder based on the specified phase info.

        Note, calling code is responsible for matching phase info to the correct phase
        builder type, or the behavior is undefined.

        Parameters
        ----------
        name : str
            object label
        phase_info : dict
            stored settings
        """
        # loop over user_options dict entries
        # if the value is not a tuple, wrap it in a tuple with the second entry of 'unitless'
        for key, value in phase_info['user_options'].items():
            if not isinstance(value, tuple):
                phase_info['user_options'][key] = (value, 'unitless')

        subsystem_options = phase_info.get('subsystem_options', {})  # TODO: aero info?
        user_options = phase_info.get('user_options', ())
        initial_guesses = AviaryValues(phase_info.get('initial_guesses', ()))
        external_subsystems = phase_info.get('external_subsystems', [])
        # TODO core subsystems in phase info?

        # TODO some of these may be purely programming API hooks, rather than for use
        # with phase info
        # - ode_class
        # - transcription
        # - external_subsystems
        # - meta_data

        phase_builder = cls(
            name,
            subsystem_options=subsystem_options,
            user_options=user_options,
            initial_guesses=initial_guesses,
            meta_data=meta_data,
            core_subsystems=core_subsystems,
            external_subsystems=external_subsystems,
            transcription=transcription,
        )

        return phase_builder

    @classmethod
    def _add_initial_guess_meta_data(cls, initial_guess: InitialGuess, desc=None):
        """
        Update supported initial guesses with a new item.

        Raises
        ------
        ValueError
            if a repeat initial guess is found
        """
        meta_data = cls._initial_guesses_meta_data_
        name = initial_guess.key

        meta_data[name] = dict(apply_initial_guess=initial_guess.apply_initial_guess, desc=desc)

    def _add_user_defined_constraints(self, phase, constraints):
        """Add each constraint and its corresponding arguments to the phase."""
        for constraint_name, kwargs in constraints.items():
            if kwargs['type'] == 'boundary':
                kwargs.pop('type')

                if 'target' in kwargs:
                    # Support for constraint aliases.
                    target = kwargs.pop('target')
                    kwargs['constraint_name'] = constraint_name
                    phase.add_boundary_constraint(target, **kwargs)
                else:
                    phase.add_boundary_constraint(constraint_name, **kwargs)

            elif kwargs['type'] == 'path':
                kwargs.pop('type')
                phase.add_path_constraint(constraint_name, **kwargs)

    def add_state(self, name, target, rate_source):
        """
        Add a state to this phase using the options in the phase_info.

        Parameters
        ----------
        name : str
            The name of this state in the phase_info options.
        target : str
            State promoted variable path to the ODE.
        rate_source : str
            Source of the state rate in the ODE.
        """
        options = self.user_options

        initial, _ = options[f'{name}_initial']
        final, _ = options[f'{name}_final']
        bounds, units = options[f'{name}_bounds']
        ref, _ = options[f'{name}_ref']
        ref0, _ = options[f'{name}_ref0']
        defect_ref, _ = options[f'{name}_defect_ref']
        solve_segments = options[f'{name}_solve_segments']

        # If a value is specified for the starting node, then fix_initial is True.
        # Otherwise, input_initial is True.
        # The problem configurator may change input_initial to False requested or necessary, (e.g.,
        # for parallel phases in MPI.)

        self.phase.add_state(
            target,
            fix_initial=initial is not None,
            input_initial=False,
            lower=bounds[0],
            upper=bounds[1],
            units=units,
            rate_source=rate_source,
            ref=ref,
            ref0=ref0,
            defect_ref=defect_ref,
            solve_segments='forward' if solve_segments else None,
        )

        if final is not None:
            constraint_ref, _ = options[f'{name}_constraint_ref']
            if constraint_ref is None:
                # If unspecified, final is a good value for it.
                constraint_ref = final
            self.phase.add_boundary_constraint(
                target,
                loc='final',
                equals=final,
                units=units,
                ref=final,
            )

    def add_control(
        self, name, target, rate_targets=None, rate2_targets=None, add_constraints=True
    ):
        """
        Add a control to this phase using the options in the phase-info.

        Parameters
        ----------
        name : str
            The name of this control in the phase_info options.
        target : str
            Control promoted variable path to the ODE.
        rate_source : list of str
            List of rate targets for this control.
        rate2_targets : Sequence of str or None
            (Optional) The parameter in the ODE to which the control 2nd derivative is connected.
        add_constraints : bool
            When True, add constraints on any declared initial and final values if this control is
            being optimized. Default is True.
        """
        options = self.user_options
        phase = self.phase

        initial, _ = options[f'{name}_initial']
        final, _ = options[f'{name}_final']
        bounds, units = options[f'{name}_bounds']
        ref, _ = options[f'{name}_ref']
        ref0, _ = options[f'{name}_ref0']
        polynomial_order = options[f'{name}_polynomial_order']
        opt = options[f'{name}_optimize']

        if ref == 1.0:
            # This has not been moved from default, so find a good value.
            candidates = [x for x in (bounds[0], bounds[1], initial, final) if x is not None]
            if len(candidates) > 0:
                ref = np.max(np.abs(np.array(candidates)))

        extra_options = {}
        if polynomial_order is not None:
            extra_options['control_type'] = 'polynomial'
            extra_options['order'] = polynomial_order

        if opt is True:
            extra_options['lower'] = bounds[0]
            extra_options['upper'] = bounds[1]
            extra_options['ref'] = ref
            extra_options['ref0'] = ref0

        if units not in ['unitless', None]:
            extra_options['units'] = units

        if rate_targets is not None:
            extra_options['rate_targets'] = rate_targets

        if rate2_targets is not None:
            extra_options['rate2_targets'] = rate2_targets

        phase.add_control(target, targets=target, opt=opt, **extra_options)

        # Add timeseries for any control.
        phase.add_timeseries_output(target)

        if not add_constraints:
            return

        # Add an initial constraint.
        if opt and initial is not None:
            phase.add_boundary_constraint(
                target, loc='initial', equals=initial, units=units, ref=ref
            )

        # Add a final constraint.
        if opt and final is not None:
            phase.add_boundary_constraint(target, loc='final', equals=final, units=units, ref=ref)

    def _add_phase(self, phase_builder: PhaseBuilderBase, aviary_options: AviaryValues):
        name = phase_builder.name
        phase = phase_builder.build_phase(aviary_options)

        self._traj.add_phase(name, phase)

        self._phases[name] = self.MappedPhase(phase, phase_builder)
