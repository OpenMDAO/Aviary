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

from aviary.mission.flops_based.ode.energy_ODE import EnergyODE
from aviary.mission.initial_guess_builders import InitialGuess
from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.variable_info.variable_meta_data import _MetaData
from aviary.variable_info.variables import Dynamic

_require_new_initial_guesses_meta_data_class_attr_ = namedtuple(
    '_require_new_initial_guesses_meta_data_class_attr_', ()
)


class PhaseBuilderBase(ABC):
    """
    Define the interface for a phase builder.

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
    )

    _initial_guesses_meta_data_ = _require_new_initial_guesses_meta_data_class_attr_()

    default_name = '_unknown phase_'

    default_ode_class = EnergyODE
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
        ode_class=None,
        transcription=None,
        subsystem_options=None,
        is_analytic_phase=False,
        num_nodes=5,
        meta_data=None,
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

        self.ode_class = ode_class
        self.transcription = transcription
        self.is_analytic_phase = is_analytic_phase
        self.num_nodes = num_nodes

        if external_subsystems is None:
            external_subsystems = []

        self.external_subsystems = external_subsystems

        if meta_data is None:
            meta_data = self.default_meta_data

        self.meta_data = meta_data

    def build_phase(self, aviary_options=None):
        """
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (empty)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        """
        ode_class = self.ode_class

        if ode_class is None:
            ode_class = self.default_ode_class

        transcription = self.transcription

        if transcription is None and not self.is_analytic_phase:
            transcription = self.make_default_transcription()

        if aviary_options is None:
            aviary_options = AviaryValues()

        kwargs = self._extra_ode_init_kwargs()

        kwargs = {'aviary_options': aviary_options, **kwargs}

        subsystem_options = self.subsystem_options

        if subsystem_options is not None:
            kwargs['subsystem_options'] = subsystem_options

        kwargs['core_subsystems'] = self.core_subsystems
        kwargs['external_subsystems'] = self.external_subsystems

        if self.is_analytic_phase:
            phase = dm.AnalyticPhase(
                ode_class=ode_class,
                ode_init_kwargs=kwargs,
                num_nodes=self.num_nodes,
            )
        else:
            phase = dm.Phase(
                ode_class=ode_class, transcription=transcription, ode_init_kwargs=kwargs
            )

        # Add a timeseries for the "mission bus variables" that will be a uniform grid, using Falck Magik™.
        # https://stackoverflow.com/questions/67771242/openmdao-dymos-interpolate-the-results-of-a-phase-onto-an-equispaced-grid
        if self.is_analytic_phase:
            tx_mission_bus = dm.GaussLobatto(num_segments=self.num_nodes, order=3, compressed=True)
        else:
            tx_mission_bus = dm.GaussLobatto(
                num_segments=transcription.options['num_segments'], order=3, compressed=True
            )
        phase.add_timeseries(
            name='mission_bus_variables', transcription=tx_mission_bus, subset='all'
        )

        # overrides should add state, controls, etc.
        return phase

    def make_default_transcription(self):
        """Return a transcription object to be used by default in build_phase."""
        user_options = self.user_options

        num_segments = user_options['num_segments']
        order = user_options['order']

        transcription = dm.Radau(num_segments=num_segments, order=order, compressed=True)

        return transcription

    def validate_initial_guesses(self):
        """
        Raise TypeError if an unsupported initial guess is found.

        Users can call this method when updating initial guesses after initialization.
        """
        initial_guesses = self.initial_guesses

        if not initial_guesses:
            return  # acceptable

        meta_data = self._initial_guesses_meta_data_

        for key in get_keys(initial_guesses):
            if key not in meta_data:
                raise TypeError(
                    f'{self.__class__.__name__}: {self.name}: unsupported initial guess: {key}'
                )

    def apply_initial_guesses(self, prob: om.Problem, traj_name, phase: dm.Phase):
        """Apply any stored initial guesses; return a list of guesses not applied."""
        not_applied = []

        phase_name = self.name
        meta_data = self._initial_guesses_meta_data_
        initial_guesses: AviaryValues = self.initial_guesses

        for key in meta_data:
            if key in initial_guesses:
                apply_initial_guess = meta_data[key]['apply_initial_guess']
                val, units = initial_guesses.get_item(key)

                apply_initial_guess(prob, traj_name, phase, phase_name, val, units)

            else:
                not_applied.append(key)

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
        """Add each constraint to this phase using the arguments given in the phase_info."""
        for constraint_name, constraint_dict in constraints.items():
            con_args = constraint_dict.copy()
            con_type = con_args.pop('type')
            if con_type == 'boundary':
                if 'target' in con_args:
                    # Support for constraint aliases.
                    target = con_args.pop('target')
                    con_args['constraint_name'] = constraint_name
                    phase.add_boundary_constraint(target, **con_args)
                else:
                    phase.add_boundary_constraint(constraint_name, **con_args)

            elif con_type == 'path':
                phase.add_path_constraint(constraint_name, **con_args)

            else:
                raise ValueError(
                    f'Invalid type "{con_type}" for constraint {constraint_name} in {phase.name}.'
                )

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

        # Try to only use arguments that aren't None because dymos' default is _undefined.
        extra_options = {}
        if ref0 is not None:
            extra_options['ref0'] = ref0
        if defect_ref is not None:
            extra_options['defect_ref'] = defect_ref

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
            solve_segments='forward' if solve_segments else None,
            **extra_options,
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

        # Try to only use arguments that aren't None because dymos' default is _undefined.
        extra_options = {}
        if polynomial_order is not None:
            extra_options['control_type'] = 'polynomial'
            extra_options['order'] = polynomial_order

        if opt is True:
            extra_options['lower'] = bounds[0]
            extra_options['upper'] = bounds[1]
            extra_options['ref'] = ref
            extra_options['ref0'] = ref0
            extra_options['continuity_ref'] = ref

            # TODO: We may want to consider letting the user setting this.
            # extra_options['rate_continuity_ref'] = ref

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


_registered_phase_builder_types = []


def register(phase_builder_t=None, *, check_repeats=True):
    """
    Register a new phase builder type.

    Note, this function qualifies as a class decorator for ease of use.

    Returns
    -------
    phase builder type
    """
    if phase_builder_t is None:

        def decorator(phase_builder_t):
            return register(phase_builder_t, check_repeats=check_repeats)

        return decorator

    if check_repeats and (phase_builder_t in _registered_phase_builder_types):
        raise ValueError('repeated phase builder type')

    _registered_phase_builder_types.append(phase_builder_t)

    return phase_builder_t


def phase_info_to_builder(name: str, phase_info: dict) -> PhaseBuilderBase:
    """
    Return a new phase builder based on the specified phase info.

    Note, the type of phase builder will be determined by calling
    phase_builder_t.from_phase_info() for each registered type in order of registration;
    the first result that is not None will be returned. If a supported phase builder type
    cannot be determined, raise ValueError.

    Raises
    ------
    ValueError
        if a supported phase builder type cannot be determined
    """
    phase_builder_t: PhaseBuilderBase = None

    for phase_builder_t in _registered_phase_builder_types:
        builder = phase_builder_t.from_phase_info(name, phase_info)

        if builder is not None:
            return builder

    raise ValueError(f'unsupported phase info: {name}')


if __name__ == '__main__':
    help(PhaseBuilderBase)
