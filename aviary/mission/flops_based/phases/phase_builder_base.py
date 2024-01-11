'''
Define utilities for building phases.

Classes
-------
InitialGuess : a utility for setting an initial guess on a problem

InitialGuessControl : a utility for setting an initial guess for a control on a problem

InitialGuessParameter : a utility for setting an initial guess for a parameter on a
problem

InitialGuessPolynomialControl : a utility for setting an initial guess for a polynomial
control on a problem

InitialGuessState : a utility for setting an initial guess for a state on a problem

InitialGuessTime : a utility for setting guesses for initial time and duration on a
problem

PhaseBuilderBase : the interface for a phase builder
'''
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Sequence

import dymos as dm
import numpy as np
import openmdao.api as om

from aviary.mission.flops_based.ode.mission_ODE import MissionODE
from aviary.utils.aviary_values import AviaryValues, get_keys

_require_new_meta_data_class_attr_ = \
    namedtuple('_require_new_meta_data_class_attr_', ())


_require_new_initial_guesses_meta_data_class_attr_ = \
    namedtuple('_require_new_initial_guesses_meta_data_class_attr_', ())


class InitialGuess:
    '''
    Define a utility for setting an initial guess on a problem.

    Attributes
    ----------
    key : str
        base name for the initial guess
    '''
    __slots__ = ('key',)

    def __init__(self, key):
        self.key = key

    def apply_initial_guess(
        self, prob: om.Problem, traj_name, phase: dm.Phase, phase_name, val, units
    ):
        '''
        Set the initial guess on the problem.
        '''
        complete_key = self._get_complete_key(traj_name, phase_name)

        # TODO: this is a short term hack in need of an appropriate long term solution
        #    - to interpolate, or not to interpolate: that is the question
        #    - the solution should probably be a value decoration (a wrapper) that is
        #      both lightweight and easy to check and unpack
        if (
            isinstance(val, np.ndarray)
            or (isinstance(val, Sequence) and not isinstance(val, str))
        ):
            val = phase.interp(self.key, val)

        prob.set_val(complete_key, val, units)

    def _get_complete_key(self, traj_name, phase_name):
        '''
        Compose the complete key for setting the initial guess.
        '''
        _ = traj_name
        _ = phase_name

        return self.key


class InitialGuessControl(InitialGuess):
    '''
    Define a utility for setting an initial guess for a control on a problem.

    Attributes
    ----------
    key : str
        base name for the control
    '''
    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        '''
        Compose the complete key for setting the control initial guess.
        '''
        key = f'{traj_name}.{phase_name}.controls:{self.key}'

        return key


class InitialGuessParameter(InitialGuess):
    '''
    Define a utility for setting an initial guess for a parameter on a problem.

    Attributes
    ----------
    key : str
        base name for the parameter
    '''
    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        '''
        Compose the complete key for setting the parameter initial guess.
        '''
        key = f'{traj_name}.{phase_name}.parameters:{self.key}'

        return key


class InitialGuessPolynomialControl(InitialGuess):
    '''
    Define a utility for setting an initial guess for a polynomial control on a problem.

    Attributes
    ----------
    key : str
        base name for the polynomial control
    '''
    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        '''
        Compose the complete key for setting the polynomial control initial guess.
        '''
        key = f'{traj_name}.{phase_name}.polynomial_controls:{self.key}'

        return key


class InitialGuessState(InitialGuess):
    '''
    Define a utility for setting an initial guess for a state on a problem.

    Attributes
    ----------
    key : str
        base name for the state
    '''
    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        '''
        Compose the complete key for setting the state initial guess.
        '''
        key = f'{traj_name}.{phase_name}.states:{self.key}'

        return key


class InitialGuessTime(InitialGuess):
    '''
    Define a utility for setting guesses for initial time and duration on a problem.

    Attributes
    ----------
    key : str ('times')
        the group identifier for guesses for initial time and duration
    '''
    __slots__ = ()

    def __init__(self, key='times'):
        super().__init__(key)

    def apply_initial_guess(
        self, prob: om.Problem, traj_name, phase: dm.Phase, phase_name, val, units
    ):
        '''
        Set the guesses for initial time and duration on the problem.
        '''
        _ = phase

        name = f'{traj_name}.{phase_name}.t_initial'
        t_initial, t_duration = val
        prob.set_val(name, t_initial, units)

        name = f'{traj_name}.{phase_name}.t_duration'
        prob.set_val(name, t_duration, units)


class PhaseBuilderBase(ABC):
    '''
    Define the interface for a phase builder.

    Attributes
    ----------
    name : str ('<unknown phase>')
        object label

    core_subsystems : (None)
        list of SubsystemBuilderBase objects that will be added to the phase ODE

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
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

    Methods
    -------
    build_phase
    make_default_transcription
    validate_options
    assign_default_options
    '''
    __slots__ = (
        'name',  'core_subsystems', 'subsystem_options', 'user_options',
        'initial_guesses', 'ode_class', 'transcription', 'aero_builder'
    )

    # region : derived type customization points
    _meta_data_ = _require_new_meta_data_class_attr_()

    _initial_guesses_meta_data_ = _require_new_initial_guesses_meta_data_class_attr_()

    default_name = '<unknown phase>'

    default_ode_class = MissionODE
    # endregion : derived type customization points

    def __init__(
        self, name=None, core_subsystems=None, aero_builder=None, user_options=None, initial_guesses=None,
        ode_class=None, transcription=None, subsystem_options=None,
    ):
        if name is None:
            name = self.default_name

        self.name = name

        if core_subsystems is None:
            core_subsystems = []

        self.core_subsystems = core_subsystems

        if subsystem_options is None:
            subsystem_options = {}

        if aero_builder is not None:
            self.aero_builder = aero_builder

        self.subsystem_options = subsystem_options

        self.user_options = user_options
        self.validate_options()
        self.assign_default_options()

        if initial_guesses is None:
            initial_guesses = AviaryValues()

        self.initial_guesses = initial_guesses
        self.validate_initial_guesses()

        self.ode_class = ode_class
        self.transcription = transcription

    def build_phase(self, aviary_options=None):
        '''
        Return a new phase object for analysis using these constraints.

        If ode_class is None, default_ode_class is used.

        If transcription is None, the return value from calling
        make_default_transcription is used.

        Parameters
        ----------
        aviary_options : AviaryValues (emtpy)
            collection of Aircraft/Mission specific options

        Returns
        -------
        dymos.Phase
        '''
        ode_class = self.ode_class

        if ode_class is None:
            ode_class = self.default_ode_class

        transcription = self.transcription

        if transcription is None:
            transcription = self.make_default_transcription()

        if aviary_options is None:
            aviary_options = AviaryValues()

        kwargs = self._extra_ode_init_kwargs()

        kwargs = {
            'aviary_options': aviary_options,
            **kwargs
        }

        subsystem_options = self.subsystem_options

        if subsystem_options is not None:
            kwargs['subsystem_options'] = subsystem_options

        kwargs['core_subsystems'] = self.core_subsystems

        phase = dm.Phase(
            ode_class=ode_class, transcription=transcription,
            ode_init_kwargs=kwargs
        )

        # overrides should add state, controls, etc.
        return phase

    def make_default_transcription(self):
        '''
        Return a transcription object to be used by default in build_phase.
        '''
        user_options = self.user_options

        num_segments, _ = user_options.get_item('num_segments')
        order, _ = user_options.get_item('order')

        transcription = dm.Radau(
            num_segments=num_segments, order=order, compressed=True)

        return transcription

    def validate_options(self):
        '''
        Raise TypeError if an unsupported option is found.

        Users can call this method when updating options after initialization.
        '''
        user_options = self.user_options

        if not user_options:
            return  # acceptable

        meta_data = self._meta_data_

        for key in get_keys(user_options):
            if key not in meta_data:
                raise TypeError(
                    f'{self.__class__.__name__}: {self.name}:'
                    f' unsupported option: {key}'
                )

    def assign_default_options(self):
        '''
        Update missing options with default values.

        If user_options is None, start with an empty AviaryValues.

        Users can call this method when replacing the user_options member after
        initialization.
        '''
        user_options = self.user_options

        if user_options is None:
            user_options = self.user_options = AviaryValues()

        meta_data = self._meta_data_

        for key in meta_data:
            if key not in user_options:
                item = meta_data[key]

                val = item['val']
                units = item['units']

                user_options.set_val(key, val, units)

    def validate_initial_guesses(self):
        '''
        Raise TypeError if an unsupported initial guess is found.

        Users can call this method when updating initial guesses after initialization.
        '''
        initial_guesses = self.initial_guesses

        if not initial_guesses:
            return  # acceptable

        meta_data = self._initial_guesses_meta_data_

        for key in get_keys(initial_guesses):
            if key not in meta_data:
                raise TypeError(
                    f'{self.__class__.__name__}: {self.name}:'
                    f' unsupported initial guess: {key}'
                )

    def apply_initial_guesses(
        self, prob: om.Problem, traj_name, phase: dm.Phase
    ):
        '''
        Apply any stored initial guesses; return a list of guesses not applied.
        '''
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
        """
        Return extra kwargs required for initializing the ODE.
        """
        return {}

    def to_phase_info(self):
        '''
        Return the stored settings as phase info.

        Returns
        -------
        tuple
            name : str
                object label
            phase_info : dict
                stored settings
        '''
        subsystem_options = self.subsystem_options  # TODO: aero info?
        user_options = dict(self.user_options)
        initial_guesses = dict(self.initial_guesses)

        # TODO some of these may be purely programming API hooks, rather than for use
        # with phase info
        # - ode_class
        # - transcription
        # - external_subsystems
        # - meta_data

        phase_info = dict(
            subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses)

        return (self.name, phase_info)

    @classmethod
    def from_phase_info(cls, name, phase_info: dict, core_subsystems=None, meta_data=None):
        '''
        Return a new phase builder based on the specified phase info.

        Note, calling code is responsible for matching phase info to the correct phase
        builder type, or the behavior is undefined.

        Parameters
        ----------
        name : str
            object label
        phase_info : dict
            stored settings
        '''
        # loop over user_options dict entries
        # if the value is not a tuple, wrap it in a tuple with the second entry of 'unitless'
        for key, value in phase_info['user_options'].items():
            if not isinstance(value, tuple):
                phase_info['user_options'][key] = (value, 'unitless')

        subsystem_options = phase_info.get(
            'subsystem_options', {})  # TODO: aero info?
        user_options = AviaryValues(phase_info.get('user_options', ()))
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
            name, subsystem_options=subsystem_options, user_options=user_options,
            initial_guesses=initial_guesses, meta_data=meta_data,
            core_subsystems=core_subsystems, external_subsystems=external_subsystems)

        return phase_builder

    @classmethod
    def _add_meta_data(cls, name, *, val, units='unitless', desc=None):
        '''
        Update supported options with a new item.

        Raises
        ------
        ValueError
            if a repeat option is found
        '''
        meta_data = cls._meta_data_

        if name in meta_data:
            raise ValueError(
                f'{cls.__name__}": meta data: repeat option: {name}'
            )

        meta_data[name] = dict(val=val, units=units, desc=desc)

    @classmethod
    def _add_initial_guess_meta_data(cls, initial_guess: InitialGuess, desc=None):
        '''
        Update supported initial guesses with a new item.

        Raises
        ------
        ValueError
            if a repeat initial guess is found
        '''
        meta_data = cls._initial_guesses_meta_data_
        name = initial_guess.key

        if name in meta_data:
            raise ValueError(
                f'{cls.__name__}": meta data: repeat initial guess: {name}'
            )

        meta_data[name] = dict(
            apply_initial_guess=initial_guess.apply_initial_guess, desc=desc)


_registered_phase_builder_types = []


def register(phase_builder_t=None, *, check_repeats=True):
    '''
    Register a new phase builder type.

    Note, this function qualifies as a class decorator for ease of use.

    Returns
    -------
    '''
    if phase_builder_t is None:
        def decorator(phase_builder_t):
            return register(phase_builder_t, check_repeats=check_repeats)

        return decorator

    if check_repeats and (phase_builder_t in _registered_phase_builder_types):
        raise ValueError('repeated phase builder type')

    _registered_phase_builder_types.append(phase_builder_t)

    return phase_builder_t


def phase_info_to_builder(name: str, phase_info: dict) -> PhaseBuilderBase:
    '''
    Return a new phase builder based on the specified phase info.

    Note, the type of phase builder will be determined by calling
    phase_builder_t.from_phase_info() for each registered type in order of registration;
    the first result that is not None will be returned. If a supported phase builder type
    cannot be determined, raise ValueError.

    Raises
    ------
    ValueError
        if a supported phase builder type cannot be determined
    '''
    phase_builder_t: PhaseBuilderBase = None

    for phase_builder_t in _registered_phase_builder_types:
        builder = phase_builder_t.from_phase_info(name, phase_info)

        if builder is not None:
            return builder

    raise ValueError(f'unsupported phase info: {name}')


if __name__ == '__main__':
    help(PhaseBuilderBase)

    try:
        PhaseBuilderBase._add_meta_data('test', val=0, units=None, desc='test')

    except Exception as error:
        print(error)

    try:
        PhaseBuilderBase._add_initial_guess_meta_data(
            InitialGuessTime(), desc='test initial guess')

    except Exception as error:
        print(error)
