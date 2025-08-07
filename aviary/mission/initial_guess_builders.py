"""
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

InitialGuessIntegrationVariable : a utility for setting guesses for initial value and duration
of an integration variable in a problem
"""

from collections.abc import Sequence

import dymos as dm
import numpy as np
import openmdao.api as om
from dymos.transcriptions.transcription_base import TranscriptionBase


class InitialGuess:
    """
    Define a utility for setting an initial guess on a problem.

    Attributes
    ----------
    key : str
        base name for the initial guess
    """

    __slots__ = ('key',)

    def __init__(self, key):
        self.key = key

    def apply_initial_guess(
        self, prob: om.Problem, traj_name, phase: dm.Phase, phase_name, val, units
    ):

        if self.key in phase.state_options:
            phase.set_state_val(self.key, val, units=units)
        elif self.key in phase.control_options:
            phase.set_control_val(self.key, val, units=units)
        elif self.key in phase.parameter_options:
            phase.set_parameter_val(self.key, val, units=units)
        elif self.key == phase.time_options['name']:
            prob.set_integ_var_val(initial=val[0], duration=val[1], units=units)
        else:
            raise ValueError(f'{phase.msginfo} Attempting to apply initial guess for {self.key}.\n'
                             'Not find in the states, control, parameters, or integration variable of the phase.')

    # def _get_complete_key(self, traj_name, phase_name):
    #     """Compose the complete key for setting the initial guess."""
    #     _ = traj_name
    #     _ = phase_name

    #     return self.key


class InitialGuessControl(InitialGuess):
    """
    Define a utility for setting an initial guess for a control on a problem.

    Attributes
    ----------
    key : str
        base name for the control
    """

    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        """Compose the complete key for setting the control initial guess."""
        key = f'{traj_name}.{phase_name}.controls:{self.key}'

        return key


class InitialGuessParameter(InitialGuess):
    """
    Define a utility for setting an initial guess for a parameter on a problem.

    Attributes
    ----------
    key : str
        base name for the parameter
    """

    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        """Compose the complete key for setting the parameter initial guess."""
        key = f'{traj_name}.{phase_name}.parameters:{self.key}'

        return key


class InitialGuessPolynomialControl(InitialGuess):
    """
    Define a utility for setting an initial guess for a polynomial control on a problem.

    Attributes
    ----------
    key : str
        base name for the polynomial control
    """

    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        """Compose the complete key for setting the polynomial control initial guess."""
        key = f'{traj_name}.{phase_name}.controls:{self.key}'
        return key


class InitialGuessState(InitialGuess):
    """
    Define a utility for setting an initial guess for a state on a problem.

    Attributes
    ----------
    key : str
        base name for the state
    """

    __slots__ = ()

    def _get_complete_key(self, traj_name, phase_name):
        """Compose the complete key for setting the state initial guess."""
        key = f'{traj_name}.{phase_name}.states:{self.key}'

        return key


class InitialGuessIntegrationVariable(InitialGuess):
    """
    Define a utility for setting guesses for the initial and duration values
    for the integration variable, usually time. We might also use this for
    other integration variables, such as velocity or distance.

    The default name for the variable here is "time".

    Attributes
    ----------
    key : str ('time')
        the group identifier for guesses for initial integration variable value and duration
    """

    __slots__ = ()

    def __init__(self, key='time'):
        super().__init__(key)

    def apply_initial_guess(
        self, prob: om.Problem, traj_name, phase: dm.Phase, phase_name, val, units
    ):
        t_initial, t_duration = val
        phase.set_integ_var_val(initial=t_initial, duration=t_duration, units=units)
