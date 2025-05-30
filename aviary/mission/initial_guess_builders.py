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
        """Set the initial guess on the problem."""
        complete_key = self._get_complete_key(traj_name, phase_name)

        # TODO: this is a short term hack in need of an appropriate long term solution
        #    - to interpolate, or not to interpolate: that is the question
        #    - the solution should probably be a value decoration (a wrapper) that is
        #      both lightweight and easy to check and unpack
        if isinstance(val, np.ndarray) or (isinstance(val, Sequence) and not isinstance(val, str)):
            val = phase.interp(self.key, val)

        try:
            prob.set_val(complete_key, val, units)
        except KeyError:
            complete_key = complete_key.replace('polynomial_controls', 'controls')
            prob.set_val(complete_key, val, units)

    def _get_complete_key(self, traj_name, phase_name):
        """Compose the complete key for setting the initial guess."""
        _ = traj_name
        _ = phase_name

        return self.key


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
        _ = phase

        name = f'{traj_name}.{phase_name}.t_initial'
        t_initial, t_duration = val
        prob.set_val(name, t_initial, units)

        name = f'{traj_name}.{phase_name}.t_duration'
        prob.set_val(name, t_duration, units)
