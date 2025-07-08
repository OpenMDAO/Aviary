import openmdao.api as om
import numpy as np

from aviary.utils.functions import promote_aircraft_and_mission_vars


class ExternalSubsystemGroup(om.Group):
    def configure(self):
        promote_aircraft_and_mission_vars(self)


def get_phase_mission_bus_lengths(traj):
    phase_mission_bus_lengths = {}
    for phase_name, phase in traj._phases.items():
        phase_mission_bus_lengths[phase_name] = phase._timeseries['mission_bus_variables'][
            'transcription'
        ].grid_data.subset_num_nodes['all']
    return phase_mission_bus_lengths


def process_guess_var(val, key, phase):
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
            if 'controls:' in key or 'states:' in key:
                # If so, strip the first part of the key to match the variable name
                # in phase
                stripped_key = ':'.join(key.split(':')[1:])

                # Interpolate the initial guess values across the phase's domain
                val = phase.interp(stripped_key, xs=xs, ys=val)
            else:
                # If not a control or state variable, interpolate the initial guess
                # values directly
                val = phase.interp(key, xs=xs, ys=val)

    # Return the processed guess value(s)
    return val
