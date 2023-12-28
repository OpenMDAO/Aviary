import inspect


def add_subsystem_variables_to_phase(phase, phase_name, external_subsystems):
    """
    Add subsystem variables like states, controls, parameters, and constraints to a phase.

    Parameters
    ----------
    phase : object
        The phase object to which variables are to be added.
    phase_name : str
        The name of the phase for which the variables are added. Useful for cases
        where the subsystem behavior varies based on phase name.
    external_subsystems : list
        List of external subsystem objects that contain variables to add to the phase.

    Returns
    -------
    phase : object
        The modified phase object with added variables.

    """

    # Loop through each subsystem in the list of external_subsystems
    for subsystem in external_subsystems:

        # Fetch the states from the current subsystem
        subsystem_states = subsystem.get_states()

        # Add each state and its corresponding arguments to the phase
        for state_name in subsystem_states:
            kwargs = subsystem_states[state_name]
            phase.add_state(state_name, **kwargs)

        # Check if 'get_controls' function in the subsystem accepts 'phase_name' as an argument
        arg_spec = inspect.getfullargspec(subsystem.get_controls)
        if 'phase_name' in arg_spec.args:
            controls = subsystem.get_controls(phase_name=phase_name)
        else:
            controls = subsystem.get_controls()

        for control_name in controls:
            kwargs = controls[control_name]
            phase.add_control(control_name, **kwargs)

        constraints = subsystem.get_constraints()

        # Add each constraint and its corresponding arguments to the phase
        for constraint_name in constraints:
            kwargs = constraints[constraint_name]
            if kwargs['type'] == 'boundary':
                kwargs.pop('type')
                phase.add_boundary_constraint(constraint_name, **kwargs)
            elif kwargs['type'] == 'path':
                kwargs.pop('type')
                phase.add_path_constraint(constraint_name, **kwargs)

    return phase


def get_initial(status, key, status_for_this_variable=False):
    # Check if status is a dictionary.
    # If so, return the value corresponding to the key or status_for_this_variable if the key is not found.
    # If not, return the value of status.
    if isinstance(status, dict):
        if key in status:
            status_for_this_variable = status[key]
    elif isinstance(status, bool):
        status_for_this_variable = status
    return status_for_this_variable
