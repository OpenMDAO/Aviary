# Import the necessary things
import aviary.api as av
from aviary.variable_info.variables import Mission, Aircraft
from aviary.utils.functions import convert_strings_to_data, set_value
from aviary.variable_info.variable_meta_data import _MetaData
import json
import numpy
import enum


def _read_sizing_json(aviary_problem, json_filename):
    """
    This function reads in an aviary problem object from a json file.

    Parameters
    ----------
    aviary_problem: OpenMDAO Aviary Problem
        Aviary problem object optimized for the aircraft design/sizing mission.
        Assumed to contain aviary_inputs and Mission.Summary.GROSS_MASS
    json_filename:   string
        User specified name and relative path of json file to save the data into

    Returns
    ----------
    Aviary Problem object with updated input values from json file

    """
    # load saved input list from json file
    with open(json_filename) as json_data_file:
        loaded_aviary_input_list = json.load(json_data_file)
        json_data_file.close()

    # Loop over input list and assign aviary problem input values
    counter = 0  # list index tracker
    for inputs in loaded_aviary_input_list:
        [var_name, var_values, var_units, var_type] = inputs

        # Initialize some flags to idetify arrays and enums
        is_array = False
        is_enum = False

        if var_type == "<class 'numpy.ndarray'>":
            is_array = True

        elif var_type == "<class 'list'>":
            # check if the list contains enums
            for i in range(len(var_values)):
                if isinstance(var_values[i], str):
                    if var_values[i].find("<") != -1:
                        # Found a list of enums: set the flag
                        is_enum = True

                        # Manipulate the string to find the value
                        tmp_var_values = var_values[i].split(':')[-1]
                        var_values[i] = tmp_var_values.replace(">", "").replace(
                            "]", "").replace("'", "").replace(" ", "")

            if is_enum:
                var_values = convert_strings_to_data(var_values)

            else:
                var_values = [var_values]

        elif var_type.find("<enum") != -1:
            # Identify enums and manipulate the string to find the value
            tmp_var_values = var_values.split(':')[-1]
            var_values = tmp_var_values.replace(">", "").replace(
                "]", "").replace("'", "").replace(" ", "")
            var_values = convert_strings_to_data([var_values])

        else:
            # values are expected to be parsed as a list to set_value function
            var_values = [var_values]

        # Check if the variable is in meta data
        if var_name in _MetaData.keys():
            try:
                aviary_problem.aviary_inputs = set_value(
                    var_name, var_values, aviary_problem.aviary_inputs, units=var_units, is_array=is_array, meta_data=_MetaData)
            except:
                # Print helpful error
                print("FAILURE: list_num = ", counter, "Input String = ", inputs,
                      "Attempted to set_value(", var_name, ",", var_values, ",", var_units, ")")
        else:
            # Not in the MetaData
            print("Name not found in MetaData: list_num =", counter, "Input String =",
                  inputs, "Attempted set_value(", var_name, ",", var_values, ",", var_units, ")")

        counter = counter + 1  # increment index tracker
    return aviary_problem


def load_off_design(json_filename, ProblemType, phase_info, payload, mission_range, mission_gross_mass):
    """
    This function loads a sized aircraft, and sets up an aviary problem to run a specified off design mission.

    Parameters
    ----------
    json_filename:      string
        User specified name and relative path of json file containing the sized aircraft data
    ProblemType:        enum
        Alternate or Fallout. Alternate requires mission_range input and Fallout requires mission_fuel input
    phase_info:     phase_info dictionary for off design mission
    payload:            float
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS
    mission_range       float
        Mission.Summary.RANGE 'NM'
    mission_gross_mass  float
        Mission.Summary.GROSS_MASS 'lbm'

    Returns
    ----------
    Aviary Problem object with completed load_inputs() for specified off design mission
    """

    # Initialize a new aviary problem and aviary_input data structure
    prob = av.AviaryProblem()
    prob.aviary_inputs = av.AviaryValues()

    prob = _read_sizing_json(prob, json_filename)

    # Update problem type
    prob.problem_type = ProblemType
    prob.aviary_inputs.set_val('settings:problem_type', ProblemType, units='unitless')

    # Set Payload
    prob.aviary_inputs.set_val(
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, payload, units='lbm')

    if ProblemType == ProblemType.ALTERNATE:
        # Set mission range, aviary will calculate required fuel
        prob.aviary_inputs.set_val(Mission.Design.RANGE, mission_range, units='NM')

    elif ProblemType == ProblemType.FALLOUT:
        # Set mission fuel and calculate gross weight, aviary will calculate range
        prob.aviary_inputs.set_val(Mission.Summary.GROSS_MASS,
                                   mission_gross_mass, units='lbm')

    # Load inputs
    prob.load_inputs(prob.aviary_inputs, phase_info)

    return prob
