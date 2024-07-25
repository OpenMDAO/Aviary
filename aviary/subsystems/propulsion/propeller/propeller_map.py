import numpy as np
import warnings
import openmdao.api as om
from openmdao.utils.units import convert_units

from aviary.subsystems.propulsion.utils import PropellerModelVariables, default_propeller_units
from aviary.utils.aviary_values import AviaryValues, NamedValues, get_keys
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.functions import get_path
from aviary.variable_info.enums import OutMachType
from aviary.variable_info.variables import Aircraft, Settings


MACH = PropellerModelVariables.MACH
CP = PropellerModelVariables.CP
CT = PropellerModelVariables.CT
J = PropellerModelVariables.J

aliases = {
    # whitespaces are replaced with underscores converted to lowercase before
    # comparison with keys
    MACH: ['m', 'mn', 'mach', 'mach_number', 'helical_mach'],
    CP: ['cp', 'power_coefficient'],
    CT: ['ct', 'thrust_coefficient'],
    J: ['j', 'advance_ratio'],
}


class PropellerMap(om.ExplicitComponent):
    """
    This class loads a user provided propeller map into memory and builds a propeller.
    Attributes
    ----------
    name : str ('propeller')
        Object label.
    options : AviaryValues (<empty>)
        Inputs and options related to propeller model.
    data : NamedVaues (<empty>), optional
        propeller map data.
    mach_type: OutMachType (MACH or HELICAL_MACH)
    """

    def __init__(self, name='propeller', options: AviaryValues = None,
                 data: NamedValues = None):
        super().__init__()

        # working copy of propeller performance data, is modified during data pre-processing
        self.data = {key: np.array([]) for key in PropellerModelVariables}

        # Create dict for variables present in propeller data with associated units
        self.propeller_variables = {}

        data_file = options.get_val(Aircraft.Engine.PROPELLER_DATA_FILE)
        self._read_data(data_file)

    def _read_data(self, data_file):
        # read csv file
        raw_data = read_data_file(data_file, aliases=aliases)

        message = f'<{data_file}>'
        # Loop through all variables in provided data. Track which valid variables are
        #    included with the data and save raw data for reference
        for key in get_keys(raw_data):
            val, units = raw_data.get_item(key)
            if key in aliases:
                # Convert data to expected units. Required so settings like tolerances
                # that assume units work as expected
                try:
                    val = np.array([convert_units(i, units, default_propeller_units[key])
                                   for i in val])
                except TypeError:
                    raise TypeError(f"{message}: units of '{units}' provided for "
                                    f'<{key.name}> are not compatible with expected units '
                                    f'of {default_propeller_units[key]}')

                # propeller_variables currently only used to store "valid" engine variables
                # as defined in PropellerModelVariables Enum
                self.propeller_variables[key] = default_propeller_units[key]

            else:
                if self.get_val(Settings.VERBOSITY).value >= 1:
                    warnings.warn(
                        f'{message}: header <{key}> was not recognized, and will be skipped')

            self.data[key] = val

        if not self.propeller_variables:
            raise UserWarning(
                f'No valid propeller variables found in data for {message}')

    def read_and_set_mach_type(self, data_file):
        # read the mach type from header.
        # sample header:
        #     J, Helical_Mach,         CP,         CT

        m_type = 'mach'  # default to freestream Mach number
        m_type_define = False
        fp = get_path(data_file)
        with open(fp, "r") as f:
            for line in f:
                tokens = line.split(',')
                if len(tokens) > 1:
                    s = tokens[1].strip().lower()
                    if s == 'mach' or s == 'helical_mach':
                        m_type = s
                        m_type_define = True
                        break

        if not m_type_define:
            warnings.warn(
                f"String 'mach_type' is not defined. Assume freestream Mach in the table.")

        return OutMachType.get_element_by_value(m_type)

    def build_propeller_interpolator(self, num_nodes, options=None):
        """
        Builds the OpenMDAO metamodel component for the propeller map.
        """
        interp_method = options.get_val(Aircraft.Engine.INTERPOLATION_METHOD)
        # interpolator object for propeller data
        propeller = om.MetaModelSemiStructuredComp(
            method=interp_method, extrapolate=True, vec_size=num_nodes)

        # add inputs and outputs to interpolator
        # depending on p, selected_mach can be Mach number (Dynamic.Mission.MACH) or helical Mach number
        propeller.add_input('selected_mach',
                            self.data[MACH],
                            units='unitless',
                            desc='Current Mach number (flight or helical)')
        propeller.add_input('power_coefficient',
                            self.data[CP],
                            units='unitless',
                            desc='Current power coefficient')
        propeller.add_input('advance_ratio',
                            self.data[J],
                            units='unitless',
                            desc='Current advance ratio')
        propeller.add_output('thrust_coefficient',
                             self.data[CT],
                             units='unitless',
                             desc='Current thrust coefficient')
        return propeller
