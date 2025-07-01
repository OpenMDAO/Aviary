import warnings
from pathlib import Path

import numpy as np
import openmdao.api as om

from aviary.utils.csv_data_file import read_data_file
from aviary.utils.data_interpolator_builder import build_data_interpolator
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic, Settings
from aviary.utils.named_values import NamedValues

aliases = {
    # whitespaces are replaced with underscores converted to lowercase before
    # comparison with keys
    'helical_mach': [
        'helical_mach',
        'mach_helical',
        'm_helical',
        'mn_helical',
        'mach_number_helical',
        'helical_mach_number',
    ],
    Dynamic.Atmosphere.MACH: ['m', 'mn', 'mach', 'mach_number'],
    'power_coefficient': ['cp', 'power_coefficient'],
    'thrust_coefficient': ['ct', 'thrust_coefficient'],
    'advance_ratio': ['j', 'advance_ratio'],
}


class PropellerMap(om.Group):
    """
    An OpenMDAO group that contains a metamodel comp for given propeller performance data as well as
    optional conversion component if required Mach number for data is helical. Used in
    PropellerPerformance.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(
            'propeller_data',
            types=NamedValues,
            default=None,
            desc='propeller performance data to be used instead of data file (optional)',
        )
        add_aviary_option(self, Aircraft.Engine.Propeller.DATA_FILE)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']
        data = self.options['propeller_data']
        data_file = self.options[Aircraft.Engine.Propeller.DATA_FILE]
        verbosity = self.options[Settings.VERBOSITY]

        if data is None:
            data, inputs, outputs = read_data_file(data_file, aliases=aliases, verbosity=verbosity)
            if verbosity > Verbosity.BRIEF:
                print(f'Reading propeller performance data from {data_file}')
        else:
            if verbosity > Verbosity.BRIEF:
                if data_file is not None:
                    warnings.warn(
                        f'Propeller performance map provided as both a data file and as data '
                        'passed in-memory. Provided data file will be not be used.'
                    )
                print(f'Reading propeller performance data from {data_file}')

        # determine the mach type from data
        mach_types = [key for key in ['mach', 'helical_mach'] if key in data]

        # Both machs being present is fine. Default to Mach number
        if len(mach_types) > 1:
            if verbosity > Verbosity.BRIEF:  # VERBOSE, DEBUG
                warnings.warn(
                    'Both Mach and helical Mach are present in propeller data file '
                    f'<{data_file}>. Using Mach number.'
                )
        # if neither Mach is present, raise an error
        if len(mach_types) == 0:
            raise UserWarning(
                'Neither Mach or helical Mach are present in propeller data file '
                f'<{data_file}>. At least one Mach input is required.'
            )

        # if propeller map requires helical mach, add a component to compute it
        if mach_types == ['helical_mach']:
            helical_mach = om.ExecComp(
                'helical_mach=(mach**2 + tip_mach**2)**0.5',
                helical_mach={'val': np.ones(nn), 'units': 'unitless'},
                mach={'val': np.ones(nn), 'units': 'unitless'},
                tip_mach={'val': np.ones(nn), 'units': 'unitless'},
            )
            self.add_subsystem(
                'helical_mach_calc', helical_mach, promotes=['*', ('mach', Dynamic.Atmosphere.MACH)]
            )

        propeller_interp = build_data_interpolator(
            interpolator_data=data,
            interpolator_outputs=outputs,
            num_nodes=nn,
        )
        self.add_subsystem('propeller_map', propeller_interp, promotes=['*'])
