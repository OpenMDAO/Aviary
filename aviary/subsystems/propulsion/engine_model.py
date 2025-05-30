"""
Define utilities for building engine models.

Classes
-------
EngineModel : the interface for an engine model builder.
"""

import warnings

import numpy as np

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.variables import Settings


class EngineModel(SubsystemBuilderBase):
    """
    Define the interface for an engine model builder.

    Attributes
    ----------
    name : str ('engine_model')
        object label.
    options : AviaryValues (<empty>)
        inputs and options related to engine model.

    Methods
    -------
    build_pre_mission
    build_mission
    build_post_mission
    get_val
    get_item
    set_val
    update
    """

    default_name = 'engine_model'

    def __init__(
        self, name: str = None, options: AviaryValues = None, meta_data: dict = None, **kwargs
    ):
        super().__init__(name, meta_data=meta_data)
        if options is not None:
            self.options = options.deepcopy()
        else:
            self.options = AviaryValues()

        # Hybrid throttle is currently the only optional independent variable, requiring
        # this flag so Aviary knows how to handle EngineModels during mission
        self.use_hybrid_throttle = False

        self._preprocess_inputs()

    def _setup(self, **kwargs):
        """
        Perform setup of EngineModel.

        All attributes and values for options in EngineModel are finalized for
        analysis.
        """
        DeprecationWarning(
            'EngineModel._setup() is redundant and will be removed in a future update.'
        )
        self._preprocess_inputs()

    def build_pre_mission(self, aviary_inputs, **kwargs):
        """
        Build an OpenMDAO system for the pre-mission computations of the engine model,
        such as sizing.

        Optional for engine models.
        Used in propulsion_sizing.py to build the pre-mission propulsion subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        """
        return None

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        """
        Build an OpenMDAO system for the mission computations of the engine model.

        Required for engine models.
        Used in propulsion_mission.py to build the propulsion mission system.

        Returns
        -------
        mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen
            during the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary during
            the mission.
        """
        raise NotImplementedError(
            'build_mission() is a required method but has not '
            f'been implemented in EngineModel <{self.name}>'
        )

    def build_post_mission(self, aviary_inputs, phase_info, phase_mission_bus_lengths, **kwargs):
        """
        Build an OpenMDAO system for the post-mission computations of the engine model.

        Optional for engine models. Currently unused by core Aviary propulsion.

        Returns
        -------
        post_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen
            after the mission. This includes time-dependent states that are
            being integrated as well as any other variables that vary after
            the mission.
        """
        return None

    def _preprocess_inputs(self):
        """
        Raises TypeError if options are not an instance of AviaryValues (or None).

        Creates warning if a provided engine option is vectorized, and only accepts
        the first entry in that vector into self.options.
        """
        options = self.options

        if options is None:
            return  # options are allowed to be empty

        # verbosity settings are needed to adjust printouts
        if Settings.VERBOSITY not in options:
            self.set_val(Settings.VERBOSITY, Verbosity.BRIEF)

        verbosity = self.get_val(Settings.VERBOSITY).value

        if not isinstance(options, AviaryValues):
            raise TypeError('EngineModel options must be an AviaryValues object')

        for key, (val, units) in options:
            # only perform vector check for variables related to engines and nacelles
            if key.startswith('aircraft:engine:') or key.startswith('aircraft:nacelle'):
                # if val is an iterable...
                if type(val) in (list, np.ndarray, tuple):
                    # but meta_data says it is not supposed to be...
                    if not isinstance(
                        self.meta_data[key]['default_value'], (list, np.ndarray, tuple)
                    ):
                        # if val is multidimensional, raise error
                        if isinstance(val[0], (list, np.ndarray, tuple)):
                            raise UserWarning(
                                f'Multidimensional {type(val)} was given for variable '
                                f'{key} in EngineModel <{self.name}>, but '
                                f'{type(self.meta_data[key]["default_value"])} '
                                'was expected.'
                            )
                        # use first item in val and warn user
                        if verbosity >= 1:
                            if len(val) > 1:
                                warnings.warn(
                                    f'The value of {key} passed to EngineModel '
                                    f'<{self.name}> is {type(val)}. Only the first '
                                    'entry in this iterable will be used.'
                                )

                    # if val is supposed to be an iterable...
                    else:
                        # but val is multidimensional, use first item and warn user
                        if isinstance(val[0], (list, np.ndarray, tuple)):
                            warnings.warn(
                                f'The value of {key} passed to EngineModel '
                                f'<{self.name}> is multidimensional {type(val)}. Only '
                                'the first entry in this iterable will be used.'
                            )
                        # and val is 1-D, then it is ok!
                        else:
                            continue

                    if isinstance(val, np.ndarray):
                        # "Convert" numpy types to standard Python types. Wrap first
                        # index in numpy array before calling item() to safeguard against
                        # non-standard types, such as objects
                        val = np.array(val[0]).item()
                    else:
                        val = val[0]
                    # update options with single value (instead of vector)
                    options.set_val(key, val, units)
            # Currently assuming that EngineModels might care about non-engine variables,
            # so they are being kept in self.options
            # else:
            #     options.delete(key)

    def update(self, options: AviaryValues, **kwargs):
        """Given a new set of AviaryValues, update the engine model and rerun setup."""
        self.options = options.deepcopy()

        self._setup(**kwargs)

    def get_val(self, key, units='unitless'):
        """
        Returns desired value from options in specified units.

        Parameters
        ----------
        key : str
            Name of requested option.
        units : str
            Unit requested option value should be converted to.

        Returns
        -------
        val
            Value of requested option in desired units.
        """
        return self.options.get_val(key, units)

    def get_item(self, key, default=(None, None)):
        """
        Return the named value and its associated units.

        Note, this method never raises `KeyError` or `TypeError`.

        Parameters
        ----------
        key : str
            the name of the item

        default : OptionalValueAndUnits (None, None)
            if the item does not exist, return this object

        Returns
        -------
        OptionalValueAndUnits
        """
        return self.options.get_item(key, default)

    def set_val(self, key, val, units='unitless'):
        """
        Updates desired value in options with specified units.

        Parameters
        ----------
        key : str
            Name of option whose value will be updated.
        val
            New value for option.
        units : str
            Unit of val.
        """
        self.options.set_val(key, val, units)
