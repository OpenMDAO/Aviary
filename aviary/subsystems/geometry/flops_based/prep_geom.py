"""
Define utilities to prepare derived values of aircraft geometry for
aerodynamics analysis.

TODO: blended-wing-body support
TODO: multiple engine model support
"""

import numpy as np
import openmdao.api as om
from numpy import pi

from aviary.subsystems.geometry.flops_based.characteristic_lengths import (
    BWBWingCharacteristicLength,
    NacelleCharacteristicLength,
    OtherCharacteristicLengths,
    WingCharacteristicLength,
)
from aviary.subsystems.geometry.flops_based.fuselage import (
    BWBDetailedCabinLayout,
    BWBFuselagePrelim,
    BWBSimpleCabinLayout,
    DetailedCabinLayout,
    FuselagePrelim,
    SimpleCabinLayout,
)
from aviary.subsystems.geometry.flops_based.wetted_area_total import WettedAreaGroup
from aviary.subsystems.geometry.flops_based.wetted_area_total import TotalWettedArea
from aviary.subsystems.geometry.flops_based.wing import WingPrelim
from aviary.subsystems.geometry.flops_based.wing_detailed_bwb import (
    BWBUpdateDetailedWingDist,
    BWBComputeDetailedWingDist,
    BWBWingPrelim,
)
from aviary.variable_info.enums import AircraftTypes, Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class PrepGeom(om.Group):
    """Prepare derived values of aircraft geometry for aerodynamics analysis."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.SIMPLE_LAYOUT)
        add_aviary_option(self, Aircraft.Design.TYPE)
        add_aviary_option(self, Aircraft.BWB.DETAILED_WING_PROVIDED)

    def setup(self):
        is_simple_layout = self.options[Aircraft.Fuselage.SIMPLE_LAYOUT]
        design_type = self.options[Aircraft.Design.TYPE]

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            if is_simple_layout:
                self.add_subsystem(
                    'fuselage_layout',
                    BWBSimpleCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'fuselage_layout',
                    BWBDetailedCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

            if self.options[Aircraft.BWB.DETAILED_WING_PROVIDED]:
                self.add_subsystem(
                    'detailed_wing',
                    BWBUpdateDetailedWingDist(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'detailed_wing',
                    BWBComputeDetailedWingDist(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
        elif design_type is AircraftTypes.TRANSPORT:
            if is_simple_layout:
                self.add_subsystem(
                    'fuselage_layout',
                    SimpleCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )
            else:
                self.add_subsystem(
                    'fuselage_layout',
                    DetailedCabinLayout(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'fuselage_prelim',
                BWBFuselagePrelim(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'fuselage_prelim', FuselagePrelim(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_prelim', BWBWingPrelim(), promotes_inputs=['*'], promotes_outputs=['*']
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'wing_prelim', WingPrelim(), promotes_inputs=['*'], promotes_outputs=['*']
            )

        self.add_subsystem(
            'wetted_area',
            WettedAreaGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'fus_ratios', _FuselageRatios(), promotes_inputs=['aircraft*'], promotes_outputs=['*']
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_characteristic_lengths',
                BWBWingCharacteristicLength(),
                promotes_inputs=['aircraft*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'wing_characteristic_lengths',
                WingCharacteristicLength(),
                promotes_inputs=['aircraft*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'nacelle_characteristic_lengths',
            NacelleCharacteristicLength(),
            promotes_inputs=['aircraft*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'other_characteristic_lengths',
            OtherCharacteristicLengths(),
            promotes_inputs=['aircraft*'],
            promotes_outputs=['*'],
        )

        # self.connect(f'prelim.{Names.CROOT}', f'other_characteristic_lengths.{Names.CROOT}')

        self.add_subsystem(
            'total_wetted_area', TotalWettedArea(), promotes_inputs=['*'], promotes_outputs=['*']
        )


class _FuselageRatios(om.ExplicitComponent):
    """
    Calculate fuselage diameter to wing span ratio and fuselage length to diameter ratio
    of aircraft geometry for FLOPS-based aerodynamics analysis.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.REF_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')

        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')

        add_aviary_output(self, Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, units='unitless')
        add_aviary_output(self, Aircraft.Fuselage.LENGTH_TO_DIAMETER, units='unitless')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
            [
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Fuselage.REF_DIAMETER,
                Aircraft.Wing.GLOVE_AND_BAT,
            ],
        )

        self.declare_partials(
            Aircraft.Fuselage.LENGTH_TO_DIAMETER,
            [
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.REF_DIAMETER,
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs[Aircraft.Wing.AREA]
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]
        avg_diam = inputs[Aircraft.Fuselage.REF_DIAMETER]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]

        diam_to_wing_span = avg_diam / (aspect_ratio * (area - glove_and_bat)) ** 0.5

        outputs[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN] = diam_to_wing_span

        if 0.0 < avg_diam:
            length = inputs[Aircraft.Fuselage.LENGTH]

            length_to_diam = length / avg_diam
        else:
            length_to_diam = 100.0  # FLOPS default value

        outputs[Aircraft.Fuselage.LENGTH_TO_DIAMETER] = length_to_diam

    def compute_partials(self, inputs, J, discrete_inputs=None):
        area = inputs[Aircraft.Wing.AREA]
        aspect_ratio = inputs[Aircraft.Wing.ASPECT_RATIO]
        avg_diam = inputs[Aircraft.Fuselage.REF_DIAMETER]
        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]

        fact = aspect_ratio * (area - glove_and_bat)
        fact2 = 1.0 / fact**1.5

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Fuselage.REF_DIAMETER] = 1.0 / fact**0.5

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Wing.ASPECT_RATIO] = (
            -0.5 * avg_diam * (area - glove_and_bat) * fact2
        )

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Wing.AREA] = (
            -0.5 * avg_diam * aspect_ratio * fact2
        )

        J[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN, Aircraft.Wing.GLOVE_AND_BAT] = (
            0.5 * avg_diam * aspect_ratio * fact2
        )

        if 0.0 < avg_diam:
            length = inputs[Aircraft.Fuselage.LENGTH]

            J[Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.REF_DIAMETER] = (
                -length / avg_diam**2
            )

            J[Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.LENGTH] = 1.0 / avg_diam

        else:
            J[Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.REF_DIAMETER] = J[
                Aircraft.Fuselage.LENGTH_TO_DIAMETER, Aircraft.Fuselage.LENGTH
            ] = 0.0
