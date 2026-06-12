import numpy as np
import openmdao.api as om

from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class BWBUpdateDetailedWingDist(om.ExplicitComponent):
    """
    Specify the shape using the detailed wing data capability. The root chord is redefined to be
    equal to the length of the chord at the outboard cabin wall, and another segment is added for
    the cabin itself.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.INPUT_STATION_DISTRIBUTION)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_inp_stations = len(self.options[Aircraft.Wing.INPUT_STATION_DISTRIBUTION])
        add_aviary_input(
            self,
            Aircraft.Wing.CHORD_PER_SEMISPAN_DISTRIBUTION,
            shape=num_inp_stations,
            units='unitless',
        )
        add_aviary_input(
            self,
            Aircraft.Wing.THICKNESS_TO_CHORD_DISTRIBUTION,
            shape=num_inp_stations,
            units='unitless',
        )
        add_aviary_input(
            self,
            Aircraft.Wing.LOAD_PATH_SWEEP_DISTRIBUTION,
            shape=num_inp_stations - 1,
            units='deg',
        )
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.OUTBOARD_SEMISPAN, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='ft')

        self.add_output(Aircraft.Wing.SPAN, units='ft')
        self.add_output(
            'BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', shape=num_inp_stations + 2, units='unitless'
        )
        self.add_output(
            'BWB_THICKNESS_TO_CHORD_DISTRIBUTION', shape=num_inp_stations + 2, units='unitless'
        )
        self.add_output('BWB_LOAD_PATH_SWEEP_DISTRIBUTION', shape=num_inp_stations + 1, units='deg')

    def setup_partials(self):
        nn = len(self.options[Aircraft.Wing.INPUT_STATION_DISTRIBUTION])

        self.declare_partials(
            Aircraft.Wing.SPAN,
            Aircraft.Fuselage.MAX_WIDTH,
            val=1.0,
        )

        self.declare_partials(
            Aircraft.Wing.SPAN,
            Aircraft.Wing.OUTBOARD_SEMISPAN,
            val=2.0,
        )

        # This is the toughest, so just cs.
        self.declare_partials(
            'BWB_CHORD_PER_SEMISPAN_DISTRIBUTION',
            [
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Wing.OUTBOARD_SEMISPAN,
                Aircraft.Wing.CHORD_PER_SEMISPAN_DISTRIBUTION,
                Aircraft.Wing.ROOT_CHORD,
            ],
            method='cs',
        )

        rows = np.array([0, 1])
        cols = np.array([0, 0])
        self.declare_partials(
            'BWB_THICKNESS_TO_CHORD_DISTRIBUTION',
            Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD,
            rows=rows,
            cols=cols,
            val=1.0,
        )

        rows = np.arange(nn) + 2
        cols = np.arange(nn)
        self.declare_partials(
            'BWB_THICKNESS_TO_CHORD_DISTRIBUTION',
            Aircraft.Wing.THICKNESS_TO_CHORD_DISTRIBUTION,
            rows=rows,
            cols=cols,
            val=1.0,
        )

        rows = np.arange(nn - 1) + 2
        cols = np.arange(nn - 1)
        self.declare_partials(
            'BWB_LOAD_PATH_SWEEP_DISTRIBUTION',
            Aircraft.Wing.LOAD_PATH_SWEEP_DISTRIBUTION,
            rows=rows,
            cols=cols,
            val=1.0,
        )

    def compute(self, inputs, outputs):
        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        osspan = inputs[Aircraft.Wing.OUTBOARD_SEMISPAN][0]
        wingspan = width + osspan * 2
        outputs[Aircraft.Wing.SPAN] = wingspan

        rate_span = (wingspan - width) / wingspan
        length = inputs[Aircraft.Fuselage.LENGTH][0]
        side_tc = inputs[Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD][0]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]

        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][2:] = inputs[
            Aircraft.Wing.CHORD_PER_SEMISPAN_DISTRIBUTION
        ]
        # User will input nondimensionalized chord, which we dimensionalize for use downstream.
        # Note: this code nondimensionalizes by element, which implies that the user might mix
        # dimensional and nondimensionalized elements, which seems unlikely.
        idx = np.where(outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'] < 5.0)
        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][idx] *= rate_span
        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][0] = length
        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][1] = root_chord

        outputs['BWB_THICKNESS_TO_CHORD_DISTRIBUTION'][0] = side_tc
        outputs['BWB_THICKNESS_TO_CHORD_DISTRIBUTION'][1] = side_tc
        outputs['BWB_THICKNESS_TO_CHORD_DISTRIBUTION'][2:] = inputs[
            Aircraft.Wing.THICKNESS_TO_CHORD_DISTRIBUTION
        ]

        outputs['BWB_LOAD_PATH_SWEEP_DISTRIBUTION'][0:2] = 0.0
        outputs['BWB_LOAD_PATH_SWEEP_DISTRIBUTION'][2:] = inputs[
            Aircraft.Wing.LOAD_PATH_SWEEP_DISTRIBUTION
        ]


class BWBComputeDetailedWingDist(om.ExplicitComponent):
    """
    BWB requires detailed wing. If it is not provided, it will be created. This component
    add a trapezoidal panel out to the total semispan with the root chord equal to the length
    of the chord at the outboard cabin wall, and the tip chord equal to 6% of wing span.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.INPUT_STATION_DISTRIBUTION)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.OUTBOARD_SEMISPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')

        self.add_output(Aircraft.Wing.SPAN, units='ft')
        self.add_output('BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', shape=3, units='unitless')
        self.add_output('BWB_THICKNESS_TO_CHORD_DISTRIBUTION', shape=3, units='unitless')
        self.add_output('BWB_LOAD_PATH_SWEEP_DISTRIBUTION', shape=2, units='deg')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Wing.SPAN,
            [
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Wing.OUTBOARD_SEMISPAN,
            ],
        )
        self.declare_partials(
            'BWB_CHORD_PER_SEMISPAN_DISTRIBUTION',
            [
                Aircraft.Fuselage.MAX_WIDTH,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Wing.OUTBOARD_SEMISPAN,
            ],
        )
        self.declare_partials(
            'BWB_THICKNESS_TO_CHORD_DISTRIBUTION',
            [
                Aircraft.Wing.THICKNESS_TO_CHORD,
                Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD,
            ],
        )
        self.declare_partials(
            'BWB_LOAD_PATH_SWEEP_DISTRIBUTION',
            [
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.OUTBOARD_SEMISPAN,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Fuselage.MAX_WIDTH,
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        num_inp_stations = len(self.options[Aircraft.Wing.INPUT_STATION_DISTRIBUTION])
        if num_inp_stations != 1:
            raise ValueError(
                'Aircraft.Wing.INPUT_STATION_DISTRIBUTION should be length 1, '
                f'however {num_inp_stations} values were provided.'
            )

        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        osspan = inputs[Aircraft.Wing.OUTBOARD_SEMISPAN][0]
        wingspan = width + osspan * 2
        outputs[Aircraft.Wing.SPAN] = wingspan
        length = inputs[Aircraft.Fuselage.LENGTH][0]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]

        wing_tip_chord = 0.06 * wingspan
        side_tc = inputs[Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD][0]
        tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD][0]
        sweep = inputs[Aircraft.Wing.SWEEP][0]
        tr_out = wing_tip_chord / root_chord
        ar_out = 2.0 * (2 * osspan) / (wing_tip_chord + root_chord)

        angle = np.tan(sweep / 57.2958) - 2.0 * (1 - tr_out) / (1 + tr_out) / ar_out
        swp_ld_path = 57.2958 * np.arctan(angle)

        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][0] = length
        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][1] = root_chord
        outputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION'][2] = wing_tip_chord

        outputs['BWB_THICKNESS_TO_CHORD_DISTRIBUTION'][0] = side_tc
        outputs['BWB_THICKNESS_TO_CHORD_DISTRIBUTION'][1] = side_tc
        outputs['BWB_THICKNESS_TO_CHORD_DISTRIBUTION'][2] = tc

        outputs['BWB_LOAD_PATH_SWEEP_DISTRIBUTION'][0] = 0.0
        outputs['BWB_LOAD_PATH_SWEEP_DISTRIBUTION'][1] = swp_ld_path
        # outputs['BWB_LOAD_PATH_SWEEP_DISTRIBUTION'][2] = swp_ld_path

    def compute_partials(self, inputs, J):
        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        osspan = inputs[Aircraft.Wing.OUTBOARD_SEMISPAN][0]
        wingspan = width + 2 * osspan
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]
        wing_tip_chord = 0.06 * (width + 2 * osspan)
        sweep = inputs[Aircraft.Wing.SWEEP][0]
        tr_out = 0.06 * (width + 2 * osspan) / root_chord
        ar_out = 2.0 * (2 * osspan) / (0.06 * (width + 2 * osspan) + root_chord)
        angle = np.tan(sweep / 57.2958) - 2.0 * (1 - tr_out) / (1 + tr_out) / ar_out

        J[Aircraft.Wing.SPAN, Aircraft.Fuselage.MAX_WIDTH] = 1.0
        J[Aircraft.Wing.SPAN, Aircraft.Wing.OUTBOARD_SEMISPAN] = 2.0

        J['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', Aircraft.Fuselage.LENGTH] = [1.0, 0.0, 0.0]
        J['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', Aircraft.Wing.ROOT_CHORD] = [
            0,
            1.0,
            0,
        ]
        J['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', Aircraft.Fuselage.MAX_WIDTH] = [0.0, 0.0, 0.06]
        J['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', Aircraft.Wing.OUTBOARD_SEMISPAN] = [0.0, 0.0, 0.12]

        J['BWB_THICKNESS_TO_CHORD_DISTRIBUTION', Aircraft.Wing.THICKNESS_TO_CHORD] = [0.0, 0.0, 1.0]

        J['BWB_THICKNESS_TO_CHORD_DISTRIBUTION', Aircraft.Fuselage.SIDEBODY_THICKNESS_TO_CHORD] = [
            1.0,
            1.0,
            0.0,
        ]

        dswp_ld_path_dsweep = 1 / (1 + angle**2) / np.cos(sweep / 57.2958) ** 2
        J['BWB_LOAD_PATH_SWEEP_DISTRIBUTION', Aircraft.Wing.SWEEP] = [
            0.0,
            dswp_ld_path_dsweep,
        ]

        dtr_out_dspan = 0.06 / root_chord
        dar_out_dspan = (
            2
            * (wing_tip_chord + root_chord - 0.06 * (wingspan - width))
            / (wing_tip_chord + root_chord) ** 2
        )
        dswp_ld_path_dspan = (
            57.2958
            / (1 + angle**2)
            * (
                4 * dtr_out_dspan / (1 + tr_out) ** 2 / ar_out
                + 2 * (2 / (1 + tr_out) - 1) * dar_out_dspan / ar_out**2
            )
        )
        J['BWB_LOAD_PATH_SWEEP_DISTRIBUTION', Aircraft.Wing.OUTBOARD_SEMISPAN] = [
            0.0,
            2 * dswp_ld_path_dspan,
        ]

        dtr_out_droot_chord = -wing_tip_chord / root_chord**2
        dar_out_droot_chord = (
            -2 * (wingspan - width) / (wing_tip_chord + root_chord) ** 2
        )
        dswp_ld_path_droot_chord = (
            57.2958
            / (1 + angle**2)
            * (
                4 * dtr_out_droot_chord / (1 + tr_out) ** 2 / ar_out
                + 2 * (2 / (1 + tr_out) - 1) * dar_out_droot_chord / ar_out**2
            )
        )
        J['BWB_LOAD_PATH_SWEEP_DISTRIBUTION', Aircraft.Wing.ROOT_CHORD] = [
            0.0,
            dswp_ld_path_droot_chord,
        ]

        dtr_out_dwidth = 0.0
        dar_out_dwidth = -4 * osspan * 0.06 / (wing_tip_chord + root_chord) ** 2

        dtr_out_dwidth = 0.06 / root_chord
        dar_out_dwidth = -4.0 * 0.06 * osspan / (0.06 * (width + 2 * osspan) + root_chord) ** 2
        dswp_ld_path_dwidth = (
            -2
            * 57.2958
            / (1 + angle**2)
            * (
                -2
                * (dtr_out_dwidth * ar_out + (1 + tr_out) * dar_out_dwidth)
                / ((1 + tr_out) * ar_out) ** 2
                + dar_out_dwidth / ar_out**2
            )
        )

        J['BWB_LOAD_PATH_SWEEP_DISTRIBUTION', Aircraft.Fuselage.MAX_WIDTH] = [
            0.0,
            dswp_ld_path_dwidth,
        ]


class BWBWingPrelim(om.ExplicitComponent):
    """preliminary calculations of wing aspect ratio for BWB using detailed wing information"""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.INPUT_STATION_DISTRIBUTION)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_inp_stations = len(self.options[Aircraft.Wing.INPUT_STATION_DISTRIBUTION])

        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        self.add_input(
            'BWB_CHORD_PER_SEMISPAN_DISTRIBUTION', shape=num_inp_stations + 2, units='unitless'
        )

        add_aviary_output(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_output(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_output(self, Aircraft.Wing.ASPECT_RATIO_REFERENCE, units='unitless')
        add_aviary_output(self, Aircraft.Wing.LOAD_FRACTION, units='unitless')

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        wingspan = inputs[Aircraft.Wing.SPAN][0]
        if wingspan <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Aircraft.Wing.SPAN must be positive.')
        rate_span = (wingspan - width) / wingspan

        # This part is repeated in BWBWingWettedArea()
        num_inp_stations = len(self.options[Aircraft.Wing.INPUT_STATION_DISTRIBUTION])
        num_bwb_stations = num_inp_stations + 2
        input_station_dist = self.options[Aircraft.Wing.INPUT_STATION_DISTRIBUTION]
        bwb_input_station_dist = np.zeros(num_bwb_stations, dtype=width.dtype)
        bwb_input_station_dist[2:] = input_station_dist

        bwb_input_station_dist = np.where(
            bwb_input_station_dist <= 1.0,
            bwb_input_station_dist * rate_span + width / wingspan,  # if x <= 1.0
            bwb_input_station_dist + width / 2.0,  # else
        )
        bwb_input_station_dist[0] = 0.0
        bwb_input_station_dist[1] = width / 2.0

        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH]

        ssm = 0.0
        bwb_chord_per_semispan_distribution = inputs['BWB_CHORD_PER_SEMISPAN_DISTRIBUTION']

        # Calculate Wing Area and Aspect Ratio for modified planform
        if bwb_chord_per_semispan_distribution[0] <= 5.0:
            C1 = bwb_chord_per_semispan_distribution[0] * wingspan / 2.0
        else:
            C1 = bwb_chord_per_semispan_distribution[0]
        if bwb_input_station_dist[0] <= 1.1:
            Y1 = bwb_input_station_dist[0] * wingspan / 2.0
        else:
            Y1 = bwb_input_station_dist[0]
        for n in range(1, num_bwb_stations):
            if bwb_chord_per_semispan_distribution[n] <= 5.0:
                C2 = bwb_chord_per_semispan_distribution[n] * wingspan / 2.0
            else:
                C2 = bwb_chord_per_semispan_distribution[n]
            if bwb_input_station_dist[n] <= 1.1:
                Y2 = bwb_input_station_dist[n] * wingspan / 2.0
            else:
                Y2 = bwb_input_station_dist[n]
            axp = (Y2 - Y1) * (C1 + C2)
            C1 = C2
            Y1 = Y2
            ssm = ssm + axp
        ar = wingspan**2 / (ssm - glove_and_bat)
        # Calculated wing area for aerodynamics
        outputs[Aircraft.Wing.AREA] = ssm
        outputs[Aircraft.Wing.ASPECT_RATIO] = ar
        outputs[Aircraft.Wing.ASPECT_RATIO_REFERENCE] = ar

        # Estimate the percent load carried by the outboard wing
        pct_load = (1.0 - width / wingspan) ** 2
        outputs[Aircraft.Wing.LOAD_FRACTION] = pct_load
