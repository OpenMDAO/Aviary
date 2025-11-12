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
        add_aviary_option(self, Aircraft.Wing.INPUT_STATION_DIST)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_stations = len(self.options[Aircraft.Wing.INPUT_STATION_DIST])
        add_aviary_input(
            self, Aircraft.Wing.CHORD_PER_SEMISPAN_DIST, shape=num_stations, units='unitless'
        )
        add_aviary_input(
            self, Aircraft.Wing.THICKNESS_TO_CHORD_DIST, shape=num_stations, units='unitless'
        )
        add_aviary_input(self, Aircraft.Wing.LOAD_PATH_SWEEP_DIST, shape=num_stations, units='deg')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='ft')
        self.add_input(
            'Rear_spar_percent_chord',
            0.7,
            units='unitless',
            desc='RSPSOB: Rear spar percent chord for BWB at side of body',
        )

        self.add_output('BWB_INPUT_STATION_DIST', shape=num_stations, units='unitless')
        self.add_output('BWB_CHORD_PER_SEMISPAN_DIST', shape=num_stations, units='unitless')
        self.add_output('BWB_THICKNESS_TO_CHORD_DIST', shape=num_stations, units='unitless')
        self.add_output('BWB_LOAD_PATH_SWEEP_DIST', shape=num_stations, units='deg')

    def setup_partials(self):
        self.declare_partials('BWB_INPUT_STATION_DIST', '*', method='fd', form='forward')

        self.declare_partials('BWB_CHORD_PER_SEMISPAN_DIST', '*', method='fd', form='forward')

        self.declare_partials(
            'BWB_THICKNESS_TO_CHORD_DIST',
            [
                Aircraft.Wing.THICKNESS_TO_CHORD_DIST,
                Aircraft.Wing.THICKNESS_TO_CHORD,
            ],
        )

        self.declare_partials(
            'BWB_LOAD_PATH_SWEEP_DIST',
            Aircraft.Wing.LOAD_PATH_SWEEP_DIST,
            val=1.0,
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        wingspan = inputs[Aircraft.Wing.SPAN][0]
        rate_span = (wingspan - width) / wingspan
        length = inputs[Aircraft.Fuselage.LENGTH][0]
        tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD][0]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]
        rear_spar_percent_chord = inputs['Rear_spar_percent_chord'][0]
        if rear_spar_percent_chord <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Rear_spar_percent_chord must be positive.')
        xl_out = root_chord / rear_spar_percent_chord

        bwb_input_station_dist = np.array(
            self.options[Aircraft.Wing.INPUT_STATION_DIST], dtype=float
        )
        bwb_input_station_dist = np.where(
            bwb_input_station_dist <= 1.0,
            bwb_input_station_dist * rate_span + width / wingspan,  # if x <= 1.0
            bwb_input_station_dist + width / 2.0,  # else
        )
        bwb_input_station_dist[0] = 0.0
        bwb_input_station_dist[1] = width / 2.0
        outputs['BWB_INPUT_STATION_DIST'] = bwb_input_station_dist

        outputs['BWB_CHORD_PER_SEMISPAN_DIST'] = inputs[Aircraft.Wing.CHORD_PER_SEMISPAN_DIST]
        idx = np.where(inputs[Aircraft.Wing.CHORD_PER_SEMISPAN_DIST] < 5.0)
        outputs['BWB_CHORD_PER_SEMISPAN_DIST'][idx] *= rate_span
        outputs['BWB_CHORD_PER_SEMISPAN_DIST'][0] = length
        outputs['BWB_CHORD_PER_SEMISPAN_DIST'][1] = xl_out

        outputs['BWB_THICKNESS_TO_CHORD_DIST'][0] = tc
        outputs['BWB_THICKNESS_TO_CHORD_DIST'][1] = tc
        outputs['BWB_THICKNESS_TO_CHORD_DIST'][2:] = inputs[Aircraft.Wing.THICKNESS_TO_CHORD_DIST][
            2:
        ]

        outputs['BWB_LOAD_PATH_SWEEP_DIST'][:] = inputs[Aircraft.Wing.LOAD_PATH_SWEEP_DIST]

    def compute_partials(self, inputs, J):
        # width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        # wingspan = inputs[Aircraft.Wing.SPAN][0]
        # rate_span = (wingspan - width) / wingspan
        # length = inputs[Aircraft.Fuselage.LENGTH][0]
        # tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD][0]
        # root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]
        # rear_spar_percent_chord = inputs['Rear_spar_percent_chord'][0]
        # xl_out = root_chord / rear_spar_percent_chord

        num_stations = len(self.options[Aircraft.Wing.INPUT_STATION_DIST])

        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD][0] = 1.0
        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD][1] = 1.0
        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD][2:] = 0.0

        diag2_matrix = np.identity(num_stations)
        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD_DIST] = diag2_matrix
        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD_DIST][0] = 0.0
        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD_DIST][1] = 0.0


class BWBComputeDetailedWingDist(om.ExplicitComponent):
    """
    BWB requires detailed wing. If it is not provided, it will be created. This component
    add a trapezoidal panel out to the total semispan with the root chord equal to the length
    of the chord at the outboard cabin wall, and the tip chord equal to 6% of wing span.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.INPUT_STATION_DIST)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='ft')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')
        self.add_input(
            'Rear_spar_percent_chord',
            0.7,
            units='unitless',
            desc='RSPSOB: Rear spar percent chord for BWB at side of body',
        )

        self.add_output('BWB_INPUT_STATION_DIST', shape=3, units='unitless')
        self.add_output('BWB_CHORD_PER_SEMISPAN_DIST', shape=3, units='unitless')
        self.add_output('BWB_THICKNESS_TO_CHORD_DIST', shape=3, units='unitless')
        self.add_output('BWB_LOAD_PATH_SWEEP_DIST', shape=3, units='deg')

    def setup_partials(self):
        self.declare_partials(
            'BWB_INPUT_STATION_DIST',
            [
                Aircraft.Fuselage.MAX_WIDTH,
            ],
        )
        self.declare_partials(
            'BWB_CHORD_PER_SEMISPAN_DIST',
            [
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Wing.SPAN,
                'Rear_spar_percent_chord',
            ],
        )
        self.declare_partials('BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD)
        self.declare_partials(
            'BWB_LOAD_PATH_SWEEP_DIST',
            [
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Fuselage.MAX_WIDTH,
                'Rear_spar_percent_chord',
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        wingspan = inputs[Aircraft.Wing.SPAN][0]
        length = inputs[Aircraft.Fuselage.LENGTH][0]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]
        rear_spar_percent_chord = inputs['Rear_spar_percent_chord'][0]
        if rear_spar_percent_chord <= 0.0:
            if verbosity > Verbosity.BRIEF:
                print('Rear_spar_percent_chord must be positive.')
        xl_out = root_chord / rear_spar_percent_chord
        wing_tip_chord = 0.06 * wingspan
        tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD][0]
        sweep = inputs[Aircraft.Wing.SWEEP][0]
        tr_out = wing_tip_chord / xl_out
        ar_out = 2.0 * (wingspan - width) / (wing_tip_chord + xl_out)

        angle = np.tan(sweep / 57.2958) - 2.0 * (1 - tr_out) / (1 + tr_out) / ar_out
        swp_ld_path = 57.2958 * np.arctan(angle)

        outputs['BWB_INPUT_STATION_DIST'][0] = 0.0
        outputs['BWB_INPUT_STATION_DIST'][1] = width / 2.0
        outputs['BWB_INPUT_STATION_DIST'][2] = 1.0

        outputs['BWB_CHORD_PER_SEMISPAN_DIST'][0] = length
        outputs['BWB_CHORD_PER_SEMISPAN_DIST'][1] = xl_out
        outputs['BWB_CHORD_PER_SEMISPAN_DIST'][2] = wing_tip_chord

        outputs['BWB_THICKNESS_TO_CHORD_DIST'][0] = tc
        outputs['BWB_THICKNESS_TO_CHORD_DIST'][1] = tc
        outputs['BWB_THICKNESS_TO_CHORD_DIST'][2] = tc

        outputs['BWB_LOAD_PATH_SWEEP_DIST'][0] = 0.0
        outputs['BWB_LOAD_PATH_SWEEP_DIST'][1] = swp_ld_path
        outputs['BWB_LOAD_PATH_SWEEP_DIST'][2] = swp_ld_path

    def compute_partials(self, inputs, J):
        width = inputs[Aircraft.Fuselage.MAX_WIDTH][0]
        wingspan = inputs[Aircraft.Wing.SPAN][0]
        length = inputs[Aircraft.Fuselage.LENGTH][0]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD][0]
        rear_spar_percent_chord = inputs['Rear_spar_percent_chord'][0]
        xl_out = root_chord / rear_spar_percent_chord
        wing_tip_chord = 0.06 * wingspan
        tc = inputs[Aircraft.Wing.THICKNESS_TO_CHORD][0]
        sweep = inputs[Aircraft.Wing.SWEEP][0]
        tr_out = wing_tip_chord / xl_out
        ar_out = 2.0 * (wingspan - width) / (wing_tip_chord + xl_out)
        angle = np.tan(sweep / 57.2958) - 2.0 * (1 - tr_out) / (1 + tr_out) / ar_out
        swp_ld_path = 57.2958 * np.arctan(angle)

        J['BWB_INPUT_STATION_DIST', Aircraft.Fuselage.MAX_WIDTH] = [0.0, 0.5, 0.0]

        J['BWB_CHORD_PER_SEMISPAN_DIST', Aircraft.Fuselage.LENGTH] = [1.0, 0.0, 0.0]
        J['BWB_CHORD_PER_SEMISPAN_DIST', Aircraft.Wing.ROOT_CHORD] = [
            0,
            1 / rear_spar_percent_chord,
            0,
        ]
        J['BWB_CHORD_PER_SEMISPAN_DIST', Aircraft.Wing.SPAN] = [0.0, 0.0, 0.06]
        J['BWB_CHORD_PER_SEMISPAN_DIST', 'Rear_spar_percent_chord'] = [
            0,
            -root_chord / rear_spar_percent_chord**2,
            0.0,
        ]

        J['BWB_THICKNESS_TO_CHORD_DIST', Aircraft.Wing.THICKNESS_TO_CHORD] = 1

        dswp_ld_path_dsweep = 1 / (1 + angle**2) / np.cos(sweep / 57.2958) ** 2
        J['BWB_LOAD_PATH_SWEEP_DIST', Aircraft.Wing.SWEEP] = [
            0.0,
            dswp_ld_path_dsweep,
            dswp_ld_path_dsweep,
        ]

        dtr_out_dspan = 0.06 * rear_spar_percent_chord / root_chord
        dar_out_dspan = (
            2
            * (wing_tip_chord + xl_out - 0.06 * (wingspan - width))
            / (wing_tip_chord + xl_out) ** 2
        )
        dswp_ld_path_dspan = (
            57.2958
            / (1 + angle**2)
            * (
                4 * dtr_out_dspan / (1 + tr_out) ** 2 / ar_out
                + 2 * (2 / (1 + tr_out) - 1) * dar_out_dspan / ar_out**2
            )
        )
        J['BWB_LOAD_PATH_SWEEP_DIST', Aircraft.Wing.SPAN] = [
            0.0,
            dswp_ld_path_dspan,
            dswp_ld_path_dspan,
        ]

        dtr_out_droot_chord = -wing_tip_chord * rear_spar_percent_chord / root_chord**2
        dar_out_droot_chord = (
            -2 * (wingspan - width) / (wing_tip_chord + xl_out) ** 2 / rear_spar_percent_chord
        )
        dswp_ld_path_droot_chord = (
            57.2958
            / (1 + angle**2)
            * (
                4 * dtr_out_droot_chord / (1 + tr_out) ** 2 / ar_out
                + 2 * (2 / (1 + tr_out) - 1) * dar_out_droot_chord / ar_out**2
            )
        )
        J['BWB_LOAD_PATH_SWEEP_DIST', Aircraft.Wing.ROOT_CHORD] = [
            0.0,
            dswp_ld_path_droot_chord,
            dswp_ld_path_droot_chord,
        ]

        dtr_out_dwidth = 0.0
        dar_out_dwidth = -2 / (wing_tip_chord + xl_out)
        dswp_ld_path_dwidth = (
            57.2958
            / (1 + angle**2)
            * (
                4 * dtr_out_dwidth / (1 + tr_out) ** 2 / ar_out
                + 2 * (2 / (1 + tr_out) - 1) * dar_out_dwidth / ar_out**2
            )
        )
        J['BWB_LOAD_PATH_SWEEP_DIST', Aircraft.Fuselage.MAX_WIDTH] = [
            0.0,
            dswp_ld_path_dwidth,
            dswp_ld_path_dwidth,
        ]

        dtr_out_drear_chord = 0.06 * wingspan / root_chord
        dar_out_drear_chord = (
            2
            * (wingspan - width)
            / (wing_tip_chord + xl_out) ** 2
            * root_chord
            / rear_spar_percent_chord**2
        )
        dswp_ld_path_drear_chord = (
            57.2958
            / (1 + angle**2)
            * (
                4 * dtr_out_drear_chord / (1 + tr_out) ** 2 / ar_out
                + 2 * (2 / (1 + tr_out) - 1) * dar_out_drear_chord / ar_out**2
            )
        )
        J['BWB_LOAD_PATH_SWEEP_DIST', 'Rear_spar_percent_chord'] = [
            0.0,
            dswp_ld_path_drear_chord,
            dswp_ld_path_drear_chord,
        ]


class BWBWingPrelim(om.ExplicitComponent):
    """preliminary calculations of wing aspect ratio for BWB using detailed wing information"""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.NUM_INTEGRATION_STATIONS)

    def setup(self):
        num_stations = self.options[Aircraft.Wing.NUM_INTEGRATION_STATIONS]

        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.GLOVE_AND_BAT, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        self.add_input('BWB_INPUT_STATION_DIST', shape=num_stations, units='unitless')
        self.add_input('BWB_CHORD_PER_SEMISPAN_DIST', shape=num_stations, units='unitless')

        add_aviary_output(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_output(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_output(self, Aircraft.Wing.LOAD_FRACTION, units='unitless')

        self.declare_partials('*', '*', method='fd', form='forward')

    def compute(self, inputs, outputs):
        input_station_dist = inputs['BWB_INPUT_STATION_DIST']
        num_stations = len(inputs['BWB_INPUT_STATION_DIST'])

        glove_and_bat = inputs[Aircraft.Wing.GLOVE_AND_BAT]
        width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        span = inputs[Aircraft.Wing.SPAN][0]

        ssm = 0.0
        bwb_chord_per_semispan_dist = inputs['BWB_CHORD_PER_SEMISPAN_DIST']

        # Calculate Wing Area and Aspect Ratio for modified planform
        if bwb_chord_per_semispan_dist[0] <= 5.0:
            C1 = bwb_chord_per_semispan_dist[0] * span / 2.0
        else:
            C1 = bwb_chord_per_semispan_dist[0]
        if input_station_dist[0] <= 1.1:
            Y1 = input_station_dist[0] * span / 2.0
        else:
            Y1 = input_station_dist[0]
        for n in range(1, num_stations):
            if bwb_chord_per_semispan_dist[n] <= 5.0:
                C2 = bwb_chord_per_semispan_dist[n] * span / 2.0
            else:
                C2 = bwb_chord_per_semispan_dist[n]
            if input_station_dist[n] <= 1.1:
                Y2 = input_station_dist[n] * span / 2.0
            else:
                Y2 = input_station_dist[n]
            axp = (Y2 - Y1) * (C1 + C2)
            C1 = C2
            Y1 = Y2
            ssm = ssm + axp
        ar = span**2 / (ssm - glove_and_bat)
        # Calculated wing area for aerodynamics
        outputs[Aircraft.Wing.AREA] = ssm
        outputs[Aircraft.Wing.ASPECT_RATIO] = ar

        # Estimate the percent load carried by the outboard wing
        pct_load = (1.0 - width / span) ** 2
        outputs[Aircraft.Wing.LOAD_FRACTION] = pct_load
