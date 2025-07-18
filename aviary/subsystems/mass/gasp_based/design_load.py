import numpy as np
import openmdao.api as om

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.utils.functions import sigmoidX, dSigmoidXdx
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission, Settings


def dquotient(u, v, du, dv):
    """d(u/v) / dv."""
    return (du * v - u * dv) / v**2


class LoadSpeeds(om.ExplicitComponent):
    """
    Computation of load speeds (such as maximum operating equivalent airspeed,
    velocity used in Gust Load Factor calculation at cruise conditions, maximum
    maneuver load factor, and minimum dive velocity).
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.PART25_STRUCTURAL_CATEGORY)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Wing.LOADING_ABOVE_20)

    def setup(self):
        add_aviary_input(self, Aircraft.Design.MAX_STRUCTURAL_SPEED, units='mi/h')

        if self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY] < 3:
            add_aviary_input(self, Aircraft.Wing.LOADING, units='lbf/ft**2')

        self.add_output(
            'max_airspeed',
            units='kn',
            desc='VM0: maximum operating equivalent airspeed',
        )
        self.add_output(
            'vel_c',
            units='kn',
            desc='VGC: Velocity used in Gust Load Factor calculation at cruise conditions.\
                        This is Minimum Design Cruise Speed for Part 23 aircraft and VM0 for Part 25 aircraft',
        )
        self.add_output(
            'max_maneuver_factor',
            units='unitless',
            desc='EMLF: maximum maneuver load factor, units are in g`s',
        )
        self.add_output('min_dive_vel', units='kn', desc='VDMIN: dive velocity')
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        max_struct_speed_mph = inputs[Aircraft.Design.MAX_STRUCTURAL_SPEED]

        CATD = self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        WGS_greater_than_20_flag = self.options[Aircraft.Wing.LOADING_ABOVE_20]

        max_struct_speed_kts = max_struct_speed_mph / 1.15

        if CATD < 3:
            wing_loading = inputs[Aircraft.Wing.LOADING]

            VCMAX = 0.9 * max_struct_speed_kts
            VCCOF = 33.0
            if WGS_greater_than_20_flag:
                VCCOF = 33.0 - (0.0550 * (wing_loading - 20.0))

            if CATD == 2:
                VCCOF = 36.0
                if WGS_greater_than_20_flag:
                    VCCOF = 36.0 - (0.0925 * (wing_loading - 20.0))

            VCMIN = VCCOF * (wing_loading**0.5)

            if smooth:
                VCMIN = VCMIN * sigmoidX(VCMIN / VCMAX, 1, -0.01) + VCMAX * sigmoidX(
                    VCMIN / VCMAX, 1, 0.01
                )
            else:
                if VCMIN > VCMAX:
                    VCMIN = VCMAX

            VDCOF = 1.4
            if WGS_greater_than_20_flag:
                VDCOF = 1.4 - (0.000625 * (wing_loading - 20.0))

            if CATD != 0:
                VDCOF = 1.5
                if WGS_greater_than_20_flag:
                    VDCOF = 1.5 - (0.001875 * (wing_loading - 20.0))

                if CATD != 1:
                    VDCOF = 1.55
                    if WGS_greater_than_20_flag:
                        VDCOF = 1.55 - (0.0025 * (wing_loading - 20.0))

            min_dive_vel = VDCOF * VCMIN

            if smooth:
                min_dive_vel = max_struct_speed_kts * sigmoidX(
                    min_dive_vel / max_struct_speed_kts, 1, -0.01
                ) + min_dive_vel * sigmoidX(min_dive_vel / max_struct_speed_kts, 1, 0.01)
            else:
                if min_dive_vel < max_struct_speed_kts:
                    min_dive_vel = max_struct_speed_kts

            max_airspeed = 0.85 * min_dive_vel
            vel_c = VCMIN
            if CATD == 0:
                max_maneuver_factor = 3.8
            if CATD == 1:
                max_maneuver_factor = 4.4
            elif CATD == 2:
                max_maneuver_factor = 6.0

        elif CATD == 3:
            max_maneuver_factor = 2.5
            min_dive_vel = 1.2 * max_struct_speed_kts
            max_airspeed = max_struct_speed_kts
            vel_c = max_airspeed

        elif CATD > 3.001:
            max_maneuver_factor = CATD
            min_dive_vel = 1.2 * max_struct_speed_kts
            max_airspeed = max_struct_speed_kts
            vel_c = max_airspeed

        outputs['max_airspeed'] = max_airspeed
        outputs['vel_c'] = vel_c
        outputs['max_maneuver_factor'] = max_maneuver_factor
        outputs['min_dive_vel'] = min_dive_vel

    def compute_partials(self, inputs, partials):
        max_struct_speed_mph = inputs[Aircraft.Design.MAX_STRUCTURAL_SPEED]

        CATD = self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        WGS_greater_than_20_flag = self.options[Aircraft.Wing.LOADING_ABOVE_20]

        max_struct_speed_kts = max_struct_speed_mph / 1.15
        dmax_struct_speed_kts_dmax_struct_speed_mph = 1 / 1.15
        dmax_struct_speed_kts_dwing_loading = 0.0

        if CATD < 3:
            wing_loading = inputs[Aircraft.Wing.LOADING]
            VCMAX = 0.9 * max_struct_speed_kts
            VCCOF = 33.0
            dVCCOF_dwing_loading = 0.0
            dVCMAX_dmax_struct_speed_mph = 0.9 / 1.15

            if WGS_greater_than_20_flag:
                VCCOF = 33.0 - (0.0550 * (wing_loading - 20.0))
                dVCCOF_dwing_loading = -0.0550

            if CATD == 2:
                VCCOF = 36
                dVCCOF_dwing_loading = 0.0
                if WGS_greater_than_20_flag:
                    VCCOF = 36.0 - (0.0925 * (wing_loading - 20.0))
                    dVCCOF_dwing_loading = -0.0925

            VCMIN = VCCOF * (wing_loading**0.5)
            dVCMIN_dwing_loading = (
                dVCCOF_dwing_loading * (wing_loading**0.5) + VCCOF * 0.5 * wing_loading**-0.5
            )
            dVCMIN_dmax_struct_speed_mph = 0.0

            if smooth:
                SigA = sigmoidX(VCMIN / VCMAX, 1, -0.01)
                SigB = sigmoidX(VCMIN / VCMAX, 1, 0.01)
                DSigB = dSigmoidXdx(VCMIN / VCMAX, 1, 0.01)
                VCMIN_1 = VCMIN * SigA + VCMAX * SigB
                dVCMIN_dwing_loading = (
                    dVCMIN_dwing_loading * SigA
                    + VCMIN * DSigB * -dVCMIN_dwing_loading / VCMAX
                    + VCMAX * DSigB * dVCMIN_dwing_loading / VCMAX
                )
                dVCMIN_dmax_struct_speed_mph = (
                    dVCMIN_dmax_struct_speed_mph * SigA
                    + VCMIN
                    * DSigB
                    * dquotient(
                        (VCMAX - VCMIN),
                        VCMAX,
                        dVCMAX_dmax_struct_speed_mph - dVCMIN_dmax_struct_speed_mph,
                        dVCMAX_dmax_struct_speed_mph,
                    )
                    + dVCMAX_dmax_struct_speed_mph * SigB
                    + VCMAX
                    * DSigB
                    * dquotient(
                        (VCMIN - VCMAX),
                        VCMAX,
                        dVCMIN_dmax_struct_speed_mph - dVCMAX_dmax_struct_speed_mph,
                        dVCMAX_dmax_struct_speed_mph,
                    )
                )
                VCMIN = VCMIN_1
            else:
                if VCMIN > VCMAX:
                    VCMIN = VCMAX
                    dVCMIN_dmax_struct_speed_mph = dVCMAX_dmax_struct_speed_mph
                    dVCMIN_dwing_loading = 0.0

            # why this block?
            if CATD == 1:
                dVCMIN_dwing_loading = 0.0
                dVCMIN_dmax_struct_speed_mph = 0.9 / 1.15

            VDCOF = 1.4
            dVDCOF_dwing_loading = 0.0

            if WGS_greater_than_20_flag:
                VDCOF = 1.4 - (0.000625 * (wing_loading - 20.0))
                dVDCOF_dwing_loading = -0.000625

            if CATD != 0:
                VDCOF = 1.5
                dVDCOF_dwing_loading = 0.0
                if WGS_greater_than_20_flag:
                    VDCOF = 1.5 - (0.001875 * (wing_loading - 20.0))
                    dVDCOF_dwing_loading = -0.001875

                if CATD != 1:
                    VDCOF = 1.55
                    dVDCOF_dwing_loading = 0.0
                    if WGS_greater_than_20_flag:
                        VDCOF = 1.55 - (0.0025 * (wing_loading - 20.0))
                        dVDCOF_dwing_loading = -0.0025

            min_dive_vel = VDCOF * VCMIN
            dmin_dive_vel_dwing_loading = (
                dVDCOF_dwing_loading * VCMIN + VDCOF * dVCMIN_dwing_loading
            )
            dmin_dive_vel_dmax_struct_speed_mph = VDCOF * dVCMIN_dmax_struct_speed_mph

            if smooth:
                SigC = sigmoidX(min_dive_vel / max_struct_speed_kts, 1, -0.01)
                SigD = sigmoidX(min_dive_vel / max_struct_speed_kts, 1, 0.01)
                DSigD = dSigmoidXdx(min_dive_vel / max_struct_speed_kts, 1, 0.01)
                min_dive_vel_1 = max_struct_speed_kts * SigC + min_dive_vel * SigD
                dmin_dive_vel_dmax_struct_speed_mph = (
                    dmax_struct_speed_kts_dmax_struct_speed_mph * SigC
                    + max_struct_speed_kts
                    * DSigD
                    * dquotient(
                        (max_struct_speed_kts - min_dive_vel),
                        max_struct_speed_kts,
                        dmax_struct_speed_kts_dmax_struct_speed_mph
                        - dmin_dive_vel_dmax_struct_speed_mph,
                        dmax_struct_speed_kts_dmax_struct_speed_mph,
                    )
                    + dmin_dive_vel_dmax_struct_speed_mph * SigD
                    + min_dive_vel
                    * DSigD
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dmax_struct_speed_mph
                        - dmax_struct_speed_kts_dmax_struct_speed_mph,
                        dmax_struct_speed_kts_dmax_struct_speed_mph,
                    )
                )
                dmin_dive_vel_dwing_loading = (
                    dmax_struct_speed_kts_dwing_loading * SigC
                    + max_struct_speed_kts
                    * DSigD
                    * dquotient(
                        max_struct_speed_kts - min_dive_vel,
                        max_struct_speed_kts,
                        dmax_struct_speed_kts_dwing_loading - dmin_dive_vel_dwing_loading,
                        dmax_struct_speed_kts_dwing_loading,
                    )
                    + dmin_dive_vel_dwing_loading * SigD
                    + min_dive_vel
                    * DSigD
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dwing_loading - dmax_struct_speed_kts_dwing_loading,
                        dmax_struct_speed_kts_dwing_loading,
                    )
                )
                min_dive_vel = min_dive_vel_1

            else:
                if min_dive_vel < max_struct_speed_kts:  # note: this creates a discontinuity
                    min_dive_vel = max_struct_speed_kts
                    dmin_dive_vel_dwing_loading = 0
                    dmin_dive_vel_dmax_struct_speed_mph = (
                        dmax_struct_speed_kts_dmax_struct_speed_mph
                    )

            partials['min_dive_vel', Aircraft.Wing.LOADING] = dmin_dive_vel_dwing_loading
            partials['min_dive_vel', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                dmin_dive_vel_dmax_struct_speed_mph
            )

            partials['max_airspeed', Aircraft.Wing.LOADING] = 0.85 * dmin_dive_vel_dwing_loading
            partials['max_airspeed', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                0.85 * dmin_dive_vel_dmax_struct_speed_mph
            )

            partials['vel_c', Aircraft.Wing.LOADING] = dVCMIN_dwing_loading
            partials['vel_c', Aircraft.Design.MAX_STRUCTURAL_SPEED] = dVCMIN_dmax_struct_speed_mph

        if CATD == 3:
            partials['max_airspeed', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['min_dive_vel', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.2 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['vel_c', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )

        elif CATD > 3.001:
            partials['max_airspeed', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['min_dive_vel', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.2 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['vel_c', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )


class LoadParameters(om.ExplicitComponent):
    """
    Computation of load parameters (such as maximum operating Mach number,
    density ratio, etc.).
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.PART25_STRUCTURAL_CATEGORY)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Mission.Design.CRUISE_ALTITUDE, units='ft')

    def setup(self):
        self.add_input(
            'vel_c',
            val=100,
            units='kn',
            desc='VGC: Velocity used in Gust Load Factor calculation at cruise conditions.\
                       This is Minimum Design Cruise Speed for Part 23 aircraft and VM0 for Part 25 aircraft',
        )
        self.add_input(
            'max_airspeed',
            val=200,
            units='kn',
            desc='VM0: maximum operating equivalent airspeed',
        )

        self.add_output(
            'max_mach',
            val=0,
            units='unitless',
            desc='EMM0: maximum operating Mach number',
        )
        self.add_output(
            'density_ratio',
            val=0,
            units='unitless',
            desc='SIGMA (in GASP): density ratio = density at Altitude / density at Sea level',
        )
        self.add_output(
            'V9',
            val=0,
            units='kn',
            desc='V9: intermediate value. Typically it is maximum flight speed.',
        )

        self.declare_partials('max_mach', 'max_airspeed')
        self.declare_partials('density_ratio', 'max_airspeed')
        self.declare_partials('V9', '*')

    def compute(self, inputs, outputs):
        vel_c = inputs['vel_c']
        max_airspeed = inputs['max_airspeed']

        cruise_alt, _ = self.options[Mission.Design.CRUISE_ALTITUDE]
        CATD = self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        if cruise_alt <= 22500.0:
            max_mach = max_airspeed / 486.33
        if cruise_alt > 22500.0 and cruise_alt <= 36000.0:
            max_mach = max_airspeed / 424.73
        if cruise_alt > 36000.0:
            max_mach = max_airspeed / 372.34

        if smooth:
            max_mach = max_mach * sigmoidX(max_mach / 0.9, 1, -0.01) + 0.9 * sigmoidX(
                max_mach / 0.9, 1, 0.01
            )

        else:
            if max_mach > 0.90:  # note: this creates a discontinuity
                max_mach = 0.90

        density_ratio = (max_airspeed / (661.7 * max_mach)) ** 1.61949

        if smooth:
            SigE = sigmoidX(density_ratio, 1, -0.01)
            SigF = sigmoidX(density_ratio, 1, 0.01)
            V9 = vel_c * SigE + 661.7 * max_mach * SigF

            if CATD < 3:
                # this line creates a smooth bounded density_ratio such that .6820<=density_ratio<=1
                density_ratio = (
                    0.6820 * sigmoidX(density_ratio / 0.6820, 1, -0.01)
                    + density_ratio * sigmoidX(density_ratio / 0.6820, 1, 0.01) * SigE
                    + SigF
                )

            else:
                # this line creates a smooth bounded density_ratio such that .53281<=density_ratio<=1
                density_ratio = (
                    0.53281 * sigmoidX(density_ratio / 0.53281, 1, -0.01)
                    + density_ratio * sigmoidX(density_ratio / 0.53281, 1, 0.01) * SigE
                    + SigF
                )

        else:
            if density_ratio >= 0.53281:  # note: this creates a discontinuity
                V9 = vel_c
                if density_ratio > 1:  # note: this creates a discontinuity
                    V9 = 661.7 * max_mach
                    density_ratio = 1.0

            else:  # note: this creates a discontinuity
                density_ratio = 0.53281
                V9 = vel_c

            if CATD < 3.0 and density_ratio <= 0.6820:  # note: this creates a discontinuity
                density_ratio = 0.6820

        outputs['max_mach'] = max_mach
        outputs['density_ratio'] = density_ratio
        outputs['V9'] = V9

    def compute_partials(self, inputs, partials):
        vel_c = inputs['vel_c']
        max_airspeed = inputs['max_airspeed']

        cruise_alt, _ = self.options[Mission.Design.CRUISE_ALTITUDE]
        CATD = self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        if cruise_alt <= 22500.0:
            max_mach = max_airspeed / 486.33
            dmax_mach_dmax_airspeed = 1 / 486.33
        if cruise_alt > 22500.0 and cruise_alt <= 36000.0:
            max_mach = max_airspeed / 424.73
            dmax_mach_dmax_airspeed = 1 / 424.73
        if cruise_alt > 36000.0:
            max_mach = max_airspeed / 372.34
            dmax_mach_dmax_airspeed = 1 / 372.34

        if smooth:
            max_mach_1 = max_mach * sigmoidX(max_mach / 0.9, 1, -0.01) + 0.9 * sigmoidX(
                max_mach / 0.9, 1, 0.01
            )
            dmax_mach_dmax_airspeed = (
                dmax_mach_dmax_airspeed * sigmoidX(max_mach / 0.9, 1, -0.01)
                + max_mach
                * dSigmoidXdx(max_mach / 0.9, 1, 0.01)
                * dquotient((0.9 - max_mach), 0.9, -dmax_mach_dmax_airspeed, 0.0)
                + 0.9
                * dSigmoidXdx(max_mach / 0.9, 1, 0.01)
                * dquotient((max_mach - 0.9), 0.9, dmax_mach_dmax_airspeed, 0.0)
            )
            max_mach = max_mach_1
        else:
            if max_mach > 0.90:  # note: this creates a discontinuity
                max_mach = 0.90
                dmax_mach_dmax_airspeed = 0.0

        density_ratio = (max_airspeed / (661.7 * max_mach)) ** 1.61949
        ddensity_ratio_dmax_airspeed = (
            1.61949
            * (max_airspeed / (661.7 * max_mach)) ** 0.61949
            * dquotient(max_airspeed, (661.7 * max_mach), 1, 661.7 * dmax_mach_dmax_airspeed)
        )

        if smooth:
            SigE = sigmoidX(density_ratio, 1, -0.01)
            SigF = sigmoidX(density_ratio, 1, 0.01)
            DSigF = dSigmoidXdx(density_ratio, 1, 0.01)
            dV9_dmax_airspeed = (
                vel_c * DSigF * (-ddensity_ratio_dmax_airspeed)
                + 661.7
                * dmax_mach_dmax_airspeed
                * SigF
                * 661.7
                * max_mach
                * DSigF
                * ddensity_ratio_dmax_airspeed
            )
            dV9_dvel_c = SigE

            if CATD < 3:
                # this line creates a smooth bounded density_ratio such that .6820<=density_ratio<=1
                density_ratio_1 = (
                    0.6820 * sigmoidX(density_ratio / 0.6820, 1, -0.01)
                    + density_ratio * sigmoidX(density_ratio / 0.6820, 1, 0.01) * SigE
                    + SigF
                )
                ddensity_ratio_dmax_airspeed = (
                    0.6820
                    * dSigmoidXdx(density_ratio / 0.6820, 1, 0.01)
                    * -ddensity_ratio_dmax_airspeed
                    / 0.6820
                    + ddensity_ratio_dmax_airspeed
                    * sigmoidX(density_ratio / 0.6820, 1, 0.01)
                    * SigF
                    + density_ratio
                    * (
                        dSigmoidXdx(density_ratio / 0.6820, 1, 0.01)
                        * ddensity_ratio_dmax_airspeed
                        / 0.6820
                        * SigE
                        + sigmoidX(density_ratio / 0.6820, 1, 0.01)
                        * DSigF
                        * -ddensity_ratio_dmax_airspeed
                    )
                    + DSigF * ddensity_ratio_dmax_airspeed
                )
                density_ratio = density_ratio_1

            else:
                # this line creates a smooth bounded density_ratio such that .53281<=density_ratio<=1
                density_ratio_1 = (
                    0.53281 * sigmoidX(density_ratio / 0.53281, 1, -0.01)
                    + density_ratio * sigmoidX(density_ratio / 0.53281, 1, 0.01) * SigE
                    + SigF
                )
                ddensity_ratio_dmax_airspeed = (
                    0.53281
                    * dSigmoidXdx(density_ratio / 0.53281, 1, 0.01)
                    * -ddensity_ratio_dmax_airspeed
                    / 0.53281
                    + ddensity_ratio_dmax_airspeed
                    * sigmoidX(density_ratio / 0.53281, 1, 0.01)
                    * SigE
                    + density_ratio
                    * (
                        dSigmoidXdx(density_ratio / 0.53281, 1, 0.01)
                        * ddensity_ratio_dmax_airspeed
                        / 0.53281
                        * SigE
                        + sigmoidX(density_ratio / 0.53281, 1, -0.01)
                        * SigF
                        * -ddensity_ratio_dmax_airspeed
                    )
                    + DSigF * ddensity_ratio_dmax_airspeed
                )
                density_ratio = density_ratio_1
        else:
            if density_ratio >= 0.53281:  # note: this creates a discontinuity
                dV9_dvel_c = 1.0
                dV9_dmax_airspeed = 0.0

                if density_ratio > 1:  # note: this creates a discontinuity
                    density_ratio = 1.0

                    dV9_dvel_c = 0.0
                    dV9_dmax_airspeed = 661.7 * dmax_mach_dmax_airspeed
                    ddensity_ratio_dmax_airspeed = 0.0

            else:  # note: this creates a discontinuity
                density_ratio = 0.53281

                dV9_dvel_c = 1.0
                dV9_dmax_airspeed = 0.0
                ddensity_ratio_dmax_airspeed = 0.0

            if CATD < 3.0 and density_ratio <= 0.6820:  # note: this creates a discontinuity
                density_ratio = 0.6820
                ddensity_ratio_dmax_airspeed = 0.0

        partials['max_mach', 'max_airspeed'] = dmax_mach_dmax_airspeed
        partials['density_ratio', 'max_airspeed'] = ddensity_ratio_dmax_airspeed
        partials['V9', 'max_airspeed'] = dV9_dmax_airspeed
        partials['V9', 'vel_c'] = dV9_dvel_c


class LiftCurveSlopeAtCruise(om.ExplicitComponent):
    """Computation of lift curve slope at cruise Mach number."""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='rad')
        add_aviary_input(self, Mission.Design.MACH, units='unitless')

        add_aviary_output(self, Aircraft.Design.LIFT_CURVE_SLOPE, units='1/rad')

        self.declare_partials(Aircraft.Design.LIFT_CURVE_SLOPE, '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        DLMC4 = inputs[Aircraft.Wing.SWEEP]
        mach = inputs[Mission.Design.MACH]

        if verbosity > Verbosity.BRIEF:
            if AR <= 0.0:
                print('Aircraft.Wing.ASPECT_RATIO must be positive.')
            if DLMC4 == np.pi / 2.0:
                print('Aircraft.Wing.SWEEP can not be 90 degrees.')

        outputs[Aircraft.Design.LIFT_CURVE_SLOPE] = (
            np.pi
            * AR
            / (
                1
                + np.sqrt(
                    1 + (((AR / (2 * np.cos(DLMC4))) ** 2) * (1 - (mach * np.cos(DLMC4)) ** 2))
                )
            )
        )

    def compute_partials(self, inputs, partials):
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        DLMC4 = inputs[Aircraft.Wing.SWEEP]
        mach = inputs[Mission.Design.MACH]

        c1 = np.sqrt(AR**2 * (-(mach**2) * np.cos(DLMC4) ** 2 + 1) + 4 * np.cos(DLMC4) ** 2)
        c2 = 2 * np.cos(DLMC4) + c1

        partials[Aircraft.Design.LIFT_CURVE_SLOPE, Aircraft.Wing.ASPECT_RATIO] = (
            4 * np.pi * np.cos(DLMC4) ** 2
        ) / (c1 * c2)
        partials[Aircraft.Design.LIFT_CURVE_SLOPE, Mission.Design.MACH] = (
            2 * np.pi * AR**3 * mach * np.cos(DLMC4) ** 3
        ) / (c1 * c2**2)
        partials[Aircraft.Design.LIFT_CURVE_SLOPE, Aircraft.Wing.SWEEP] = (
            -np.pi
            * AR
            * (
                -8 * np.cos(DLMC4) ** 3 * np.sin(DLMC4)
                + 4 * np.cos(DLMC4) ** 2 * np.sin(2 * DLMC4)
                + AR**2 * np.sin(2 * DLMC4)
            )
        ) / (np.cos(DLMC4) * c1 * c2**2)


class LoadFactors(om.ExplicitComponent):
    """Computation of structural ultimate load factor."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.LOADING, units='lbf/ft**2')

        self.add_input(
            'density_ratio',
            val=0.5,
            units='unitless',
            desc='SIGMA (in GASP): density ratio = density at Altitude / density at Sea level',
        )
        self.add_input(
            'V9',
            val=100,
            units='kn',
            desc='V9: intermediate value. Typically it is maximum flight speed.',
        )
        self.add_input('min_dive_vel', val=250, units='kn', desc='VDMIN: dive velocity')
        self.add_input(
            'max_maneuver_factor',
            val=0.72,
            units='unitless',
            desc='EMLF: maximum maneuver load factor, units are in g`s',
        )

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')
        add_aviary_input(self, Aircraft.Design.LIFT_CURVE_SLOPE, units='1/rad')

        add_aviary_output(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')

        self.declare_partials(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, '*')

    def compute(self, inputs, outputs):
        wing_loading = inputs[Aircraft.Wing.LOADING]
        density_ratio = inputs['density_ratio']
        V9 = inputs['V9']
        min_dive_vel = inputs['min_dive_vel']
        max_maneuver_factor = inputs['max_maneuver_factor']
        avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
        Cl_alpha = inputs[Aircraft.Design.LIFT_CURVE_SLOPE]

        ULF_from_maneuver = self.options[Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        mass_ratio = (
            2.0
            * wing_loading
            / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
        )
        k_load_factor = 0.88 * mass_ratio / (5.3 + mass_ratio)
        cruise_load_factor = 1.0 + ((k_load_factor * 50.0 * V9 * Cl_alpha) / (498.0 * wing_loading))
        dive_load_factor = 1.0 + (
            (k_load_factor * 25.0 * min_dive_vel * Cl_alpha) / (498.0 * wing_loading)
        )
        gust_load_factor = dive_load_factor

        if smooth:
            gust_load_factor = dive_load_factor * sigmoidX(
                cruise_load_factor / dive_load_factor, 1, -0.01
            ) + cruise_load_factor * sigmoidX(cruise_load_factor / dive_load_factor, 1, 0.01)

        else:
            if cruise_load_factor > dive_load_factor:  # note: this creates a discontinuity
                gust_load_factor = cruise_load_factor

        ULF = 1.5 * max_maneuver_factor

        if smooth:
            ULF = 1.5 * (
                gust_load_factor * sigmoidX(max_maneuver_factor / gust_load_factor, 1, -0.01)
                + max_maneuver_factor * sigmoidX(max_maneuver_factor / gust_load_factor, 1, 0.01)
            )

        else:
            if gust_load_factor > max_maneuver_factor:  # note: this creates a discontinuity
                ULF = 1.5 * gust_load_factor

        if ULF_from_maneuver is True:
            ULF = 1.5 * max_maneuver_factor

        outputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = ULF

    def compute_partials(self, inputs, partials):
        wing_loading = inputs[Aircraft.Wing.LOADING]
        density_ratio = inputs['density_ratio']
        V9 = inputs['V9']
        min_dive_vel = inputs['min_dive_vel']
        max_maneuver_factor = inputs['max_maneuver_factor']
        avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
        Cl_alpha = inputs[Aircraft.Design.LIFT_CURVE_SLOPE]

        ULF_from_maneuver = self.options[Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

        mass_ratio = (
            2.0
            * wing_loading
            / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
        )
        k_load_factor = 0.88 * mass_ratio / (5.3 + mass_ratio)
        cruise_load_factor = 1.0 + ((k_load_factor * 50.0 * V9 * Cl_alpha) / (498.0 * wing_loading))
        dive_load_factor = 1.0 + (
            (k_load_factor * 25.0 * min_dive_vel * Cl_alpha) / (498.0 * wing_loading)
        )
        gust_load_factor = dive_load_factor

        dmass_ratio_dwing_loading = 2.0 / (
            density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2
        )
        dmass_ratio_ddensity_ratio = (
            -2.0
            * wing_loading
            / (density_ratio**2 * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
        )
        dmass_ratio_davg_chord = (
            -2.0
            * wing_loading
            / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord**2 * Cl_alpha * 32.2)
        )
        dmass_ratio_dCl_alpha = (
            -2.0
            * wing_loading
            / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha**2 * 32.2)
        )

        dk_load_factor_dwing_loading = dquotient(
            0.88 * mass_ratio,
            5.3 + mass_ratio,
            0.88 * dmass_ratio_dwing_loading,
            dmass_ratio_dwing_loading,
        )
        dk_load_factor_ddensity_ratio = dquotient(
            0.88 * mass_ratio,
            5.3 + mass_ratio,
            0.88 * dmass_ratio_ddensity_ratio,
            dmass_ratio_ddensity_ratio,
        )
        dk_load_factor_davg_chord = dquotient(
            0.88 * mass_ratio,
            5.3 + mass_ratio,
            0.88 * dmass_ratio_davg_chord,
            dmass_ratio_davg_chord,
        )
        dk_load_factor_dCl_alpha = dquotient(
            0.88 * mass_ratio,
            5.3 + mass_ratio,
            0.88 * dmass_ratio_dCl_alpha,
            dmass_ratio_dCl_alpha,
        )

        dcruise_load_factor_dwing_loading = dquotient(
            (k_load_factor * 50.0 * V9 * Cl_alpha),
            (498.0 * wing_loading),
            (dk_load_factor_dwing_loading * 50.0 * V9 * Cl_alpha),
            498,
        )
        dcruise_load_factor_ddensity_ratio = (
            dk_load_factor_ddensity_ratio * 50.0 * V9 * Cl_alpha
        ) / (498.0 * wing_loading)
        dcruise_load_factor_davg_chord = (dk_load_factor_davg_chord * 50.0 * V9 * Cl_alpha) / (
            498.0 * wing_loading
        )
        dcruise_load_factor_dCl_alpha = (
            dk_load_factor_dCl_alpha * 50.0 * V9 * Cl_alpha + k_load_factor * 50 * V9
        ) / (498.0 * wing_loading)
        dcruise_load_factor_dV9 = (k_load_factor * 50.0 * Cl_alpha) / (498.0 * wing_loading)

        ddive_load_factor_dwing_loading = dquotient(
            (k_load_factor * 25.0 * min_dive_vel * Cl_alpha),
            (498.0 * wing_loading),
            (dk_load_factor_dwing_loading * 25.0 * min_dive_vel * Cl_alpha),
            498,
        )
        ddive_load_factor_ddensity_ratio = (
            dk_load_factor_ddensity_ratio * 25 * min_dive_vel * Cl_alpha
        ) / (498.0 * wing_loading)
        ddive_load_factor_davg_chord = (
            dk_load_factor_davg_chord * 25 * min_dive_vel * Cl_alpha
        ) / (498.0 * wing_loading)
        ddive_load_factor_dCl_alpha = (
            dk_load_factor_dCl_alpha * 25 * min_dive_vel * Cl_alpha
            + k_load_factor * 50 * min_dive_vel
        ) / (498.0 * wing_loading)
        (k_load_factor * 25 * Cl_alpha) / (498.0 * wing_loading)
        ddive_load_factor_dV9 = 0.0

        dgust_load_factor_dwing_loading = dquotient(
            (k_load_factor * 25.0 * min_dive_vel * Cl_alpha),
            (498.0 * wing_loading),
            (dk_load_factor_dwing_loading * 25.0 * min_dive_vel * Cl_alpha),
            498,
        )
        dgust_load_factor_ddensity_ratio = (
            dk_load_factor_ddensity_ratio * 25 * min_dive_vel * Cl_alpha
        ) / (498.0 * wing_loading)
        dgust_load_factor_davg_chord = (
            dk_load_factor_davg_chord * 25 * min_dive_vel * Cl_alpha
        ) / (498.0 * wing_loading)
        dgust_load_factor_dCl_alpha = (
            dk_load_factor_dCl_alpha * 25 * min_dive_vel * Cl_alpha
            + k_load_factor * 25 * min_dive_vel
        ) / (498.0 * wing_loading)
        dgust_load_factor_dmin_dive_vel = (k_load_factor * 25 * Cl_alpha) / (498.0 * wing_loading)
        dgust_load_factor_dV9 = 0.0

        if smooth:
            SigG = sigmoidX(cruise_load_factor / dive_load_factor, 1, -0.01)
            SigH = sigmoidX(cruise_load_factor / dive_load_factor, 1, 0.01)
            DSigG = dSigmoidXdx(cruise_load_factor / dive_load_factor, 1, -0.01)
            DSigH = dSigmoidXdx(cruise_load_factor / dive_load_factor, 1, 0.01)
            gust_load_factor_1 = dive_load_factor * SigG + cruise_load_factor * SigH
            dgust_load_factor_dwing_loading = (
                ddive_load_factor_dwing_loading * SigG
                + dive_load_factor
                * DSigG
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dwing_loading - dcruise_load_factor_dwing_loading,
                    ddive_load_factor_dwing_loading,
                )
                + dcruise_load_factor_dwing_loading * SigH
                + cruise_load_factor
                * DSigG
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_dwing_loading - ddive_load_factor_dwing_loading,
                    ddive_load_factor_dwing_loading,
                )
            )
            dgust_load_factor_ddensity_ratio = (
                ddive_load_factor_ddensity_ratio * SigG
                + dive_load_factor
                * DSigG
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_ddensity_ratio - dcruise_load_factor_ddensity_ratio,
                    ddive_load_factor_ddensity_ratio,
                )
                + dcruise_load_factor_ddensity_ratio * SigH
                + cruise_load_factor
                * DSigH
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_ddensity_ratio - ddive_load_factor_ddensity_ratio,
                    ddive_load_factor_ddensity_ratio,
                )
            )
            dgust_load_factor_davg_chord = (
                ddive_load_factor_davg_chord * SigG
                + dive_load_factor
                * DSigH
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_davg_chord - dcruise_load_factor_davg_chord,
                    ddive_load_factor_davg_chord,
                )
                + dcruise_load_factor_davg_chord * SigH
                + cruise_load_factor
                * DSigH
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_davg_chord - ddive_load_factor_davg_chord,
                    ddive_load_factor_davg_chord,
                )
            )
            dgust_load_factor_dCl_alpha = (
                ddive_load_factor_dCl_alpha * SigG
                + dive_load_factor
                * DSigH
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dCl_alpha - dcruise_load_factor_dCl_alpha,
                    ddive_load_factor_dCl_alpha,
                )
                + dcruise_load_factor_dCl_alpha * SigH
                + cruise_load_factor
                * DSigH
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_dCl_alpha - ddive_load_factor_dCl_alpha,
                    ddive_load_factor_dCl_alpha,
                )
            )
            dgust_load_factor_dV9 = (
                dive_load_factor
                * DSigH
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dV9 - dcruise_load_factor_dV9,
                    ddive_load_factor_dV9,
                )
                + dcruise_load_factor_dV9 * SigH
                + cruise_load_factor
                * DSigH
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_dV9 - ddive_load_factor_dV9,
                    ddive_load_factor_dV9,
                )
            )
            gust_load_factor = gust_load_factor_1
        else:
            if cruise_load_factor > dive_load_factor:  # note: this creates a discontinuity
                gust_load_factor = cruise_load_factor

                dgust_load_factor_dwing_loading = dcruise_load_factor_dwing_loading
                dgust_load_factor_ddensity_ratio = dcruise_load_factor_ddensity_ratio
                dgust_load_factor_davg_chord = dcruise_load_factor_davg_chord
                dgust_load_factor_dCl_alpha = dcruise_load_factor_dCl_alpha
                dgust_load_factor_dV9 = dcruise_load_factor_dV9
                dgust_load_factor_dmin_dive_vel = 0.0

        dULF_dmax_maneuver_factor = 1.5
        dULF_dwing_loading = 0.0
        dULF_ddensity_ratio = 0.0
        dULF_davg_chord = 0.0
        dULF_dCl_alpha = 0.0
        dULF_dV9 = 0.0
        dULF_dmin_dive_vel = 0.0

        if smooth:
            SigK = sigmoidX(max_maneuver_factor / gust_load_factor, 1, -0.01)
            SigL = sigmoidX(max_maneuver_factor / gust_load_factor, 1, 0.01)
            DSigL = dSigmoidXdx(max_maneuver_factor / gust_load_factor, 1, 0.01)
            dULF_dmax_maneuver_factor = 1.5 * (
                gust_load_factor
                * DSigL
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    -1.0,
                    0.0,
                )
                + SigL
                + max_maneuver_factor
                * DSigL
                * dquotient((max_maneuver_factor - gust_load_factor), gust_load_factor, 1.0, 0.0)
            )
            dULF_dwing_loading = 1.5 * (
                dgust_load_factor_dwing_loading * SigK
                + gust_load_factor
                * DSigL
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_dwing_loading,
                    dgust_load_factor_dwing_loading,
                )
                + max_maneuver_factor
                * DSigL
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_dwing_loading,
                    dgust_load_factor_dwing_loading,
                )
            )
            dULF_ddensity_ratio = 1.5 * (
                dgust_load_factor_ddensity_ratio * SigK
                + gust_load_factor
                * DSigL
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_ddensity_ratio,
                    dgust_load_factor_ddensity_ratio,
                )
                + max_maneuver_factor
                * DSigL
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_ddensity_ratio,
                    dgust_load_factor_ddensity_ratio,
                )
            )
            dULF_davg_chord = 1.5 * (
                dgust_load_factor_davg_chord * SigK
                + gust_load_factor
                * DSigL
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_davg_chord,
                    dgust_load_factor_davg_chord,
                )
                + max_maneuver_factor
                * DSigL
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_davg_chord,
                    dgust_load_factor_davg_chord,
                )
            )
            dULF_dCl_alpha = 1.5 * (
                dgust_load_factor_dCl_alpha * SigK
                + gust_load_factor
                * DSigL
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_dCl_alpha,
                    dgust_load_factor_dCl_alpha,
                )
                + max_maneuver_factor
                * DSigL
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_dCl_alpha,
                    dgust_load_factor_dCl_alpha,
                )
            )
            dULF_dV9 = 1.5 * (
                dgust_load_factor_dV9 * SigK
                + gust_load_factor
                * DSigL
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_dV9,
                    dgust_load_factor_dV9,
                )
                + max_maneuver_factor
                * DSigL
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_dV9,
                    dgust_load_factor_dV9,
                )
            )
            dULF_dmin_dive_vel = 0.0
        else:
            if gust_load_factor > max_maneuver_factor:  # note: this creates a discontinuity
                dULF_dmax_maneuver_factor = 0.0
                dULF_dwing_loading = 1.5 * dgust_load_factor_dwing_loading
                dULF_ddensity_ratio = 1.5 * dgust_load_factor_ddensity_ratio
                dULF_davg_chord = 1.5 * dgust_load_factor_davg_chord
                dULF_dCl_alpha = 1.5 * dgust_load_factor_dCl_alpha
                dULF_dV9 = 1.5 * dgust_load_factor_dV9
                dULF_dmin_dive_vel = 1.5 * dgust_load_factor_dmin_dive_vel

        if ULF_from_maneuver is True:
            dULF_dmax_maneuver_factor = 1.5
            dULF_dwing_loading = 0.0
            dULF_ddensity_ratio = 0.0
            dULF_davg_chord = 0.0
            dULF_dCl_alpha = 0.0
            dULF_dV9 = 0.0
            dULF_dmin_dive_vel = 0.0

        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'max_maneuver_factor'] = (
            dULF_dmax_maneuver_factor
        )
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Aircraft.Wing.LOADING] = dULF_dwing_loading
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'density_ratio'] = dULF_ddensity_ratio
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Aircraft.Wing.AVERAGE_CHORD] = dULF_davg_chord
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Aircraft.Design.LIFT_CURVE_SLOPE] = (
            dULF_dCl_alpha
        )
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'V9'] = dULF_dV9
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'min_dive_vel'] = dULF_dmin_dive_vel


class DesignLoadGroup(om.Group):
    """
    Design load group for GASP-based tube and wing type aircraft mass.
    """

    def setup(self):
        self.add_subsystem(
            'speeds',
            LoadSpeeds(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[
                'max_airspeed',
                'vel_c',
                'max_maneuver_factor',
                'min_dive_vel',
            ],
        )

        self.add_subsystem(
            'params',
            LoadParameters(),
            promotes_inputs=[
                'max_airspeed',
                'vel_c',
            ],
            promotes_outputs=['density_ratio', 'V9', 'max_mach'],
        )

        self.add_subsystem(
            'Cl_Alpha_calc',
            LiftCurveSlopeAtCruise(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'factors',
            LoadFactors(),
            promotes_inputs=[
                'max_maneuver_factor',
                'min_dive_vel',
                'density_ratio',
                'V9',
            ]
            + ['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )


class BWBLoadSpeeds(om.ExplicitComponent):
    """
    Computation of load speeds (such as maximum operating equivalent airspeed,
    velocity used in Gust Load Factor calculation at cruise conditions, maximum
    maneuver load factor, and minimum dive velocity).
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.PART25_STRUCTURAL_CATEGORY)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Wing.LOADING_ABOVE_20)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Design.MAX_STRUCTURAL_SPEED, units='mi/h')

        if self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY] < 3:
            add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
            add_aviary_input(self, Aircraft.Wing.EXPOSED_AREA, units='ft**2')

        self.add_output(
            'max_airspeed',
            units='kn',
            desc='VM0: maximum operating equivalent airspeed',
        )
        self.add_output(
            'vel_c',
            units='kn',
            desc='VGC: Velocity used in Gust Load Factor calculation at cruise conditions.\
                        This is Minimum Design Cruise Speed for Part 23 aircraft and \
                        VM0 for Part 25 aircraft',
        )
        self.add_output(
            'max_maneuver_factor',
            units='unitless',
            desc='EMLF: maximum maneuver load factor, units are in g`s',
        )
        self.add_output('min_dive_vel', units='kn', desc='VDMIN: dive velocity')
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]

        max_struct_speed_mph = inputs[Aircraft.Design.MAX_STRUCTURAL_SPEED]

        CATD = self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        WGS_greater_than_20_flag = self.options[Aircraft.Wing.LOADING_ABOVE_20]

        max_struct_speed_kts = max_struct_speed_mph / 1.15

        if CATD < 3:
            gross_mass = inputs[Mission.Design.GROSS_MASS]
            exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
            if verbosity > Verbosity.BRIEF:
                if exp_wing_area <= 0.0:
                    print('Aircraft.Wing.EXPOSED_AREA must be positive.')
                if gross_mass <= 0.0:
                    print('Mission.Design.GROSS_MASS must be positive.')
            wing_loading = gross_mass / exp_wing_area

            VCMAX = 0.9 * max_struct_speed_kts
            if CATD <= 1:
                if WGS_greater_than_20_flag:
                    VCCOF = 33.0 - (0.0550 * (wing_loading - 20.0))
                else:
                    VCCOF = 33.0
            elif CATD == 2:
                if WGS_greater_than_20_flag:
                    VCCOF = 36.0 - (0.0925 * (wing_loading - 20.0))
                else:
                    VCCOF = 36.0

            VCMIN = VCCOF * (wing_loading**0.5)

            if smooth:
                VCMIN = VCMIN * sigmoidX(VCMIN / VCMAX, 1, -0.01) + VCMAX * sigmoidX(
                    VCMIN / VCMAX, 1, 0.01
                )
            else:
                if VCMIN > VCMAX:
                    VCMIN = VCMAX

            if CATD == 0:
                if WGS_greater_than_20_flag:
                    VDCOF = 1.4 - (0.000625 * (wing_loading - 20.0))
                else:
                    VDCOF = 1.4
            elif CATD == 1:
                if WGS_greater_than_20_flag:
                    VDCOF = 1.5 - (0.001875 * (wing_loading - 20.0))
                else:
                    VDCOF = 1.5
            elif CATD == 2:
                if WGS_greater_than_20_flag:
                    VDCOF = 1.55 - (0.0025 * (wing_loading - 20.0))
                else:
                    VDCOF = 1.55

            min_dive_vel = VDCOF * VCMIN

            if smooth:
                min_dive_vel = max_struct_speed_kts * sigmoidX(
                    min_dive_vel / max_struct_speed_kts, 1, -0.01
                ) + min_dive_vel * sigmoidX(min_dive_vel / max_struct_speed_kts, 1, 0.01)
            else:
                if min_dive_vel < max_struct_speed_kts:
                    min_dive_vel = max_struct_speed_kts

            max_airspeed = 0.85 * min_dive_vel
            vel_c = VCMIN

            if CATD == 0:
                max_maneuver_factor = 3.8
            elif CATD == 1:
                max_maneuver_factor = 4.4
            elif CATD == 2:
                max_maneuver_factor = 6.0

        elif CATD == 3:
            max_maneuver_factor = 2.5
            min_dive_vel = 1.2 * max_struct_speed_kts
            max_airspeed = max_struct_speed_kts
            vel_c = max_airspeed

        elif CATD > 3.001:
            max_maneuver_factor = CATD
            min_dive_vel = 1.2 * max_struct_speed_kts
            max_airspeed = max_struct_speed_kts
            vel_c = max_airspeed

        outputs['max_airspeed'] = max_airspeed
        outputs['vel_c'] = vel_c
        outputs['max_maneuver_factor'] = max_maneuver_factor
        outputs['min_dive_vel'] = min_dive_vel

    def compute_partials(self, inputs, partials):
        max_struct_speed_mph = inputs[Aircraft.Design.MAX_STRUCTURAL_SPEED]

        CATD = self.options[Aircraft.Design.PART25_STRUCTURAL_CATEGORY]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        WGS_greater_than_20_flag = self.options[Aircraft.Wing.LOADING_ABOVE_20]

        max_struct_speed_kts = max_struct_speed_mph / 1.15
        dmax_struct_speed_kts_dmax_struct_speed_mph = 1 / 1.15
        dmax_struct_speed_kts_dgross_mass = 0.0
        dmax_struct_speed_kts_dexp_wing_area = 0.0

        if CATD < 3:
            gross_mass = inputs[Mission.Design.GROSS_MASS]
            exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
            wing_loading = gross_mass / exp_wing_area
            dwing_loading_dgross_mass = 1 / exp_wing_area
            dwing_loading_dexp_wing_area = -gross_mass / exp_wing_area**2

            VCMAX = 0.9 * max_struct_speed_kts
            dVCMAX_dmax_struct_speed_mph = 0.9 / 1.15

            if CATD <= 1:
                if WGS_greater_than_20_flag:
                    VCCOF = 33.0 - (0.0550 * (wing_loading - 20.0))
                    dVCCOF_dwing_loading = -0.0550
                    dVCCOF_dgross_mass = dVCCOF_dwing_loading * dwing_loading_dgross_mass
                    dVCCOF_dexp_wing_area = dVCCOF_dwing_loading * dwing_loading_dexp_wing_area
                else:
                    VCCOF = 33.0
                    dVCCOF_dwing_loading = 0.0
                    dVCCOF_dgross_mass = 0.0
                    dVCCOF_dexp_wing_area = 0.0
            elif CATD == 2:
                if WGS_greater_than_20_flag:
                    VCCOF = 36.0 - (0.0925 * (wing_loading - 20.0))
                    dVCCOF_dwing_loading = -0.0925
                    dVCCOF_dgross_mass = dVCCOF_dwing_loading * dwing_loading_dgross_mass
                    dVCCOF_dexp_wing_area = dVCCOF_dwing_loading * dwing_loading_dexp_wing_area
                else:
                    VCCOF = 36
                    dVCCOF_dwing_loading = 0.0
                    dVCCOF_dgross_mass = 0.0
                    dVCCOF_dexp_wing_area = 0.0

            wl_sqrt = wing_loading**0.5
            d_wl_sqrt_d_wing_loading = 0.5 * wing_loading**-0.5
            VCMIN = VCCOF * wl_sqrt
            dVCMIN_dgross_mass = (
                dVCCOF_dgross_mass * wl_sqrt
                + VCCOF * d_wl_sqrt_d_wing_loading * dwing_loading_dgross_mass
            )
            dVCMIN_dexp_wing_area = (
                dVCCOF_dexp_wing_area * wl_sqrt
                + VCCOF * d_wl_sqrt_d_wing_loading * dwing_loading_dexp_wing_area
            )
            dVCMIN_dmax_struct_speed_mph = 0.0

            if smooth:
                SigA = sigmoidX(VCMIN / VCMAX, 1, -0.01)
                SigB = sigmoidX(VCMIN / VCMAX, 1, 0.01)
                DSigB = dSigmoidXdx(VCMIN / VCMAX, 1, 0.01)
                VCMIN_1 = VCMIN * SigA + VCMAX * SigB
                dVCMIN_dgross_mass = (
                    dVCMIN_dgross_mass * SigA
                    - VCMIN * DSigB * dVCMIN_dgross_mass / VCMAX
                    + VCMAX * DSigB * dVCMIN_dgross_mass / VCMAX
                )
                dVCMIN_dexp_wing_area = (
                    dVCMIN_dexp_wing_area * SigA
                    - VCMIN * DSigB * dVCMIN_dexp_wing_area / VCMAX
                    + VCMAX * DSigB * dVCMIN_dexp_wing_area / VCMAX
                )
                dVCMIN_dmax_struct_speed_mph = (
                    dVCMIN_dmax_struct_speed_mph * sigmoidX(VCMIN / VCMAX, 1, -0.01)
                    + VCMIN
                    * DSigB
                    * dquotient(
                        (VCMAX - VCMIN),
                        VCMAX,
                        dVCMAX_dmax_struct_speed_mph - dVCMIN_dmax_struct_speed_mph,
                        dVCMAX_dmax_struct_speed_mph,
                    )
                    + dVCMAX_dmax_struct_speed_mph * SigB
                    + VCMAX
                    * DSigB
                    * dquotient(
                        (VCMIN - VCMAX),
                        VCMAX,
                        dVCMIN_dmax_struct_speed_mph - dVCMAX_dmax_struct_speed_mph,
                        dVCMAX_dmax_struct_speed_mph,
                    )
                )
                VCMIN = VCMIN_1
            else:
                if VCMIN > VCMAX:
                    VCMIN = VCMAX
                    dVCMIN_dmax_struct_speed_mph = dVCMAX_dmax_struct_speed_mph
                    dVCMIN_dgross_mass = 0.0
                    dVCMIN_dexp_wing_area = 0.0

            if CATD == 0:
                if WGS_greater_than_20_flag:
                    VDCOF = 1.4 - (0.000625 * (wing_loading - 20.0))
                    dVDCOF_dwing_loading = -0.000625
                    dVDCOF_dgross_mass = dVDCOF_dwing_loading * dwing_loading_dgross_mass
                    dVDCOF_dexp_wing_area = dVDCOF_dwing_loading * dwing_loading_dexp_wing_area
                else:
                    VDCOF = 1.4
                    dVDCOF_dwing_loading = 0.0
                    dVDCOF_dgross_mass = 0.0
                    dVDCOF_dexp_wing_area = 0.0
            if CATD == 1:
                if WGS_greater_than_20_flag:
                    VDCOF = 1.5 - (0.001875 * (wing_loading - 20.0))
                    dVDCOF_dwing_loading = -0.001875
                    dVDCOF_dgross_mass = dVDCOF_dwing_loading * dwing_loading_dgross_mass
                    dVDCOF_dexp_wing_area = dVDCOF_dwing_loading * dwing_loading_dexp_wing_area
                else:
                    VDCOF = 1.5
                    dVDCOF_dwing_loading = 0.0
                    dVDCOF_dgross_mass = 0.0
                    dVDCOF_dexp_wing_area = 0.0
            if CATD == 2:
                if WGS_greater_than_20_flag:
                    VDCOF = 1.55 - (0.0025 * (wing_loading - 20.0))
                    dVDCOF_dwing_loading = -0.0025
                    dVDCOF_dgross_mass = dVDCOF_dwing_loading * dwing_loading_dgross_mass
                    dVDCOF_dexp_wing_area = dVDCOF_dwing_loading * dwing_loading_dexp_wing_area
                else:
                    VDCOF = 1.55
                    dVDCOF_dwing_loading = 0.0
                    dVDCOF_dgross_mass = 0.0
                    dVDCOF_dexp_wing_area = 0.0

            min_dive_vel = VDCOF * VCMIN
            dmin_dive_vel_dgross_mass = dVDCOF_dgross_mass * VCMIN + VDCOF * dVCMIN_dgross_mass
            dmin_dive_vel_dexp_wing_area = (
                dVDCOF_dexp_wing_area * VCMIN + VDCOF * dVCMIN_dexp_wing_area
            )
            dmin_dive_vel_dmax_struct_speed_mph = VDCOF * dVCMIN_dmax_struct_speed_mph

            if smooth:
                SigC = sigmoidX(min_dive_vel / max_struct_speed_kts, 1, -0.01)
                SigD = sigmoidX(min_dive_vel / max_struct_speed_kts, 1, 0.01)
                DSigD = dSigmoidXdx(min_dive_vel / max_struct_speed_kts, 1, 0.01)
                min_dive_vel_1 = max_struct_speed_kts * SigC + min_dive_vel * SigD
                dmin_dive_vel_dmax_struct_speed_mph = (
                    dmax_struct_speed_kts_dmax_struct_speed_mph * SigC
                    + max_struct_speed_kts
                    * DSigD
                    * dquotient(
                        (max_struct_speed_kts - min_dive_vel),
                        max_struct_speed_kts,
                        (
                            dmax_struct_speed_kts_dmax_struct_speed_mph
                            - dmin_dive_vel_dmax_struct_speed_mph
                        ),
                        dmax_struct_speed_kts_dmax_struct_speed_mph,
                    )
                    + dmin_dive_vel_dmax_struct_speed_mph * SigD
                    + min_dive_vel
                    * DSigD
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dmax_struct_speed_mph
                        - dmax_struct_speed_kts_dmax_struct_speed_mph,
                        dmax_struct_speed_kts_dmax_struct_speed_mph,
                    )
                )
                dmin_dive_vel_dgross_mass = (
                    dmax_struct_speed_kts_dgross_mass * SigC
                    + max_struct_speed_kts
                    * DSigD
                    * dquotient(
                        max_struct_speed_kts - min_dive_vel,
                        max_struct_speed_kts,
                        dmax_struct_speed_kts_dgross_mass - dmin_dive_vel_dgross_mass,
                        dmax_struct_speed_kts_dgross_mass,
                    )
                    + dmin_dive_vel_dgross_mass * SigD
                    + min_dive_vel
                    * DSigD
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dgross_mass - dmax_struct_speed_kts_dgross_mass,
                        dmax_struct_speed_kts_dgross_mass,
                    )
                )
                dmin_dive_vel_dexp_wing_area = (
                    dmax_struct_speed_kts_dexp_wing_area * SigC
                    + max_struct_speed_kts
                    * DSigD
                    * dquotient(
                        max_struct_speed_kts - min_dive_vel,
                        max_struct_speed_kts,
                        dmax_struct_speed_kts_dexp_wing_area - dmin_dive_vel_dexp_wing_area,
                        dmax_struct_speed_kts_dexp_wing_area,
                    )
                    + dmin_dive_vel_dexp_wing_area * SigD
                    + min_dive_vel
                    * DSigD
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dexp_wing_area - dmax_struct_speed_kts_dexp_wing_area,
                        dmax_struct_speed_kts_dexp_wing_area,
                    )
                )
                min_dive_vel = min_dive_vel_1
            else:
                if min_dive_vel < max_struct_speed_kts:  # note: this creates a discontinuity
                    min_dive_vel = max_struct_speed_kts
                    dmin_dive_vel_dgross_mass = 0
                    dmin_dive_vel_dexp_wing_area = 0
                    dmin_dive_vel_dmax_struct_speed_mph = (
                        dmax_struct_speed_kts_dmax_struct_speed_mph
                    )

            partials['min_dive_vel', Mission.Design.GROSS_MASS] = dmin_dive_vel_dgross_mass
            partials['min_dive_vel', Aircraft.Wing.EXPOSED_AREA] = dmin_dive_vel_dexp_wing_area
            partials['min_dive_vel', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                dmin_dive_vel_dmax_struct_speed_mph
            )

            partials['max_airspeed', Mission.Design.GROSS_MASS] = 0.85 * dmin_dive_vel_dgross_mass
            partials['max_airspeed', Aircraft.Wing.EXPOSED_AREA] = (
                0.85 * dmin_dive_vel_dexp_wing_area
            )
            partials['max_airspeed', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                0.85 * dmin_dive_vel_dmax_struct_speed_mph
            )

            partials['vel_c', Mission.Design.GROSS_MASS] = dVCMIN_dgross_mass
            partials['vel_c', Aircraft.Wing.EXPOSED_AREA] = dVCMIN_dexp_wing_area
            partials['vel_c', Aircraft.Design.MAX_STRUCTURAL_SPEED] = dVCMIN_dmax_struct_speed_mph

        if CATD == 3:
            partials['max_airspeed', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['min_dive_vel', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.2 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['vel_c', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
        elif CATD > 3.001:
            partials['max_airspeed', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['min_dive_vel', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.2 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials['vel_c', Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )


class BWBLoadFactors(om.ExplicitComponent):
    """
    Computation of structural ultimate load factor.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.EXPOSED_AREA, units='ft**2')

        self.add_input(
            'density_ratio',
            units='unitless',
            desc='SIGMA (in GASP): density ratio = density at Altitude / density at Sea level',
        )
        self.add_input(
            'V9',
            units='kn',
            desc='V9: intermediate value. Typically it is maximum flight speed.',
        )
        self.add_input('min_dive_vel', units='kn', desc='VDMIN: dive velocity')
        self.add_input(
            'max_maneuver_factor',
            units='unitless',
            desc='EMLF: maximum maneuver load factor, units are in g`s',
        )
        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, units='ft')
        add_aviary_input(self, Aircraft.Design.LIFT_CURVE_SLOPE, units='1/rad')

        add_aviary_output(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')

        self.declare_partials(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        ULF_from_maneuver = self.options[Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER]

        if ULF_from_maneuver == True:
            ULF = 1.5 * max_maneuver_factor
        else:
            gross_mass = inputs[Mission.Design.GROSS_MASS]
            exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
            if verbosity > Verbosity.BRIEF:
                if exp_wing_area <= 0.0:
                    print('Aircraft.Wing.EXPOSED_AREA must be positive.')
                if gross_mass <= 0.0:
                    print('Mission.Design.GROSS_MASS must be positive.')
            wing_loading = gross_mass / exp_wing_area

            density_ratio = inputs['density_ratio']
            V9 = inputs['V9']
            min_dive_vel = inputs['min_dive_vel']
            max_maneuver_factor = inputs['max_maneuver_factor']
            avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
            Cl_alpha = inputs[Aircraft.Design.LIFT_CURVE_SLOPE]

            smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

            mass_ratio = (
                2.0
                * wing_loading
                / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
            )
            k_load_factor = 0.88 * mass_ratio / (5.3 + mass_ratio)
            cruise_load_factor = 1.0 + (
                (k_load_factor * 50.0 * V9 * Cl_alpha) / (498.0 * wing_loading)
            )
            dive_load_factor = 1.0 + (
                (k_load_factor * 25.0 * min_dive_vel * Cl_alpha) / (498.0 * wing_loading)
            )

            # set gust_load_factor between cruise_load_factor and dive_load_factor
            if smooth:
                gust_load_factor = dive_load_factor * sigmoidX(
                    cruise_load_factor / dive_load_factor, 1, -0.01
                ) + cruise_load_factor * sigmoidX(cruise_load_factor / dive_load_factor, 1, 0.01)
            else:
                if cruise_load_factor > dive_load_factor:
                    gust_load_factor = cruise_load_factor
                else:
                    gust_load_factor = dive_load_factor

            # set ULF between max_maneuver_factor and gust_load_factor
            if smooth:
                ULF = 1.5 * (
                    gust_load_factor * sigmoidX(max_maneuver_factor / gust_load_factor, 1, -0.01)
                    + max_maneuver_factor
                    * sigmoidX(max_maneuver_factor / gust_load_factor, 1, 0.01)
                )
            else:
                if gust_load_factor > max_maneuver_factor:
                    ULF = 1.5 * gust_load_factor
                else:
                    ULF = 1.5 * max_maneuver_factor

        outputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = ULF

    def compute_partials(self, inputs, partials):
        ULF_from_maneuver = self.options[Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER]

        if ULF_from_maneuver == True:
            # ULF = 1.5 * max_maneuver_factor
            dULF_dmax_maneuver_factor = 1.5
            dULF_dgross_mass = 0.0
            dULF_dexp_wing_areas = 0.0
            dULF_ddensity_ratio = 0.0
            dULF_davg_chord = 0.0
            dULF_dCl_alpha = 0.0
            dULF_dV9 = 0.0
            dULF_dmin_dive_vel = 0.0
        else:
            gross_mass = inputs[Mission.Design.GROSS_MASS]
            exp_wing_area = inputs[Aircraft.Wing.EXPOSED_AREA]
            wing_loading = gross_mass / exp_wing_area
            dwing_loading_dgross_mass = 1 / exp_wing_area
            dwing_loading_dexp_wing_area = -gross_mass / exp_wing_area**2

            density_ratio = inputs['density_ratio']
            V9 = inputs['V9']
            min_dive_vel = inputs['min_dive_vel']
            max_maneuver_factor = inputs['max_maneuver_factor']
            avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
            Cl_alpha = inputs[Aircraft.Design.LIFT_CURVE_SLOPE]

            smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]

            mass_ratio = (
                2.0
                * wing_loading
                / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
            )
            k_load_factor = 0.88 * mass_ratio / (5.3 + mass_ratio)
            cruise_load_factor = 1.0 + (
                (k_load_factor * 50.0 * V9 * Cl_alpha) / (498.0 * wing_loading)
            )
            dive_load_factor = 1.0 + (
                (k_load_factor * 25.0 * min_dive_vel * Cl_alpha) / (498.0 * wing_loading)
            )

            dmass_ratio_dgross_mass = (
                2.0
                * dwing_loading_dgross_mass
                / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
            )
            dmass_ratio_dexp_wing_area = (
                2.0
                * dwing_loading_dexp_wing_area
                / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
            )
            dmass_ratio_ddensity_ratio = (
                -2.0
                * wing_loading
                / (density_ratio**2 * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
            )
            dmass_ratio_davg_chord = (
                -2.0
                * wing_loading
                / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord**2 * Cl_alpha * 32.2)
            )
            dmass_ratio_dCl_alpha = (
                -2.0
                * wing_loading
                / (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha**2 * 32.2)
            )

            dk_load_factor_dgross_mass = dquotient(
                0.88 * mass_ratio,
                5.3 + mass_ratio,
                0.88 * dmass_ratio_dgross_mass,
                dmass_ratio_dgross_mass,
            )
            dk_load_factor_dexp_wing_areas = dquotient(
                0.88 * mass_ratio,
                5.3 + mass_ratio,
                0.88 * dmass_ratio_dexp_wing_area,
                dmass_ratio_dexp_wing_area,
            )
            dk_load_factor_ddensity_ratio = dquotient(
                0.88 * mass_ratio,
                5.3 + mass_ratio,
                0.88 * dmass_ratio_ddensity_ratio,
                dmass_ratio_ddensity_ratio,
            )
            dk_load_factor_davg_chord = dquotient(
                0.88 * mass_ratio,
                5.3 + mass_ratio,
                0.88 * dmass_ratio_davg_chord,
                dmass_ratio_davg_chord,
            )
            dk_load_factor_dCl_alpha = dquotient(
                0.88 * mass_ratio,
                5.3 + mass_ratio,
                0.88 * dmass_ratio_dCl_alpha,
                dmass_ratio_dCl_alpha,
            )

            dcruise_load_factor_dgross_mass = dquotient(
                (k_load_factor * 50.0 * V9 * Cl_alpha),
                (498.0 * wing_loading),
                (dk_load_factor_dgross_mass * 50.0 * V9 * Cl_alpha),
                498 * dwing_loading_dgross_mass,
            )
            dcruise_load_factor_dexp_wing_areas = dquotient(
                (k_load_factor * 50.0 * V9 * Cl_alpha),
                (498.0 * wing_loading),
                (dk_load_factor_dexp_wing_areas * 50.0 * V9 * Cl_alpha),
                498 * dwing_loading_dexp_wing_area,
            )
            dcruise_load_factor_ddensity_ratio = (
                dk_load_factor_ddensity_ratio * 50.0 * V9 * Cl_alpha
            ) / (498.0 * wing_loading)
            dcruise_load_factor_davg_chord = (dk_load_factor_davg_chord * 50.0 * V9 * Cl_alpha) / (
                498.0 * wing_loading
            )
            dcruise_load_factor_dCl_alpha = (
                dk_load_factor_dCl_alpha * 50.0 * V9 * Cl_alpha + k_load_factor * 50 * V9
            ) / (498.0 * wing_loading)
            dcruise_load_factor_dV9 = (k_load_factor * 50.0 * Cl_alpha) / (498.0 * wing_loading)
            dcruise_load_factor_dmin_dive_vel = 0.0

            ddive_load_factor_dgross_mass = dquotient(
                (k_load_factor * 25.0 * min_dive_vel * Cl_alpha),
                (498.0 * wing_loading),
                (dk_load_factor_dgross_mass * 25.0 * min_dive_vel * Cl_alpha),
                498 * dwing_loading_dgross_mass,
            )
            ddive_load_factor_dexp_wing_areas = dquotient(
                (k_load_factor * 25.0 * min_dive_vel * Cl_alpha),
                (498.0 * wing_loading),
                (dk_load_factor_dexp_wing_areas * 25.0 * min_dive_vel * Cl_alpha),
                498 * dwing_loading_dexp_wing_area,
            )
            ddive_load_factor_ddensity_ratio = (
                dk_load_factor_ddensity_ratio * 25 * min_dive_vel * Cl_alpha
            ) / (498.0 * wing_loading)
            ddive_load_factor_davg_chord = (
                dk_load_factor_davg_chord * 25 * min_dive_vel * Cl_alpha
            ) / (498.0 * wing_loading)
            ddive_load_factor_dCl_alpha = (
                dk_load_factor_dCl_alpha * 25 * min_dive_vel * Cl_alpha
                + k_load_factor * 50 * min_dive_vel
            ) / (498.0 * wing_loading)
            ddive_load_factor_dmin_dive_vel = (k_load_factor * 25 * Cl_alpha) / (
                498.0 * wing_loading
            )
            ddive_load_factor_dV9 = 0.0

            # set gust_load_factor and gust_load_factor partials
            if smooth:
                SigG = sigmoidX(cruise_load_factor / dive_load_factor, 1, -0.01)
                SigH = sigmoidX(cruise_load_factor / dive_load_factor, 1, 0.01)
                DSigG = dSigmoidXdx(cruise_load_factor / dive_load_factor, 1, -0.01)
                DSigH = dSigmoidXdx(cruise_load_factor / dive_load_factor, 1, 0.01)
                gust_load_factor = dive_load_factor * SigG + cruise_load_factor * SigH
                dgust_load_factor_dgross_mass = (
                    ddive_load_factor_dgross_mass * SigG
                    + dive_load_factor
                    * DSigG
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_dgross_mass - dcruise_load_factor_dgross_mass,
                        ddive_load_factor_dgross_mass,
                    )
                    + dcruise_load_factor_dgross_mass * SigH
                    + cruise_load_factor
                    * DSigG
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        dcruise_load_factor_dgross_mass - ddive_load_factor_dgross_mass,
                        ddive_load_factor_dgross_mass,
                    )
                )
                dgust_load_factor_dexp_wing_areas = (
                    ddive_load_factor_dexp_wing_areas * SigG
                    + dive_load_factor
                    * DSigG
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_dexp_wing_areas - dcruise_load_factor_dexp_wing_areas,
                        ddive_load_factor_dexp_wing_areas,
                    )
                    + dcruise_load_factor_dexp_wing_areas * SigH
                    + cruise_load_factor
                    * DSigG
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        dcruise_load_factor_dexp_wing_areas - ddive_load_factor_dexp_wing_areas,
                        ddive_load_factor_dexp_wing_areas,
                    )
                )
                dgust_load_factor_ddensity_ratio = (
                    ddive_load_factor_ddensity_ratio * SigG
                    + dive_load_factor
                    * DSigG
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_ddensity_ratio - dcruise_load_factor_ddensity_ratio,
                        ddive_load_factor_ddensity_ratio,
                    )
                    + dcruise_load_factor_ddensity_ratio * SigH
                    + cruise_load_factor
                    * DSigH
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        dcruise_load_factor_ddensity_ratio - ddive_load_factor_ddensity_ratio,
                        ddive_load_factor_ddensity_ratio,
                    )
                )
                dgust_load_factor_davg_chord = (
                    ddive_load_factor_davg_chord * SigG
                    + dive_load_factor
                    * DSigH
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_davg_chord - dcruise_load_factor_davg_chord,
                        ddive_load_factor_davg_chord,
                    )
                    + dcruise_load_factor_davg_chord * SigH
                    + cruise_load_factor
                    * DSigH
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        dcruise_load_factor_davg_chord - ddive_load_factor_davg_chord,
                        ddive_load_factor_davg_chord,
                    )
                )
                dgust_load_factor_dCl_alpha = (
                    ddive_load_factor_dCl_alpha * SigG
                    + dive_load_factor
                    * DSigH
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_dCl_alpha - dcruise_load_factor_dCl_alpha,
                        ddive_load_factor_dCl_alpha,
                    )
                    + dcruise_load_factor_dCl_alpha * SigH
                    + cruise_load_factor
                    * DSigH
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        dcruise_load_factor_dCl_alpha - ddive_load_factor_dCl_alpha,
                        ddive_load_factor_dCl_alpha,
                    )
                )
                dgust_load_factor_dV9 = (
                    dive_load_factor
                    * DSigH
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_dV9 - dcruise_load_factor_dV9,
                        ddive_load_factor_dV9,
                    )
                    + dcruise_load_factor_dV9 * SigH
                    + cruise_load_factor
                    * DSigH
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        dcruise_load_factor_dV9 - ddive_load_factor_dV9,
                        ddive_load_factor_dV9,
                    )
                )
                dgust_loading_dmin_dive_vel = (
                    ddive_load_factor_dmin_dive_vel * SigG
                    + dive_load_factor
                    * DSigH
                    * dquotient(
                        (dive_load_factor - cruise_load_factor),
                        dive_load_factor,
                        ddive_load_factor_dmin_dive_vel,
                        ddive_load_factor_dmin_dive_vel,
                    )
                    + cruise_load_factor
                    * DSigH
                    * dquotient(
                        (cruise_load_factor - dive_load_factor),
                        dive_load_factor,
                        -ddive_load_factor_dmin_dive_vel,
                        ddive_load_factor_dmin_dive_vel,
                    )
                )
            else:
                if cruise_load_factor > dive_load_factor:
                    gust_load_factor = cruise_load_factor
                    dgust_load_factor_dgross_mass = dcruise_load_factor_dgross_mass
                    dgust_load_factor_dexp_wing_areas = dcruise_load_factor_dexp_wing_areas
                    dgust_load_factor_ddensity_ratio = dcruise_load_factor_ddensity_ratio
                    dgust_load_factor_davg_chord = dcruise_load_factor_davg_chord
                    dgust_load_factor_dCl_alpha = dcruise_load_factor_dCl_alpha
                    dgust_load_factor_dV9 = dcruise_load_factor_dV9
                    dgust_load_factor_dmin_dive_vel = 0.0
                else:
                    gust_load_factor = dive_load_factor
                    dgust_load_factor_dgross_mass = dquotient(
                        (k_load_factor * 25.0 * min_dive_vel * Cl_alpha),
                        (498.0 * wing_loading),
                        (dk_load_factor_dgross_mass * 25.0 * min_dive_vel * Cl_alpha),
                        498 * dwing_loading_dgross_mass,
                    )
                    dgust_load_factor_dexp_wing_areas = dquotient(
                        (k_load_factor * 25.0 * min_dive_vel * Cl_alpha),
                        (498.0 * wing_loading),
                        (dk_load_factor_dexp_wing_areas * 25.0 * min_dive_vel * Cl_alpha),
                        498 * dwing_loading_dexp_wing_area,
                    )
                    dgust_load_factor_ddensity_ratio = (
                        dk_load_factor_ddensity_ratio * 25 * min_dive_vel * Cl_alpha
                    ) / (498.0 * wing_loading)
                    dgust_load_factor_davg_chord = (
                        dk_load_factor_davg_chord * 25 * min_dive_vel * Cl_alpha
                    ) / (498.0 * wing_loading)
                    dgust_load_factor_dCl_alpha = (
                        dk_load_factor_dCl_alpha * 25 * min_dive_vel * Cl_alpha
                        + k_load_factor * 25 * min_dive_vel
                    ) / (498.0 * wing_loading)
                    dgust_load_factor_dmin_dive_vel = (k_load_factor * 25 * Cl_alpha) / (
                        498.0 * wing_loading
                    )
                    dgust_load_factor_dV9 = 0.0

            # set ULF partials
            if smooth:
                SigK = sigmoidX(max_maneuver_factor / gust_load_factor, 1, -0.01)
                SigL = sigmoidX(max_maneuver_factor / gust_load_factor, 1, 0.01)
                DSigL = dSigmoidXdx(max_maneuver_factor / gust_load_factor, 1, 0.01)
                # ULF = 1.5 * (
                #     gust_load_factor
                #     * sigmoidX(max_maneuver_factor / gust_load_factor, 1, 0.01)
                #     + max_maneuver_factor
                #     * sigmoidX(max_maneuver_factor / gust_load_factor, 1, 0.01)
                # )
                dULF_dmax_maneuver_factor = 1.5 * (
                    gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        -1.0,
                        0.0,
                    )
                    + SigL
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor), gust_load_factor, 1.0, 0.0
                    )
                )
                dULF_dgross_mass = 1.5 * (
                    dgust_load_factor_dgross_mass * SigK
                    + gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        dgust_load_factor_dgross_mass,
                        dgust_load_factor_dgross_mass,
                    )
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor),
                        gust_load_factor,
                        -dgust_load_factor_dgross_mass,
                        dgust_load_factor_dgross_mass,
                    )
                )
                dULF_dexp_wing_areas = 1.5 * (
                    dgust_load_factor_dexp_wing_areas * SigK
                    + gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        dgust_load_factor_dexp_wing_areas,
                        dgust_load_factor_dexp_wing_areas,
                    )
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor),
                        gust_load_factor,
                        -dgust_load_factor_dexp_wing_areas,
                        dgust_load_factor_dexp_wing_areas,
                    )
                )
                dULF_ddensity_ratio = 1.5 * (
                    dgust_load_factor_ddensity_ratio * SigK
                    + gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        dgust_load_factor_ddensity_ratio,
                        dgust_load_factor_ddensity_ratio,
                    )
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor),
                        gust_load_factor,
                        -dgust_load_factor_ddensity_ratio,
                        dgust_load_factor_ddensity_ratio,
                    )
                )
                dULF_davg_chord = 1.5 * (
                    dgust_load_factor_davg_chord * SigK
                    + gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        dgust_load_factor_davg_chord,
                        dgust_load_factor_davg_chord,
                    )
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor),
                        gust_load_factor,
                        -dgust_load_factor_davg_chord,
                        dgust_load_factor_davg_chord,
                    )
                )
                dULF_dCl_alpha = 1.5 * (
                    dgust_load_factor_dCl_alpha * SigK
                    + gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        dgust_load_factor_dCl_alpha,
                        dgust_load_factor_dCl_alpha,
                    )
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor),
                        gust_load_factor,
                        -dgust_load_factor_dCl_alpha,
                        dgust_load_factor_dCl_alpha,
                    )
                )
                dULF_dV9 = 1.5 * (
                    dgust_load_factor_dV9 * SigK
                    + gust_load_factor
                    * DSigL
                    * dquotient(
                        (gust_load_factor - max_maneuver_factor),
                        gust_load_factor,
                        dgust_load_factor_dV9,
                        dgust_load_factor_dV9,
                    )
                    + max_maneuver_factor
                    * DSigL
                    * dquotient(
                        (max_maneuver_factor - gust_load_factor),
                        gust_load_factor,
                        -dgust_load_factor_dV9,
                        dgust_load_factor_dV9,
                    )
                )
                dULF_dmin_dive_vel = 0.0
            else:
                if gust_load_factor > max_maneuver_factor:
                    # ULF = 1.5 * gust_load_factor
                    dULF_dmax_maneuver_factor = 0.0
                    dULF_dgross_mass = 1.5 * dgust_load_factor_dgross_mass
                    dULF_dexp_wing_areas = 1.5 * dgust_load_factor_dexp_wing_areas
                    dULF_ddensity_ratio = 1.5 * dgust_load_factor_ddensity_ratio
                    dULF_davg_chord = 1.5 * dgust_load_factor_davg_chord
                    dULF_dCl_alpha = 1.5 * dgust_load_factor_dCl_alpha
                    dULF_dV9 = 1.5 * dgust_load_factor_dV9
                    dULF_dmin_dive_vel = 1.5 * dgust_load_factor_dmin_dive_vel
                else:
                    # ULF = 1.5 * max_maneuver_factor
                    dULF_dmax_maneuver_factor = 1.5
                    dULF_dgross_mass = 0.0
                    dULF_dexp_wing_areas = 0.0
                    dULF_ddensity_ratio = 0.0
                    dULF_davg_chord = 0.0
                    dULF_dCl_alpha = 0.0
                    dULF_dV9 = 0.0
                    dULF_dmin_dive_vel = 0.0

        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'max_maneuver_factor'] = (
            dULF_dmax_maneuver_factor
        )
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Mission.Design.GROSS_MASS] = dULF_dgross_mass
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Aircraft.Wing.EXPOSED_AREA] = (
            dULF_dexp_wing_areas
        )
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'density_ratio'] = dULF_ddensity_ratio
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Aircraft.Wing.AVERAGE_CHORD] = dULF_davg_chord
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, Aircraft.Design.LIFT_CURVE_SLOPE] = (
            dULF_dCl_alpha
        )
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'V9'] = dULF_dV9
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 'min_dive_vel'] = dULF_dmin_dive_vel


class BWBDesignLoadGroup(om.Group):
    """
    Design load group for GASP-based BWB type aircraft mass.
    """

    def setup(self):
        self.add_subsystem(
            'speeds',
            BWBLoadSpeeds(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['max_airspeed', 'vel_c', 'max_maneuver_factor', 'min_dive_vel'],
        )

        self.add_subsystem(
            'params',
            LoadParameters(),
            promotes_inputs=['max_airspeed', 'vel_c'],
            promotes_outputs=['density_ratio', 'V9', 'max_mach'],
        )

        self.add_subsystem(
            'CL_Alpha',
            LiftCurveSlopeAtCruise(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'factors',
            BWBLoadFactors(),
            promotes_inputs=['max_maneuver_factor', 'min_dive_vel', 'density_ratio', 'V9']
            + ['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )
