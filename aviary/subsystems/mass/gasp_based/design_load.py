import numpy as np
import openmdao.api as om

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


def sig(x):
    return 1 / (1 + np.exp(-100 * x))


def dsig(x):
    return 100 * np.exp(-100 * x) / (np.exp(-100 * x) + 1) ** 2


def dquotient(u, v, du, dv):
    return (du * v - u * dv) / v**2


class LoadSpeeds(om.ExplicitComponent):
    """
    Computation of load speeds (such as maximum operating equivalent airspeed,
    velocity used in Gust Load Factor calculation at cruise conditions, maximum
    maneuver load factor, and minimum dive velocity).
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Aircraft.Design.MAX_STRUCTURAL_SPEED,
                         val=200, units="mi/h")

        if self.options["aviary_options"].get_val(Aircraft.Design.PART25_STRUCTURAL_CATEGORY, units='unitless') < 3:

            add_aviary_input(self, Aircraft.Wing.LOADING, val=128)

        self.add_output(
            "max_airspeed",
            val=0,
            units="kn",
            desc="VM0: maximum operating equivalent airspeed",
        )
        self.add_output(
            "vel_c",
            val=0,
            units="kn",
            desc="VGC: Velocity used in Gust Load Factor calculation at cruise conditions.\
                This is Minimum Design Cruise Speed for Part 23 aircraft and VM0 for Part 25 aircraft",
        )
        self.add_output(
            "max_maneuver_factor",
            val=0,
            units="unitless",
            desc="EMLF: maximum maneuver load factor, units are in g`s",
        )
        self.add_output("min_dive_vel", val=0, units="kn", desc="VDMIN: dive velocity")

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):

        max_struct_speed_mph = inputs[Aircraft.Design.MAX_STRUCTURAL_SPEED]

        CATD = self.options["aviary_options"].get_val(
            Aircraft.Design.PART25_STRUCTURAL_CATEGORY, units='unitless')
        smooth = self.options["aviary_options"].get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')
        WGS_greater_than_20_flag = self.options["aviary_options"].get_val(
            Aircraft.Wing.LOADING_ABOVE_20, units='unitless')

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
                VCMIN = VCMIN * sig((VCMAX - VCMIN) / VCMAX) + VCMAX * sig(
                    (VCMIN - VCMAX) / VCMAX
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
                min_dive_vel = max_struct_speed_kts * sig(
                    (max_struct_speed_kts - min_dive_vel) / max_struct_speed_kts
                ) + min_dive_vel * sig(
                    (min_dive_vel - max_struct_speed_kts) / max_struct_speed_kts
                )
            else:
                if min_dive_vel < max_struct_speed_kts:
                    min_dive_vel = max_struct_speed_kts

            max_airspeed = 0.85 * min_dive_vel
            vel_c = VCMIN
            max_maneuver_factor = 3.8
            if CATD == 1.0:
                max_maneuver_factor = 4.4
            elif CATD == 2.0:
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

        outputs["max_airspeed"] = max_airspeed
        outputs["vel_c"] = vel_c
        outputs["max_maneuver_factor"] = max_maneuver_factor
        outputs["min_dive_vel"] = min_dive_vel

    def compute_partials(self, inputs, partials):

        max_struct_speed_mph = inputs[Aircraft.Design.MAX_STRUCTURAL_SPEED]

        CATD = self.options["aviary_options"].get_val(
            Aircraft.Design.PART25_STRUCTURAL_CATEGORY, units='unitless')
        smooth = self.options["aviary_options"].get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')
        WGS_greater_than_20_flag = self.options["aviary_options"].get_val(
            Aircraft.Wing.LOADING_ABOVE_20, units='unitless')

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
                dVCCOF_dwing_loading * (wing_loading**0.5)
                + VCCOF * 0.5 * wing_loading**-0.5
            )
            dVCMIN_dmax_struct_speed_mph = 0.0

            if smooth:
                VCMIN_1 = VCMIN * sig((VCMAX - VCMIN) / VCMAX) + VCMAX * sig(
                    (VCMIN - VCMAX) / VCMAX
                )
                dVCMIN_dwing_loading = (
                    dVCMIN_dwing_loading * sig((VCMAX - VCMIN) / VCMAX)
                    + VCMIN
                    * dsig((VCMAX - VCMIN) / VCMAX)
                    * -dVCMIN_dwing_loading
                    / VCMAX
                    + VCMAX
                    * dsig((VCMIN - VCMAX) / VCMAX)
                    * dVCMIN_dwing_loading
                    / VCMAX
                )
                dVCMIN_dmax_struct_speed_mph = (
                    dVCMIN_dmax_struct_speed_mph * sig((VCMAX - VCMIN) / VCMAX)
                    + VCMIN
                    * dsig((VCMAX - VCMIN) / VCMAX)
                    * dquotient(
                        (VCMAX - VCMIN),
                        VCMAX,
                        dVCMAX_dmax_struct_speed_mph - dVCMIN_dmax_struct_speed_mph,
                        dVCMAX_dmax_struct_speed_mph,
                    )
                    + dVCMAX_dmax_struct_speed_mph * sig((VCMIN - VCMAX) / VCMAX)
                    + VCMAX
                    * dsig((VCMIN - VCMAX) / VCMAX)
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
                min_dive_vel_1 = max_struct_speed_kts * sig(
                    (max_struct_speed_kts - min_dive_vel) / max_struct_speed_kts
                ) + min_dive_vel * sig(
                    (min_dive_vel - max_struct_speed_kts) / max_struct_speed_kts
                )
                dmin_dive_vel_dmax_struct_speed_mph = (
                    dmax_struct_speed_kts_dmax_struct_speed_mph
                    * sig((max_struct_speed_kts - min_dive_vel) / max_struct_speed_kts)
                    + max_struct_speed_kts
                    * dsig((max_struct_speed_kts - min_dive_vel) / max_struct_speed_kts)
                    * dquotient(
                        (max_struct_speed_kts - min_dive_vel),
                        max_struct_speed_kts,
                        (
                            dmax_struct_speed_kts_dmax_struct_speed_mph
                            - dmin_dive_vel_dmax_struct_speed_mph
                        ),
                        dmax_struct_speed_kts_dmax_struct_speed_mph,
                    )
                    + dmin_dive_vel_dmax_struct_speed_mph
                    * sig((min_dive_vel - max_struct_speed_kts) / max_struct_speed_kts)
                    + min_dive_vel
                    * dsig((min_dive_vel - max_struct_speed_kts) / max_struct_speed_kts)
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dmax_struct_speed_mph
                        - dmax_struct_speed_kts_dmax_struct_speed_mph,
                        dmax_struct_speed_kts_dmax_struct_speed_mph,
                    )
                )
                dmin_dive_vel_dwing_loading = (
                    dmax_struct_speed_kts_dwing_loading
                    * sig((max_struct_speed_kts - min_dive_vel) / max_struct_speed_kts)
                    + max_struct_speed_kts
                    * dsig((max_struct_speed_kts - min_dive_vel) / max_struct_speed_kts)
                    * dquotient(
                        max_struct_speed_kts - min_dive_vel,
                        max_struct_speed_kts,
                        dmax_struct_speed_kts_dwing_loading
                        - dmin_dive_vel_dwing_loading,
                        dmax_struct_speed_kts_dwing_loading,
                    )
                    + dmin_dive_vel_dwing_loading
                    * sig((min_dive_vel - max_struct_speed_kts) / max_struct_speed_kts)
                    + min_dive_vel
                    * dsig((min_dive_vel - max_struct_speed_kts) / max_struct_speed_kts)
                    * dquotient(
                        (min_dive_vel - max_struct_speed_kts),
                        max_struct_speed_kts,
                        dmin_dive_vel_dwing_loading
                        - dmax_struct_speed_kts_dwing_loading,
                        dmax_struct_speed_kts_dwing_loading,
                    )
                )
                min_dive_vel = min_dive_vel_1

            else:
                if (
                    min_dive_vel < max_struct_speed_kts
                ):  # note: this creates a discontinuity
                    min_dive_vel = max_struct_speed_kts
                    dmin_dive_vel_dwing_loading = 0
                    dmin_dive_vel_dmax_struct_speed_mph = (
                        dmax_struct_speed_kts_dmax_struct_speed_mph
                    )

            partials[
                "min_dive_vel", Aircraft.Wing.LOADING
            ] = dmin_dive_vel_dwing_loading
            partials[
                "min_dive_vel", Aircraft.Design.MAX_STRUCTURAL_SPEED
            ] = dmin_dive_vel_dmax_struct_speed_mph

            partials["max_airspeed", Aircraft.Wing.LOADING] = (
                0.85 * dmin_dive_vel_dwing_loading
            )
            partials["max_airspeed", Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                0.85 * dmin_dive_vel_dmax_struct_speed_mph
            )

            partials["vel_c", Aircraft.Wing.LOADING] = dVCMIN_dwing_loading
            partials["vel_c", Aircraft.Design.MAX_STRUCTURAL_SPEED] = dVCMIN_dmax_struct_speed_mph

        if CATD == 3:

            partials[
                "max_airspeed", Aircraft.Design.MAX_STRUCTURAL_SPEED
            ] = dmax_struct_speed_kts_dmax_struct_speed_mph
            partials["min_dive_vel", Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.2 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials["vel_c", Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )

        elif CATD > 3.001:

            partials["max_airspeed", Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials["min_dive_vel", Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.2 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )
            partials["vel_c", Aircraft.Design.MAX_STRUCTURAL_SPEED] = (
                1.0 * dmax_struct_speed_kts_dmax_struct_speed_mph
            )


class LoadParameters(om.ExplicitComponent):
    """
    Computation of load parameters (such as maximum operating mach number,
    density ratio, etc.)
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        self.add_input(
            "vel_c",
            val=100,
            units="kn",
            desc="VGC: Velocity used in Gust Load Factor calculation at cruise conditions.\
                This is Minimum Design Cruise Speed for Part 23 aircraft and VM0 for Part 25 aircraft",
        )
        self.add_input(
            "max_airspeed",
            val=200,
            units="kn",
            desc="VM0: maximum operating equivalent airspeed",
        )

        self.add_output(
            "max_mach", val=0, units="unitless", desc="EMM0: maximum operating mach number"
        )
        self.add_output(
            "density_ratio",
            val=0,
            units="unitless",
            desc="SIGMA (in GASP): density ratio = density at Altitude / density at Sea level",
        )
        self.add_output(
            "V9", val=0, units="kn", desc="V9: intermediate value. Typically it is maximum flight speed."
        )

        self.declare_partials("max_mach", "max_airspeed")
        self.declare_partials("density_ratio", "max_airspeed")
        self.declare_partials("V9", "*")

    def compute(self, inputs, outputs):

        vel_c = inputs["vel_c"]
        max_airspeed = inputs["max_airspeed"]

        cruise_alt = self.options["aviary_options"].get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')
        CATD = self.options["aviary_options"].get_val(
            Aircraft.Design.PART25_STRUCTURAL_CATEGORY, units='unitless')
        smooth = self.options["aviary_options"].get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')

        if cruise_alt <= 22500.0:
            max_mach = max_airspeed / 486.33
        if cruise_alt > 22500.0 and cruise_alt <= 36000.0:
            max_mach = max_airspeed / 424.73
            dmax_mach_dmax_airspeed = 1 / 424.73
        if cruise_alt > 36000.0:
            max_mach = max_airspeed / 372.34

        if smooth:
            max_mach = max_mach * sig((0.9 - max_mach) / 0.9) + 0.9 * sig(
                (max_mach - 0.9) / 0.9
            )

        else:
            if max_mach > 0.90:  # note: this creates a discontinuity
                max_mach = 0.90

        density_ratio = (max_airspeed / (661.7 * max_mach)) ** 1.61949

        if smooth:
            V9 = vel_c * sig(1 - density_ratio) + 661.7 * \
                max_mach * sig(density_ratio - 1)

            if CATD < 3:
                # this line creates a smooth bounded density_ratio such that .6820<=density_ratio<=1
                density_ratio = (
                    0.6820 * sig((0.6820 - density_ratio) / 0.6820)
                    + density_ratio * sig((density_ratio - 0.6820) /
                                          0.6820) * sig(1 - density_ratio)
                    + sig(density_ratio - 1)
                )

            else:
                # this line creates a smooth bounded density_ratio such that .53281<=density_ratio<=1
                density_ratio = (
                    0.53281 * sig((0.53281 - density_ratio) / 0.53281)
                    + density_ratio * sig((density_ratio - 0.53281) /
                                          0.53281) * sig(1 - density_ratio)
                    + sig(density_ratio - 1)
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

        outputs["max_mach"] = max_mach
        outputs["density_ratio"] = density_ratio
        outputs["V9"] = V9

    def compute_partials(self, inputs, partials):

        vel_c = inputs["vel_c"]
        max_airspeed = inputs["max_airspeed"]

        cruise_alt = self.options["aviary_options"].get_val(
            Mission.Design.CRUISE_ALTITUDE, units='ft')
        CATD = self.options["aviary_options"].get_val(
            Aircraft.Design.PART25_STRUCTURAL_CATEGORY, units='unitless')
        smooth = self.options["aviary_options"].get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')

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
            max_mach_1 = max_mach * sig((0.9 - max_mach) / 0.9) + 0.9 * sig(
                (max_mach - 0.9) / 0.9
            )
            dmax_mach_dmax_airspeed = (
                dmax_mach_dmax_airspeed * sig((0.9 - max_mach) / 0.9)
                + max_mach
                * dsig((0.9 - max_mach) / 0.9)
                * dquotient((0.9 - max_mach), 0.9, -dmax_mach_dmax_airspeed, 0.0)
                + 0.9
                * dsig((max_mach - 0.9) / 0.9)
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
            * dquotient(
                max_airspeed, (661.7 * max_mach), 1, 661.7 * dmax_mach_dmax_airspeed
            )
        )

        if smooth:
            V9_1 = vel_c * sig(1 - density_ratio) + 661.7 * \
                max_mach * sig(density_ratio - 1)
            dV9_dmax_airspeed = (
                vel_c * dsig(1 - density_ratio) * -ddensity_ratio_dmax_airspeed
                + 661.7
                * dmax_mach_dmax_airspeed
                * sig(density_ratio - 1)
                * 661.7
                * max_mach
                * dsig(density_ratio - 1)
                * ddensity_ratio_dmax_airspeed
            )
            dV9_dvel_c = sig(1 - density_ratio)
            V9 = V9_1

            if CATD < 3:
                # this line creates a smooth bounded density_ratio such that .6820<=density_ratio<=1
                density_ratio_1 = (
                    0.6820 * sig((0.6820 - density_ratio) / 0.6820)
                    + density_ratio * sig((density_ratio - 0.6820) /
                                          0.6820) * sig(1 - density_ratio)
                    + sig(density_ratio - 1)
                )
                ddensity_ratio_dmax_airspeed = (
                    0.6820
                    * dsig((0.6820 - density_ratio) / 0.6820)
                    * -ddensity_ratio_dmax_airspeed
                    / 0.6820
                    + ddensity_ratio_dmax_airspeed
                    * sig((density_ratio - 0.6820) / 0.6820)
                    * sig(1 - density_ratio)
                    + density_ratio
                    * (
                        dsig((density_ratio - 0.6820) / 0.6820)
                        * ddensity_ratio_dmax_airspeed
                        / 0.6820
                        * sig(1 - density_ratio)
                        + sig((density_ratio - 0.6820) / 0.6820)
                        * dsig(1 - density_ratio)
                        * -ddensity_ratio_dmax_airspeed
                    )
                    + dsig(density_ratio - 1) * ddensity_ratio_dmax_airspeed
                )
                density_ratio = density_ratio_1

            else:
                # this line creates a smooth bounded density_ratio such that .53281<=density_ratio<=1
                density_ratio_1 = (
                    0.53281 * sig((0.53281 - density_ratio) / 0.53281)
                    + density_ratio * sig((density_ratio - 0.53281) /
                                          0.53281) * sig(1 - density_ratio)
                    + sig(density_ratio - 1)
                )
                ddensity_ratio_dmax_airspeed = (
                    0.53281
                    * dsig((0.53281 - density_ratio) / 0.53281)
                    * -ddensity_ratio_dmax_airspeed
                    / 0.53281
                    + ddensity_ratio_dmax_airspeed
                    * sig((density_ratio - 0.53281) / 0.53281)
                    * sig(1 - density_ratio)
                    + density_ratio
                    * (
                        dsig((density_ratio - 0.53281) / 0.53281)
                        * ddensity_ratio_dmax_airspeed
                        / 0.53281
                        * sig(1 - density_ratio)
                        + sig((density_ratio - 0.53281) / 0.53281)
                        * dsig(1 - density_ratio)
                        * -ddensity_ratio_dmax_airspeed
                    )
                    + dsig(density_ratio - 1) * ddensity_ratio_dmax_airspeed
                )
                density_ratio = density_ratio_1
        else:
            if density_ratio >= 0.53281:  # note: this creates a discontinuity
                V9 = vel_c
                dV9_dvel_c = 1.0
                dV9_dmax_airspeed = 0.0

                if density_ratio > 1:  # note: this creates a discontinuity
                    V9 = 661.7 * max_mach
                    density_ratio = 1.0

                    dV9_dvel_c = 0.0
                    dV9_dmax_airspeed = 661.7 * dmax_mach_dmax_airspeed
                    ddensity_ratio_dmax_airspeed = 0.0

            else:  # note: this creates a discontinuity
                density_ratio = 0.53281
                V9 = vel_c

                dV9_dvel_c = 1.0
                dV9_dmax_airspeed = 0.0
                ddensity_ratio_dmax_airspeed = 0.0

            if CATD < 3.0 and density_ratio <= 0.6820:  # note: this creates a discontinuity
                density_ratio = 0.6820
                ddensity_ratio_dmax_airspeed = 0.0

        partials["max_mach", "max_airspeed"] = dmax_mach_dmax_airspeed
        partials["density_ratio", "max_airspeed"] = ddensity_ratio_dmax_airspeed
        partials["V9", "max_airspeed"] = dV9_dmax_airspeed
        partials["V9", "vel_c"] = dV9_dvel_c


class LiftCurveSlopeAtCruise(om.ExplicitComponent):
    """
    Computation of lift curve slope at cruise mach number
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=10.13)
        add_aviary_input(self, Aircraft.Wing.SWEEP, val=0.436, units="rad")
        add_aviary_input(self, Mission.Design.MACH, val=0.8)

        add_aviary_output(self, Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765)

        self.declare_partials(Aircraft.Design.LIFT_CURVE_SLOPE, "*")

    def compute(self, inputs, outputs):
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        DLMC4 = inputs[Aircraft.Wing.SWEEP]
        mach = inputs[Mission.Design.MACH]

        outputs[Aircraft.Design.LIFT_CURVE_SLOPE] = (
            np.pi * AR / (1 + np.sqrt(1 + (((AR / (2 * np.cos(DLMC4))) ** 2) * (1 - (mach * np.cos(DLMC4)) ** 2)))))

    def compute_partials(self, inputs, partials):
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        DLMC4 = inputs[Aircraft.Wing.SWEEP]
        mach = inputs[Mission.Design.MACH]

        c1 = np.sqrt(AR**2 * (-mach**2*np.cos(DLMC4)**2 + 1) + 4*np.cos(DLMC4)**2)
        c2 = 2*np.cos(DLMC4) + c1

        partials[Aircraft.Design.LIFT_CURVE_SLOPE, Aircraft.Wing.ASPECT_RATIO] = (
            4 * np.pi * np.cos(DLMC4)**2) / (c1 * c2)
        partials[Aircraft.Design.LIFT_CURVE_SLOPE, Mission.Design.MACH] = (
            2 * np.pi * AR**3 * mach * np.cos(DLMC4)**3) / (c1 * c2**2)
        partials[Aircraft.Design.LIFT_CURVE_SLOPE, Aircraft.Wing.SWEEP] = (-np.pi * AR * (-8 * np.cos(DLMC4)**3 * np.sin(
            DLMC4) + 4 * np.cos(DLMC4)**2 * np.sin(2*DLMC4) + AR**2 * np.sin(2*DLMC4))) / (np.cos(DLMC4) * c1 * c2**2)


class LoadFactors(om.ExplicitComponent):
    """
    Computation of structural ultimate load factor.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        add_aviary_input(self, Aircraft.Wing.LOADING, val=128)

        self.add_input(
            "density_ratio",
            val=0.5,
            units="unitless",
            desc="SIGMA (in GASP): density ratio = density at Altitude / density at Sea level",
        )
        self.add_input(
            "V9",
            val=100,
            units="kn",
            desc="V9: intermediate value. Typically it is maximum flight speed.",
        )
        self.add_input("min_dive_vel", val=250, units="kn", desc="VDMIN: dive velocity")
        self.add_input(
            "max_maneuver_factor",
            val=0.72,
            units="unitless",
            desc="EMLF: maximum maneuver load factor, units are in g`s",
        )

        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD, val=12.6131)
        add_aviary_input(self, Aircraft.Design.LIFT_CURVE_SLOPE, val=7.1765)

        add_aviary_output(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.5)

        self.declare_partials(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, "*")

    def compute(self, inputs, outputs):

        wing_loading = inputs[Aircraft.Wing.LOADING]
        density_ratio = inputs["density_ratio"]
        V9 = inputs["V9"]
        min_dive_vel = inputs["min_dive_vel"]
        max_maneuver_factor = inputs["max_maneuver_factor"]
        avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
        Cl_alpha = inputs[Aircraft.Design.LIFT_CURVE_SLOPE]

        ULF_from_maneuver = self.options["aviary_options"].get_val(
            Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER, units='unitless')
        smooth = self.options["aviary_options"].get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')

        mass_ratio = (
            2.0 * wing_loading /
            (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
        )
        k_load_factor = 0.88 * mass_ratio / (5.3 + mass_ratio)
        cruise_load_factor = 1.0 + (
            (k_load_factor * 50.0 * V9 * Cl_alpha) / (498.0 * wing_loading)
        )
        dive_load_factor = 1.0 + (
            (k_load_factor * 25.0 * min_dive_vel * Cl_alpha) / (498.0 * wing_loading)
        )
        gust_load_factor = dive_load_factor

        if smooth:
            gust_load_factor = dive_load_factor * sig(
                (dive_load_factor - cruise_load_factor) / dive_load_factor
            ) + cruise_load_factor * sig(
                (cruise_load_factor - dive_load_factor) / dive_load_factor
            )

        else:
            if (
                cruise_load_factor > dive_load_factor
            ):  # note: this creates a discontinuity
                gust_load_factor = cruise_load_factor

        ULF = 1.5 * max_maneuver_factor

        if smooth:
            ULF = 1.5 * (
                gust_load_factor
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + max_maneuver_factor
                * sig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
            )

        else:
            if (
                gust_load_factor > max_maneuver_factor
            ):  # note: this creates a discontinuity
                ULF = 1.5 * gust_load_factor

        if ULF_from_maneuver == True:
            ULF = 1.5 * max_maneuver_factor

        outputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = ULF

    def compute_partials(self, inputs, partials):

        wing_loading = inputs[Aircraft.Wing.LOADING]
        density_ratio = inputs["density_ratio"]
        V9 = inputs["V9"]
        min_dive_vel = inputs["min_dive_vel"]
        max_maneuver_factor = inputs["max_maneuver_factor"]
        avg_chord = inputs[Aircraft.Wing.AVERAGE_CHORD]
        Cl_alpha = inputs[Aircraft.Design.LIFT_CURVE_SLOPE]

        ULF_from_maneuver = self.options["aviary_options"].get_val(
            Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER, units='unitless')
        smooth = self.options["aviary_options"].get_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, units='unitless')

        mass_ratio = (
            2.0 * wing_loading /
            (density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2)
        )
        k_load_factor = 0.88 * mass_ratio / (5.3 + mass_ratio)
        cruise_load_factor = 1.0 + (
            (k_load_factor * 50.0 * V9 * Cl_alpha) / (498.0 * wing_loading)
        )
        dive_load_factor = 1.0 + (
            (k_load_factor * 25.0 * min_dive_vel * Cl_alpha) / (498.0 * wing_loading)
        )
        gust_load_factor = dive_load_factor

        dmass_ratio_dwing_loading = 2.0 / (
            density_ratio * RHO_SEA_LEVEL_ENGLISH * avg_chord * Cl_alpha * 32.2
        )
        dmass_ratio_ddensity_ratio = (
            -2.0 * wing_loading / (density_ratio**2 * RHO_SEA_LEVEL_ENGLISH *
                                   avg_chord * Cl_alpha * 32.2)
        )
        dmass_ratio_davg_chord = (
            -2.0 * wing_loading / (density_ratio * RHO_SEA_LEVEL_ENGLISH *
                                   avg_chord**2 * Cl_alpha * 32.2)
        )
        dmass_ratio_dCl_alpha = (
            -2.0 * wing_loading / (density_ratio * RHO_SEA_LEVEL_ENGLISH *
                                   avg_chord * Cl_alpha**2 * 32.2)
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
        dcruise_load_factor_ddensity_ratio = (dk_load_factor_ddensity_ratio * 50.0 * V9 * Cl_alpha) / (
            498.0 * wing_loading
        )
        dcruise_load_factor_davg_chord = (
            dk_load_factor_davg_chord * 50.0 * V9 * Cl_alpha
        ) / (498.0 * wing_loading)
        dcruise_load_factor_dCl_alpha = (
            dk_load_factor_dCl_alpha * 50.0 * V9 * Cl_alpha + k_load_factor * 50 * V9
        ) / (498.0 * wing_loading)
        dcruise_load_factor_dV9 = (k_load_factor * 50.0 * Cl_alpha) / (
            498.0 * wing_loading
        )
        dcruise_load_factor_dmin_dive_vel = 0.0

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
        ddive_load_factor_dmin_dive_vel = (k_load_factor * 25 * Cl_alpha) / (
            498.0 * wing_loading
        )
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
            + k_load_factor * 50 * min_dive_vel
        ) / (498.0 * wing_loading)
        dgust_load_factor_dmin_dive_vel = (k_load_factor * 25 * Cl_alpha) / (
            498.0 * wing_loading
        )
        dgust_load_factor_dV9 = 0.0

        if smooth:
            gust_load_factor_1 = dive_load_factor * sig(
                (dive_load_factor - cruise_load_factor) / dive_load_factor
            ) + cruise_load_factor * sig(
                (cruise_load_factor - dive_load_factor) / dive_load_factor
            )
            dgust_load_factor_dwing_loading = (
                ddive_load_factor_dwing_loading
                * sig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                + dive_load_factor
                * dsig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dwing_loading - dcruise_load_factor_dwing_loading,
                    ddive_load_factor_dwing_loading,
                )
                + dcruise_load_factor_dwing_loading
                * sig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                + cruise_load_factor
                * dsig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_dwing_loading - ddive_load_factor_dwing_loading,
                    ddive_load_factor_dwing_loading,
                )
            )
            dgust_load_factor_ddensity_ratio = (
                ddive_load_factor_ddensity_ratio
                * sig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                + dive_load_factor
                * dsig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_ddensity_ratio - dcruise_load_factor_ddensity_ratio,
                    ddive_load_factor_ddensity_ratio,
                )
                + dcruise_load_factor_ddensity_ratio
                * sig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                + cruise_load_factor
                * dsig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_ddensity_ratio - ddive_load_factor_ddensity_ratio,
                    ddive_load_factor_ddensity_ratio,
                )
            )
            dgust_load_factor_davg_chord = (
                ddive_load_factor_davg_chord
                * sig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                + dive_load_factor
                * dsig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_davg_chord - dcruise_load_factor_davg_chord,
                    ddive_load_factor_davg_chord,
                )
                + dcruise_load_factor_davg_chord
                * sig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                + cruise_load_factor
                * dsig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_davg_chord - ddive_load_factor_davg_chord,
                    ddive_load_factor_davg_chord,
                )
            )
            dgust_load_factor_dCl_alpha = (
                ddive_load_factor_dCl_alpha
                * sig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                + dive_load_factor
                * dsig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dCl_alpha - dcruise_load_factor_dCl_alpha,
                    ddive_load_factor_dCl_alpha,
                )
                + dcruise_load_factor_dCl_alpha
                * sig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                + cruise_load_factor
                * dsig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_dCl_alpha - ddive_load_factor_dCl_alpha,
                    ddive_load_factor_dCl_alpha,
                )
            )
            dgust_load_factor_dV9 = (
                dive_load_factor
                * dsig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dV9 - dcruise_load_factor_dV9,
                    ddive_load_factor_dV9,
                )
                + dcruise_load_factor_dV9
                * sig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                + cruise_load_factor
                * dsig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    dcruise_load_factor_dV9 - ddive_load_factor_dV9,
                    ddive_load_factor_dV9,
                )
            )
            dgust_loading_dmin_dive_vel = (
                ddive_load_factor_dmin_dive_vel
                * sig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                + dive_load_factor
                * dsig((dive_load_factor - cruise_load_factor) / dive_load_factor)
                * dquotient(
                    (dive_load_factor - cruise_load_factor),
                    dive_load_factor,
                    ddive_load_factor_dmin_dive_vel,
                    ddive_load_factor_dmin_dive_vel,
                )
                + cruise_load_factor
                * dsig((cruise_load_factor - dive_load_factor) / dive_load_factor)
                * dquotient(
                    (cruise_load_factor - dive_load_factor),
                    dive_load_factor,
                    -ddive_load_factor_dmin_dive_vel,
                    ddive_load_factor_dmin_dive_vel,
                )
            )
            gust_load_factor = gust_load_factor_1
        else:

            if (
                cruise_load_factor > dive_load_factor
            ):  # note: this creates a discontinuity
                gust_load_factor = cruise_load_factor

                dgust_load_factor_dwing_loading = dcruise_load_factor_dwing_loading
                dgust_load_factor_ddensity_ratio = dcruise_load_factor_ddensity_ratio
                dgust_load_factor_davg_chord = dcruise_load_factor_davg_chord
                dgust_load_factor_dCl_alpha = dcruise_load_factor_dCl_alpha
                dgust_load_factor_dV9 = dcruise_load_factor_dV9
                dgust_load_factor_dmin_dive_vel = 0.0

        ULF = 1.5 * max_maneuver_factor
        dULF_dmax_maneuver_factor = 1.5
        dULF_dwing_loading = 0.0
        dULF_ddensity_ratio = 0.0
        dULF_davg_chord = 0.0
        dULF_dCl_alpha = 0.0
        dULF_dV9 = 0.0
        dULF_dmin_dive_vel = 0.0

        if smooth:
            ULF_1 = 1.5 * (
                gust_load_factor
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + max_maneuver_factor
                * sig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
            )
            dULF_dmax_maneuver_factor = 1.5 * (
                gust_load_factor
                * dsig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    -1.0,
                    0.0,
                )
                + sig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                + max_maneuver_factor
                * dsig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                * dquotient(
                    (max_maneuver_factor - gust_load_factor), gust_load_factor, 1.0, 0.0
                )
            )
            dULF_dwing_loading = 1.5 * (
                dgust_load_factor_dwing_loading
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + gust_load_factor
                * dsig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_dwing_loading,
                    dgust_load_factor_dwing_loading,
                )
                + max_maneuver_factor
                * dsig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_dwing_loading,
                    dgust_load_factor_dwing_loading,
                )
            )
            dULF_ddensity_ratio = 1.5 * (
                dgust_load_factor_ddensity_ratio
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + gust_load_factor
                * dsig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_ddensity_ratio,
                    dgust_load_factor_ddensity_ratio,
                )
                + max_maneuver_factor
                * dsig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_ddensity_ratio,
                    dgust_load_factor_ddensity_ratio,
                )
            )
            dULF_davg_chord = 1.5 * (
                dgust_load_factor_davg_chord
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + gust_load_factor
                * dsig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_davg_chord,
                    dgust_load_factor_davg_chord,
                )
                + max_maneuver_factor
                * dsig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_davg_chord,
                    dgust_load_factor_davg_chord,
                )
            )
            dULF_dCl_alpha = 1.5 * (
                dgust_load_factor_dCl_alpha
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + gust_load_factor
                * dsig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_dCl_alpha,
                    dgust_load_factor_dCl_alpha,
                )
                + max_maneuver_factor
                * dsig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_dCl_alpha,
                    dgust_load_factor_dCl_alpha,
                )
            )
            dULF_dV9 = 1.5 * (
                dgust_load_factor_dV9
                * sig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                + gust_load_factor
                * dsig((gust_load_factor - max_maneuver_factor) / gust_load_factor)
                * dquotient(
                    (gust_load_factor - max_maneuver_factor),
                    gust_load_factor,
                    dgust_load_factor_dV9,
                    dgust_load_factor_dV9,
                )
                + max_maneuver_factor
                * dsig((max_maneuver_factor - gust_load_factor) / gust_load_factor)
                * dquotient(
                    (max_maneuver_factor - gust_load_factor),
                    gust_load_factor,
                    -dgust_load_factor_dV9,
                    dgust_load_factor_dV9,
                )
            )
            dULF_dmin_dive_vel = 0.0
            ULF = ULF_1
        else:
            if (
                gust_load_factor > max_maneuver_factor
            ):  # note: this creates a discontinuity
                ULF = 1.5 * gust_load_factor

                dULF_dmax_maneuver_factor = 0.0
                dULF_dwing_loading = 1.5 * dgust_load_factor_dwing_loading
                dULF_ddensity_ratio = 1.5 * dgust_load_factor_ddensity_ratio
                dULF_davg_chord = 1.5 * dgust_load_factor_davg_chord
                dULF_dCl_alpha = 1.5 * dgust_load_factor_dCl_alpha
                dULF_dV9 = 1.5 * dgust_load_factor_dV9
                dULF_dmin_dive_vel = 1.5 * dgust_load_factor_dmin_dive_vel

        if ULF_from_maneuver == True:
            ULF = 1.5 * max_maneuver_factor

            dULF_dmax_maneuver_factor = 1.5
            dULF_dwing_loading = 0.0
            dULF_ddensity_ratio = 0.0
            dULF_davg_chord = 0.0
            dULF_dCl_alpha = 0.0
            dULF_dV9 = 0.0
            dULF_dmin_dive_vel = 0.0

        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                 "max_maneuver_factor"] = dULF_dmax_maneuver_factor
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                 Aircraft.Wing.LOADING] = dULF_dwing_loading
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                 "density_ratio"] = dULF_ddensity_ratio
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                 Aircraft.Wing.AVERAGE_CHORD] = dULF_davg_chord
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                 Aircraft.Design.LIFT_CURVE_SLOPE] = dULF_dCl_alpha
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, "V9"] = dULF_dV9
        partials[Aircraft.Wing.ULTIMATE_LOAD_FACTOR, "min_dive_vel"] = dULF_dmin_dive_vel


class DesignLoadGroup(om.Group):
    """
    Design load group for GASP-based mass.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        aviary_options = self.options['aviary_options']

        self.add_subsystem(
            "speeds",
            LoadSpeeds(
                aviary_options=aviary_options,
            ),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[
                "max_airspeed",
                "vel_c",
                "max_maneuver_factor",
                "min_dive_vel",
            ],
        )

        self.add_subsystem(
            "params",
            LoadParameters(
                aviary_options=aviary_options,
            ),
            promotes_inputs=["max_airspeed", "vel_c"],
            promotes_outputs=["density_ratio", "V9", "max_mach"],
        )

        self.add_subsystem(
            "Cl_Alpha_calc",
            LiftCurveSlopeAtCruise(),
            promotes_inputs=["aircraft:*", "mission:*"],
            promotes_outputs=["aircraft:*"],
        )

        self.add_subsystem(
            "factors",
            LoadFactors(
                aviary_options=aviary_options,
            ),
            promotes_inputs=[
                "max_maneuver_factor",
                "min_dive_vel",
                "density_ratio",
                "V9",
            ]
            + ["aircraft:*"],
            promotes_outputs=[
                "aircraft:*"
            ],
        )
        self.set_input_defaults(Aircraft.Wing.LOADING, val=128, units="lbf/ft**2")
