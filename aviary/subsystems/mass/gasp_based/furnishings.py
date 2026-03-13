import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.math import sigmoidX, dSigmoidXdx, smooth_max, d_smooth_max
from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.enums import GASPEngineType


def get_num_of_flight_attendent(num_pax):
    num_flight_attendants = 0
    if num_pax >= 20.0:
        num_flight_attendants = 1
    if num_pax >= 51.0:
        num_flight_attendants = 2
    if num_pax >= 101.0:
        num_flight_attendants = 3
    if num_pax >= 151.0:
        num_flight_attendants = 4
    if num_pax >= 201.0:
        num_flight_attendants = 5
    if num_pax >= 251.0:
        num_flight_attendants = 6

    return num_flight_attendants


def get_num_of_pilots(num_pax, engine_type):
    num_pilots = 1
    if num_pax > 9.0:
        num_pilots = 2
    if engine_type is GASPEngineType.TURBOJET and num_pax > 5.0:
        num_pilots = 2
    if num_pax >= 351.0:
        num_pilots = 3

    return num_pilots


def get_num_of_lavatories(num_pax):
    num_lavatories = 0
    if num_pax > 25.0:
        num_lavatories = 1
    if num_pax >= 51.0:
        num_lavatories = 2
    if num_pax >= 101.0:
        num_lavatories = 3
    if num_pax >= 151.0:
        num_lavatories = 4
    if num_pax >= 201.0:
        num_lavatories = 5
    if num_pax >= 251.0:
        num_lavatories = 6

    return num_lavatories


class FurnishingMass(om.ExplicitComponent):
    """
    Computation of furnishing mass.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.Furnishings.USE_EMPIRICAL_EQUATION)
        self.options.declare('mu', default=1.0, types=float)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')

        self.add_output(Aircraft.Furnishings.MASS, units='lbm')

        self.declare_partials(
            Aircraft.Furnishings.MASS,
            [
                Aircraft.Fuselage.AVG_DIAMETER,
                Mission.Design.GROSS_MASS,
                Aircraft.Furnishings.MASS_SCALER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.CABIN_AREA,
            ],
        )

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            # note: this technically creates a discontinuity
            # TODO: Doesn't occur in large single aisle
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
        else:
            if PAX >= 50:
                if empirical:
                    # commonly used empirical furnishing weight equation
                    furnishing_wt_additional = scaler * PAX
                else:
                    # linear regression formula
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                # baseline furnishings (crew seats, cockpit, lavatories, galleys)
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0
        # Theoretically, we should make sure that furnishing_wt >= 0 here

        if smooth:
            furnishing_wt = smooth_max(furnishing_wt, 30.0, mu)
        else:
            furnishing_wt = np.maximum(furnishing_wt, 30.0)

        outputs[Aircraft.Furnishings.MASS] = furnishing_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
            dfurnishing_wt_dgross_wt_initial = 0.065
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dfus_len = 0.0
            dfurnishing_wt_dscaler = 0.0
            dfurnishing_wt_dacabin = 0.0
        else:
            if PAX >= 50:
                if empirical:
                    furnishing_wt_additional = scaler * PAX
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = PAX
                    dfurnishing_wt_additional_dacabin = 0.0
                else:
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = 0.0
                    dfurnishing_wt_additional_dacabin = 0.0
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
                dfurnishing_wt_dgross_wt_initial = dfurnishing_wt_additional_dgross_wt_initial + 0.0
                dfurnishing_wt_dcabin_width = (
                    dfurnishing_wt_additional_dcabin_width
                    + num_pilots * (1.0 / 12.0) * 90.0
                    + 1.5 * 0.75 * fus_len * np.pi / 2.0
                )
                dfurnishing_wt_dfus_len = (
                    dfurnishing_wt_additional_dfus_len + 1.5 * 0.75 * cabin_width * np.pi / 2.0
                )
                dfurnishing_wt_dscaler = dfurnishing_wt_additional_dscaler + 0.0
                dfurnishing_wt_dacabin = dfurnishing_wt_additional_dacabin + 0.5
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0

                dCPX_lin_dcabin_width = 10.516
                if smooth:
                    dCPX_dcabin_width = (
                        1 * dSigmoidXdx(CPX_lin / 28, 1, 0.01) * -dCPX_lin_dcabin_width
                        + dCPX_lin_dcabin_width
                        * CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + CPX_lin
                        * dSigmoidXdx(CPX_lin / 28, 1, 0.01)
                        / 28
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        * dCPX_lin_dcabin_width
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * dSigmoidXdx(CPX_lin / 62, 1, -0.01)
                        / 62
                        * -dCPX_lin_dcabin_width
                        + 1 * dSigmoidXdx(CPX_lin / 62, 1, 0.01) * dCPX_lin_dcabin_width
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0
                    if cabin_width > 8.90:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0

                dfurnishing_wt_dcabin_width = PAX * dCPX_dcabin_width
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        if smooth:
            sm_fac = d_smooth_max(furnishing_wt, 30.0, mu)
            dfurnishing_wt_dcabin_width = sm_fac * dfurnishing_wt_dcabin_width
            dfurnishing_wt_dgross_wt_initial = sm_fac * dfurnishing_wt_dgross_wt_initial
            dfurnishing_wt_dfus_len = sm_fac * dfurnishing_wt_dfus_len
            dfurnishing_wt_dscaler = sm_fac * dfurnishing_wt_dscaler
            dfurnishing_wt_dacabin = sm_fac * dfurnishing_wt_dacabin
        else:
            if furnishing_wt < 30.0:  # note: this technically creates a discontinuity
                dfurnishing_wt_dcabin_width = 0.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        partials[Aircraft.Furnishings.MASS, Mission.Design.GROSS_MASS] = (
            dfurnishing_wt_dgross_wt_initial
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.AVG_DIAMETER] = (
            dfurnishing_wt_dcabin_width / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.LENGTH] = (
            dfurnishing_wt_dfus_len / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] = (
            dfurnishing_wt_dscaler / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.CABIN_AREA] = (
            dfurnishing_wt_dacabin / GRAV_ENGLISH_LBM
        )


class BWBFurnishingMass(om.ExplicitComponent):
    """
    Computation of furnishing mass for BWB
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)
        add_aviary_option(self, Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Aircraft.Furnishings.USE_EMPIRICAL_EQUATION)
        self.options.declare('mu', default=1.0, types=float)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.HYDRAULIC_DIAMETER, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')

        self.add_output(Aircraft.Furnishings.MASS, units='lbm')

        self.declare_partials(
            Aircraft.Furnishings.MASS,
            [
                Aircraft.Fuselage.HYDRAULIC_DIAMETER,
                Mission.Design.GROSS_MASS,
                Aircraft.Furnishings.MASS_SCALER,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.CABIN_AREA,
            ],
        )

    def compute(self, inputs, outputs):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            # note: this technically creates a discontinuity
            # TODO: Doesn't occur in large single aisle
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
        else:
            if PAX >= 50:
                if empirical:
                    # commonly used empirical furnishing weight equation
                    furnishing_wt_additional = scaler * PAX
                else:
                    # linear regression formula
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                # baseline furnishings (crew seats, cockpit, lavatories, galleys)
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0

        if smooth:
            furnishing_wt = smooth_max(furnishing_wt, 30.0, mu)
        else:
            furnishing_wt = np.maximum(furnishing_wt, 30.0)

        outputs[Aircraft.Furnishings.MASS] = furnishing_wt / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, partials):
        PAX = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        smooth = self.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES]
        empirical = self.options[Aircraft.Furnishings.USE_EMPIRICAL_EQUATION]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        mu = self.options['mu']

        num_pilots = get_num_of_pilots(PAX, engine_type)
        num_flight_attendants = get_num_of_flight_attendent(PAX)
        lavatories = get_num_of_lavatories(PAX)

        gross_wt_initial = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fus_len = inputs[Aircraft.Fuselage.LENGTH]
        cabin_width = inputs[Aircraft.Fuselage.HYDRAULIC_DIAMETER]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        acabin = inputs[Aircraft.Fuselage.CABIN_AREA]

        if gross_wt_initial <= 10000.0:
            furnishing_wt = 0.065 * gross_wt_initial - 59.0
            dfurnishing_wt_dgross_wt_initial = 0.065
            dfurnishing_wt_dcabin_width = 0.0
            dfurnishing_wt_dfus_len = 0.0
            dfurnishing_wt_dscaler = 0.0
            dfurnishing_wt_dacabin = 0.0
        else:
            if PAX >= 50:
                if empirical:
                    furnishing_wt_additional = scaler * PAX
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = PAX
                    dfurnishing_wt_additional_dacabin = 0.0
                else:
                    furnishing_wt_additional = 118.4 * PAX - 4190.0
                    dfurnishing_wt_additional_dgross_wt_initial = 0.0
                    dfurnishing_wt_additional_dcabin_width = 0.0
                    dfurnishing_wt_additional_dfus_len = 0.0
                    dfurnishing_wt_additional_dscaler = 0.0
                    dfurnishing_wt_additional_dacabin = 0.0
                cabin_len = 0.75 * fus_len
                agalley = 0.50 * PAX
                furnishing_wt = (
                    furnishing_wt_additional
                    + num_pilots * (1.0 + cabin_width / 12.0) * 90.0
                    + num_flight_attendants * 30.0
                    + lavatories * 240.0
                    + agalley * 12.0
                    + 1.5 * cabin_len * cabin_width * np.pi / 2.0
                    + 0.5 * acabin
                )
                dfurnishing_wt_dgross_wt_initial = dfurnishing_wt_additional_dgross_wt_initial + 0.0
                dfurnishing_wt_dcabin_width = (
                    dfurnishing_wt_additional_dcabin_width
                    + num_pilots * (1.0 / 12.0) * 90.0
                    + 1.5 * 0.75 * fus_len * np.pi / 2.0
                )
                dfurnishing_wt_dfus_len = (
                    dfurnishing_wt_additional_dfus_len + 1.5 * 0.75 * cabin_width * np.pi / 2.0
                )
                dfurnishing_wt_dscaler = dfurnishing_wt_additional_dscaler + 0.0
                dfurnishing_wt_dacabin = dfurnishing_wt_additional_dacabin + 0.5
            else:
                CPX_lin = 28.0 + 10.516 * (cabin_width - 5.667)
                if smooth:
                    CPX = (
                        28 * sigmoidX(CPX_lin / 28, 1, -0.01)
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + 62 * sigmoidX(CPX_lin / 62, 1, 0.01)
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        CPX = 28.0
                    elif cabin_width > 8.90:  # note: this technically creates a discontinuity
                        CPX = 62.0
                furnishing_wt = CPX * PAX + 310.0

                dCPX_lin_dcabin_width = 10.516
                if smooth:
                    dCPX_dcabin_width = (
                        1 * dSigmoidXdx(CPX_lin / 28, 1, 0.01) * -dCPX_lin_dcabin_width
                        + dCPX_lin_dcabin_width
                        * CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        + CPX_lin
                        * dSigmoidXdx(CPX_lin / 28, 1, 0.01)
                        / 28
                        * sigmoidX(CPX_lin / 62, 1, -0.01)
                        * dCPX_lin_dcabin_width
                        + CPX_lin
                        * sigmoidX(CPX_lin / 28, 1, 0.01)
                        * dSigmoidXdx(CPX_lin / 62, 1, -0.01)
                        / 62
                        * -dCPX_lin_dcabin_width
                        + 1 * dSigmoidXdx(CPX_lin / 62, 1, 0.01) * dCPX_lin_dcabin_width
                    )
                else:
                    if cabin_width <= 5.667:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0
                    if cabin_width > 8.90:  # note: this technically creates a discontinuity
                        dCPX_dcabin_width = 0.0

                dfurnishing_wt_dcabin_width = PAX * dCPX_dcabin_width
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        if smooth:
            sm_fac = d_smooth_max(furnishing_wt, 30.0, mu)
            dfurnishing_wt_dcabin_width = sm_fac * dfurnishing_wt_dcabin_width
            dfurnishing_wt_dgross_wt_initial = sm_fac * dfurnishing_wt_dgross_wt_initial
            dfurnishing_wt_dfus_len = sm_fac * dfurnishing_wt_dfus_len
            dfurnishing_wt_dscaler = sm_fac * dfurnishing_wt_dscaler
            dfurnishing_wt_dacabin = sm_fac * dfurnishing_wt_dacabin
        else:
            if furnishing_wt < 30.0:  # note: this technically creates a discontinuity
                dfurnishing_wt_dcabin_width = 0.0
                dfurnishing_wt_dgross_wt_initial = 0.0
                dfurnishing_wt_dfus_len = 0.0
                dfurnishing_wt_dscaler = 0.0
                dfurnishing_wt_dacabin = 0.0

        partials[Aircraft.Furnishings.MASS, Mission.Design.GROSS_MASS] = (
            dfurnishing_wt_dgross_wt_initial
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.HYDRAULIC_DIAMETER] = (
            dfurnishing_wt_dcabin_width / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.LENGTH] = (
            dfurnishing_wt_dfus_len / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] = (
            dfurnishing_wt_dscaler / GRAV_ENGLISH_LBM
        )
        partials[Aircraft.Furnishings.MASS, Aircraft.Fuselage.CABIN_AREA] = (
            dfurnishing_wt_dacabin / GRAV_ENGLISH_LBM
        )
