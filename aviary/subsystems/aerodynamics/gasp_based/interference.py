import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.constants import GRAV_ENGLISH_GASP


def sigX(x):
    sig = 1 / (1 + np.exp(-x))

    return sig


def dSigXdX(x):
    derivative = -1 / (1 + np.exp(-x)) ** 2 * (-1 * np.exp(-x))

    return derivative


FCFWC = 1
FCFWT = 1


class WingFuselageInterference(om.ExplicitComponent):
    """
    This calculates an additional flat plate drag area due to general aerodynamic interference for wing-fuselage interference
    (based on results from Hoerner's drag)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        add_aviary_input(self, Aircraft.Wing.AREA)
        add_aviary_input(self, Aircraft.Wing.FORM_FACTOR)
        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD)
        add_aviary_input(self, Aircraft.Wing.SPAN)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO)
        add_aviary_input(self, Aircraft.Wing.MOUNTING_TYPE)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP)
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER)
        add_aviary_input(self, Aircraft.Wing.CENTER_DISTANCE)
        add_aviary_input(self, Dynamic.Mission.MACH)
        add_aviary_input(self, Dynamic.Mission.TEMPERATURE)
        add_aviary_input(self, Dynamic.Mission.KINEMATIC_VISCOSITY)

        add_aviary_output(
            self, Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, np.full(nn, 1.23456))

    def setup_partials(self):
        nn = self.options["num_nodes"]
        # arange = np.arange(nn)
        self.declare_partials(
            Aircraft.Wing.FORM_FACTOR, [
                Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
                Mission.Design.MACH,
                Aircraft.Wing.SWEEP])  # , rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        SW, CKW, CBARW, B, SLM, HWING, TCR, TCT, SWF, XWQLF, EM, T0, XKV = inputs.values()

        # reli = reynolds number per foot
        # reli and fmach are dynamic (calculated at every point during the trajectory)

        # from gaspmain.f
        RELI = EM * np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32*T0)/XKV  # dynamic

        # from aero.f
        # CFIN CALCULATION FROM SCHLICHTING PG. 635-665
        ALBAS = np.log10(10_000_000.)**2.58  # constant
        FMACH = (1. + 0.144*EM**2)**0.65  # dynamic

        CFIN = 0.455/ALBAS/FMACH  # dynamic
        CDWI = FCFWC*FCFWT*CFIN  # dynamic
        FEW = SW * CDWI * CKW * ((np.log10(RELI * CBARW)/7.)**(-2.6))  # dynamic

        # from interference.f
        CROOT = 2*SW/(B*(1.0 + SLM))  # root chord # constant
        CDW0 = FEW/SW  # dynamic
        ZW_RF = 2.*HWING - 1.  # constant

        wtofd = TCR*CROOT/SWF  # wing_thickness_over_fuselage_diameter # constant
        WIDTHFTOP = (SWF*(1.0 - (ZW_RF + wtofd)**2)**0.5) * \
            (sigX(ZW_RF+wtofd+1)-sigX(ZW_RF+wtofd-1))  # constant
        WIDTHFBOT = (SWF*(1.0 - (ZW_RF - wtofd)**2)**0.5) * \
            (sigX(ZW_RF-wtofd+1)-sigX(ZW_RF-wtofd-1))  # constant

        WBODYWF = 0.5*(WIDTHFTOP + WIDTHFBOT)  # constant
        TCBODYWF = TCR - WBODYWF/B*(TCR - TCT)  # constant
        CBODYWF = CROOT*(1.0 - WBODYWF/B*(1.0 - SLM))  # constant

        KVWF = .0194 - 0.14817*ZW_RF + 1.3515*ZW_RF**2  # constant
        KLWF = 0.13077 + 1.9791*XWQLF + 3.3325*XWQLF**2 - \
            10.095*XWQLF**3 + 4.7229*XWQLF**4  # constant
        KDTWF = 0.73543 + .028571*SWF/(TCBODYWF*CBODYWF)  # constant
        FEINTWF = 1.5*(TCBODYWF**3)*(CBODYWF**2)*KVWF*KLWF*KDTWF  # constant
        AREASHIELDWF = 0.5*(CROOT + CBODYWF)*WBODYWF  # constant
        FESHIELDWF = 1*CDW0*AREASHIELDWF  # dynamic
        FEIWF = FEINTWF - FESHIELDWF  # dynamic

        outputs[Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR] = FEIWF

    def compute_partials(self, inputs, J):
        pass
        #  = inputs.values()

        # cos1 = np.cos(rlmc4)

    #    SUBROUTINE INTERFERENCE(FEIWF)

    #    IF (ABS(ZW_RF + TCR*CROOT/SWF).GE.1.0) THEN
    #       WIDTHFTOP = 0.0
    #    ELSE
    #       WIDTHFTOP = SWF*(1.0 - (ZW_RF + TCR*CROOT/SWF)**2)**0.5
    #    IF (ABS(ZW_RF - TCR*CROOT/SWF).GE.1.0) THEN
    #       WIDTHFBOT = 0.0
    #    ELSE
    #       WIDTHFBOT = SWF*(1.0 - (ZW_RF - TCR*CROOT/SWF)**2)**0.5
