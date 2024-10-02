import numpy as np
import openmdao.api as om
import os

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


class WingFuselageInterference_premission(om.ExplicitComponent):
    """
    This calculates an additional flat plate drag area due to general aerodynamic interference for wing-fuselage interference
    (based on results from Hoerner's drag)
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA)
        add_aviary_input(self, Aircraft.Wing.SPAN)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO)
        add_aviary_input(self, Aircraft.Wing.MOUNTING_TYPE)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP)
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER)
        add_aviary_input(self, Aircraft.Wing.CENTER_DISTANCE)

        self.add_output('interference_independent_of_shielded_area', 1.23456)
        self.add_output('drag_loss_due_to_shielded_wing_area', 1.23456)

    def setup_partials(self):
        self.declare_partials(
            'interference_independent_of_shielded_area', [
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.MOUNTING_TYPE,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.CENTER_DISTANCE])
        self.declare_partials(
            'drag_loss_due_to_shielded_wing_area', [
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.MOUNTING_TYPE,
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.CENTER_DISTANCE])

    def compute(self, inputs, outputs):
        SW, B, SLM, HWING, TCR, TCT, SWF, XWQLF = inputs.values()
        # from interference.f
        CROOT = 2*SW/(B*(1.0 + SLM))  # root chord # constant
        ZW_RF = 2.*HWING - 1.  # constant

        wtofd = TCR*CROOT/SWF  # wing_thickness_over_fuselage_diameter # constant
        # WIDTHFTOP = (SWF*(1.0 - (ZW_RF + wtofd)**2)**0.5) * \
        #     (sigX(ZW_RF+wtofd+1)-sigX(ZW_RF+wtofd-1))  # constant
        # WIDTHFBOT = (SWF*(1.0 - (ZW_RF - wtofd)**2)**0.5) * \
        #     (sigX(ZW_RF-wtofd+1)-sigX(ZW_RF-wtofd-1))  # constant

        if (abs(ZW_RF + wtofd) >= 1.0):
            WIDTHFTOP = 0.0
        else:
            WIDTHFTOP = SWF*(1.0 - (ZW_RF + wtofd)**2)**0.5
            # -.5*(ZW_RF + wtofd)*(1.0-(ZW_RF + wtofd)**2)**-0.5
        if (abs(ZW_RF - wtofd) >= 1.0):
            WIDTHFBOT = 0.0
        else:
            WIDTHFBOT = SWF*(1.0 - (ZW_RF - wtofd)**2)**0.5

        WBODYWF = 0.5*(WIDTHFTOP + WIDTHFBOT)  # constant
        TCBODYWF = TCR - WBODYWF/B*(TCR - TCT)  # constant
        CBODYWF = CROOT*(1.0 - WBODYWF/B*(1.0 - SLM))  # constant

        KVWF = .0194 - 0.14817*ZW_RF + 1.3515*ZW_RF**2  # constant
        KLWF = 0.13077 + 1.9791*XWQLF + 3.3325*XWQLF**2 - \
            10.095*XWQLF**3 + 4.7229*XWQLF**4  # constant
        KDTWF = 0.73543 + .028571*SWF/(TCBODYWF*CBODYWF)  # constant
        FEINTWF = 1.5*(TCBODYWF**3)*(CBODYWF**2)*KVWF*KLWF*KDTWF  # constant
        AREASHIELDWF = 0.5*(CROOT + CBODYWF)*WBODYWF  # constant

        # interference drag independent of shielded area
        outputs['interference_independent_of_shielded_area'] = FEINTWF
        # the loss in drag due to the shielded wing area
        outputs['drag_loss_due_to_shielded_wing_area'] = AREASHIELDWF

    def compute_partials(self, inputs, J):
        SW, B, SLM, HWING, TCR, TCT, SWF, XWQLF = inputs.values()

        CROOT = 2*SW/(B*(1 + SLM))
        dCROOT_dSW = 2/(B*(1 + SLM))
        dCROOT_dB = -2*SW/(B**2*(1 + SLM))
        dCROOT_dSLM = -2*SW/(B*(1 + SLM)**2)
        ZW_RF = 2*HWING - 1

        wtofd = TCR*CROOT/SWF  # wing_thickness_over_fuselage_diameter
        dwtofd_dTCR = CROOT/SWF
        dwtofd_dCROOT = TCR/SWF
        dwtofd_dSWF = -TCR*CROOT/SWF**2
        if (abs(ZW_RF + wtofd) >= 1):
            WIDTHFTOP = 0.0
            dWIDTHFTOP_dSWF = 0
            dWIDTHFTOP_dHWING = 0
            dWIDTHFTOP_dTCR = 0
            dWIDTHFTOP_dSW = 0
            dWIDTHFTOP_dB = 0
            dWIDTHFTOP_dSLM = 0
        else:
            WIDTHFTOP = SWF*(1.0 - (2*HWING - 1 + wtofd)**2)**0.5
            dWIDTHFTOP_dSWF = (1.0 - (2*HWING - 1 + wtofd)**2)**0.5 + .5*SWF*(1.0 -
                                                                              (2*HWING - 1 + wtofd)**2)**-0.5 * (-2*(2*HWING - 1 + wtofd)*dwtofd_dSWF)
            dWIDTHFTOP_dHWING = .5*SWF * \
                (1.0 - (2*HWING - 1 + wtofd)**2)**-0.5 * (-2*(2*HWING - 1 + wtofd)*2)
            dWIDTHFTOP_dTCR = .5*SWF*(1.0 - (2*HWING - 1 + wtofd)
                                      ** 2)**-0.5 * (-2*(2*HWING - 1 + wtofd)*dwtofd_dTCR)
            dWIDTHFTOP_dSW = .5*SWF*(1.0 - (2*HWING - 1 + wtofd)**2)**- \
                0.5 * (-2*(2*HWING - 1 + wtofd)*dwtofd_dCROOT*dCROOT_dSW)
            dWIDTHFTOP_dB = .5*SWF*(1.0 - (2*HWING - 1 + wtofd)**2)**- \
                0.5 * (-2*(2*HWING - 1 + wtofd)*dwtofd_dCROOT*dCROOT_dB)
            dWIDTHFTOP_dSLM = .5*SWF*(1.0 - (2*HWING - 1 + wtofd)**2)**- \
                0.5 * (-2*(2*HWING - 1 + wtofd)*dwtofd_dCROOT*dCROOT_dSLM)
            # -.5*(ZW_RF + wtofd)*(1.0-(ZW_RF + wtofd)**2)**-0.5
        if (abs(ZW_RF - wtofd) >= 1):
            WIDTHFBOT = 0.0
        else:
            WIDTHFBOT = SWF*(1.0 - (2*HWING - 1 - wtofd)**2)**0.5
            dWIDTHFBOT_dSWF = (1.0 - (2*HWING - 1 - wtofd)**2)**0.5 + .5*SWF*(1.0 -
                                                                              (2*HWING - 1 - wtofd)**2)**-0.5 * (-2*(2*HWING - 1 - wtofd)*-dwtofd_dSWF)
            dWIDTHFBOT_dHWING = .5*SWF * \
                (1.0 - (2*HWING - 1 - wtofd)**2)**-0.5 * (-2*(2*HWING - 1 - wtofd)*2)
            dWIDTHFBOT_dTCR = .5*SWF*(1.0 - (2*HWING - 1 - wtofd)
                                      ** 2)**-0.5 * (-2*(2*HWING - 1 - wtofd)*-dwtofd_dTCR)
            dWIDTHFBOT_dSW = .5*SWF*(1.0 - (2*HWING - 1 - wtofd)**2)**- \
                0.5 * (-2*(2*HWING - 1 - wtofd)*-dwtofd_dCROOT*dCROOT_dSW)
            dWIDTHFBOT_dB = .5*SWF*(1.0 - (2*HWING - 1 - wtofd)**2)**- \
                0.5 * (-2*(2*HWING - 1 - wtofd)*-dwtofd_dCROOT*dCROOT_dB)
            dWIDTHFBOT_dSLM = .5*SWF*(1.0 - (2*HWING - 1 - wtofd)**2)**- \
                0.5 * (-2*(2*HWING - 1 - wtofd)*-dwtofd_dCROOT*dCROOT_dSLM)

        TCBODYWF = TCR - 0.5*(WIDTHFTOP + WIDTHFBOT)/B*(TCR - TCT)
        CBODYWF = CROOT*(1 - 0.5*(WIDTHFTOP + WIDTHFBOT)/B*(1 - SLM))

        KVWF = .0194 - 0.14817*(2*HWING - 1) + 1.3515*(2*HWING - 1)**2
        KLWF = 0.13077 + 1.9791*XWQLF + 3.3325*XWQLF**2 - \
            10.095*XWQLF**3 + 4.7229*XWQLF**4
        KDTWF = 0.73543 + .028571*SWF/(TCBODYWF*CBODYWF)
        FEINTWF = 1.5*(TCBODYWF**3)*(CBODYWF**2)*KVWF*KLWF*KDTWF
        AREASHIELDWF = 0.5*(CROOT + CBODYWF)*0.5*(WIDTHFTOP + WIDTHFBOT)


class WingFuselageInterference_dynamic(om.ExplicitComponent):
    """
    This calculates an additional flat plate drag area due to general aerodynamic interference for wing-fuselage interference
    (based on results from Hoerner's drag)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        add_aviary_input(self, Aircraft.Wing.AREA)
        add_aviary_input(self, Aircraft.Wing.FORM_FACTOR, 1.25)
        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD)
        add_aviary_input(self, Dynamic.Mission.MACH, shape=nn)  # np.ones(nn))
        add_aviary_input(self, Dynamic.Mission.TEMPERATURE, shape=nn)  # np.ones(nn))
        add_aviary_input(self, Dynamic.Mission.KINEMATIC_VISCOSITY,
                         shape=nn)  # np.ones(nn))
        self.add_input('interference_independent_of_shielded_area')
        self.add_input('drag_loss_due_to_shielded_wing_area')

        add_aviary_output(
            self, Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, np.full(nn, 1.23456))

    def setup_partials(self):
        nn = self.options["num_nodes"]
        # arange = np.arange(nn)
        # self.declare_partials(
        #     Aircraft.Wing.FORM_FACTOR, [
        #         Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED,
        #         Mission.Design.MACH,
        #         Aircraft.Wing.SWEEP])  # , rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        SW, CKW, CBARW, EM, T0, XKV, AREASHIELDWF, FEINTWF = inputs.values()

        # from gaspmain.f
        # reli = reynolds number per foot
        RELI = np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32) * EM * np.sqrt(T0)/XKV  # dynamic

        # from aero.f
        # CFIN CALCULATION FROM SCHLICHTING PG. 635-665
        CFIN = 0.455/np.log10(10_000_000.)**2.58/(1. + 0.144*EM**2)**0.65  # dynamic
        CDWI = FCFWC*FCFWT*CFIN  # dynamic
        FEW = SW * CDWI * CKW * ((np.log10(RELI * CBARW)/7.)**(-2.6))  # dynamic
        # from interference.f
        FEIWF = FEINTWF - 1*FEW/SW*AREASHIELDWF  # dynamic

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


class WingFuselageInterference(om.Group):
    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        if os.environ['TESTFLO_RUNNING']:
            self.add_subsystem("static_calculations",
                               WingFuselageInterference_premission(), promotes=["*"])
        self.add_subsystem("dynamic_calculations", WingFuselageInterference_dynamic(
            num_nodes=nn), promotes=["*"])
