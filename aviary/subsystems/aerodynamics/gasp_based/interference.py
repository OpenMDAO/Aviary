import numpy as np
import openmdao.api as om
from openmdao.utils import cs_safe as cs
import os

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Mission
from aviary.constants import GRAV_ENGLISH_GASP

FCFWC = 1
FCFWT = 1


class RootChord(om.ExplicitComponent):
    """Calculates the wing root chord"""

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA)
        add_aviary_input(self, Aircraft.Wing.SPAN)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO)

        self.add_output('CROOT', 1.23456)

    def compute(self, inputs, outputs):
        SW, B, SLM = inputs.values()

        outputs['CROOT'] = 2*SW/(B*(1.0 + SLM))

    def setup_partials(self):
        self.declare_partials(
            'CROOT', [
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
            ])

    def compute_partials(self, inputs, J):
        SW, B, SLM = inputs.values()

        J['CROOT', Aircraft.Wing.AREA] = 2/(B*(1.0 + SLM))
        J['CROOT', Aircraft.Wing.SPAN] = -2*SW/(B**2*(1.0 + SLM))
        J['CROOT', Aircraft.Wing.TAPER_RATIO] = -2*SW/(B*(1.0 + SLM)**2)


class CommonVariables(om.ExplicitComponent):
    """
    Calculates the wing_thickness_over_fuselage_diameter and an
    intermediate reference variable that maps the wing attachment
    location from -1 (bottom of fuselage) to +1 (top of fuselage)
    """

    def setup(self):
        self.add_input('CROOT')
        add_aviary_input(self, Aircraft.Wing.MOUNTING_TYPE)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT)
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER)

        self.add_output('ZW_RF', 1.23456)
        self.add_output('wtofd', 1.23456)

    def compute(self, inputs, outputs):
        CROOT, HWING, TCR, SWF = inputs.values()

        outputs['ZW_RF'] = 2*HWING - 1
        outputs['wtofd'] = TCR*CROOT/SWF  # wing_thickness_over_fuselage_diameter

    def setup_partials(self):
        self.declare_partials('ZW_RF', [Aircraft.Wing.MOUNTING_TYPE], val=2)
        self.declare_partials(
            'wtofd', [
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                'CROOT',
                Aircraft.Fuselage.AVG_DIAMETER,
            ])

    def compute_partials(self, inputs, J):
        CROOT, HWING, TCR, SWF = inputs.values()

        J['wtofd', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = CROOT/SWF
        J['wtofd', 'CROOT'] = TCR/SWF
        J['wtofd', Aircraft.Fuselage.AVG_DIAMETER] = -TCR*CROOT/SWF**2


class TopAndBottomWidth(om.ExplicitComponent):
    """Calculates the fuselage width at the top and bottom"""

    def setup(self):
        self.add_input('ZW_RF')
        self.add_input('wtofd')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER)

        self.add_output('WBODYWF', 1.23456)

    def compute(self, inputs, outputs):
        ZW_RF, wtofd, SWF = inputs.values()

        if (cs.abs(ZW_RF + wtofd) >= 1.0):
            WIDTHFTOP = 0.0
        else:
            WIDTHFTOP = SWF*(1.0 - (ZW_RF + wtofd)**2)**0.5
        if (cs.abs(ZW_RF - wtofd) >= 1.0):
            WIDTHFBOT = 0.0
        else:
            WIDTHFBOT = SWF*(1.0 - (ZW_RF - wtofd)**2)**0.5

        outputs['WBODYWF'] = 0.5*(WIDTHFTOP + WIDTHFBOT)

    def setup_partials(self):
        self.declare_partials(
            'WBODYWF', [
                'ZW_RF',
                'wtofd',
                Aircraft.Fuselage.AVG_DIAMETER,
            ])

    def compute_partials(self, inputs, J):
        ZW_RF, wtofd, SWF = inputs.values()

        if (cs.abs(ZW_RF + wtofd) >= 1.0):
            dTOP_dZWRF = 0
            dTOP_dwtofd = 0
            dTOP_dSWF = 0
        else:
            dTOP_dZWRF = -SWF*(1.0 - (ZW_RF + wtofd)**2)**-0.5 * (ZW_RF + wtofd)
            dTOP_dwtofd = -SWF*(1.0 - (ZW_RF + wtofd)**2)**-0.5 * (ZW_RF + wtofd)
            dTOP_dSWF = (1.0 - (ZW_RF + wtofd)**2)**0.5
        if (cs.abs(ZW_RF - wtofd) >= 1.0):
            dBOT_dZWRF = 0
            dBOT_dwtofd = 0
            dBOT_dSWF = 0
        else:
            dBOT_dZWRF = -SWF*(1.0 - (ZW_RF - wtofd)**2)**-0.5 * (ZW_RF - wtofd)
            dBOT_dwtofd = SWF*(1.0 - (ZW_RF - wtofd)**2)**-0.5 * (ZW_RF - wtofd)
            dBOT_dSWF = (1.0 - (ZW_RF - wtofd)**2)**0.5

        J['WBODYWF', 'ZW_RF'] = 0.5*(dTOP_dZWRF + dBOT_dZWRF)
        J['WBODYWF', 'wtofd'] = 0.5*(dTOP_dwtofd + dBOT_dwtofd)
        J['WBODYWF', Aircraft.Fuselage.AVG_DIAMETER] = 0.5*(dTOP_dSWF + dBOT_dSWF)


class BodyRatios(om.ExplicitComponent):
    """Calculates some intermediate variables that are based on unitless ratios"""

    def setup(self):
        self.add_input('WBODYWF')
        self.add_input('CROOT')
        add_aviary_input(self, Aircraft.Wing.SPAN)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_ROOT)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD_TIP)

        self.add_output('TCBODYWF', 1.23456)
        self.add_output('CBODYWF', 1.23456)

    def compute(self, inputs, outputs):
        WBODYWF, CROOT, B, SLM, TCR, TCT = inputs.values()

        outputs['TCBODYWF'] = TCR - WBODYWF/B*(TCR - TCT)
        outputs['CBODYWF'] = CROOT*(1.0 - WBODYWF/B*(1.0 - SLM))

    def setup_partials(self):
        self.declare_partials(
            'TCBODYWF', [
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
                'WBODYWF',
                Aircraft.Wing.SPAN,
                Aircraft.Wing.THICKNESS_TO_CHORD_TIP,
            ])
        self.declare_partials(
            'CBODYWF', [
                'CROOT',
                'WBODYWF',
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
            ])

    def compute_partials(self, inputs, J):
        WBODYWF, CROOT, B, SLM, TCR, TCT = inputs.values()

        J['TCBODYWF', Aircraft.Wing.THICKNESS_TO_CHORD_ROOT] = 1 - WBODYWF/B
        J['TCBODYWF', 'WBODYWF'] = -1/B*(TCR - TCT)
        J['TCBODYWF', Aircraft.Wing.SPAN] = WBODYWF/B**2*(TCR - TCT)
        J['TCBODYWF', Aircraft.Wing.THICKNESS_TO_CHORD_TIP] = WBODYWF/B

        J['CBODYWF', 'CROOT'] = (1.0 - WBODYWF/B*(1.0 - SLM))
        J['CBODYWF', 'WBODYWF'] = CROOT*(-1/B*(1.0 - SLM))
        J['CBODYWF', Aircraft.Wing.SPAN] = CROOT*(WBODYWF/B**2*(1.0 - SLM))
        J['CBODYWF', Aircraft.Wing.TAPER_RATIO] = CROOT*(WBODYWF/B)


class InterferenceDrag(om.ExplicitComponent):
    """
    Calculates the interference_independent_of_shielded_area and
    drag_loss_due_to_shielded_wing_area
    """

    def setup(self):
        self.add_input('WBODYWF')
        self.add_input('CROOT')
        self.add_input('TCBODYWF')
        self.add_input('CBODYWF')
        self.add_input('ZW_RF')
        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER)
        add_aviary_input(self, Aircraft.Wing.CENTER_DISTANCE)

        self.add_output('interference_independent_of_shielded_area', 1.23456)
        self.add_output('drag_loss_due_to_shielded_wing_area', 1.23456)

    def compute(self, inputs, outputs):
        WBODYWF, CROOT, TCBODYWF, CBODYWF, ZW_RF, SWF, XWQLF = inputs.values()

        # factor due to vertical location
        KVWF = .0194 - 0.14817*ZW_RF + 1.3515*ZW_RF**2
        # factor due to longitudinal location
        KLWF = 0.13077 + 1.9791*XWQLF + 3.3325*XWQLF**2 - \
            10.095*XWQLF**3 + 4.7229*XWQLF**4
        # factor due to fuselage diameter / thickness
        KDTWF = 0.73543 + .028571*SWF/(TCBODYWF*CBODYWF)

        FEINTWF = 1.5*(TCBODYWF**3)*(CBODYWF**2)*KVWF*KLWF*KDTWF
        AREASHIELDWF = 0.5*(CROOT + CBODYWF)*WBODYWF

        # interference drag independent of shielded area
        outputs['interference_independent_of_shielded_area'] = FEINTWF
        # the loss in drag due to the shielded wing area
        outputs['drag_loss_due_to_shielded_wing_area'] = AREASHIELDWF

    def setup_partials(self):
        self.declare_partials(
            'interference_independent_of_shielded_area', [
                'TCBODYWF',
                'CBODYWF',
                'ZW_RF',
                Aircraft.Fuselage.AVG_DIAMETER,
                Aircraft.Wing.CENTER_DISTANCE,
            ])
        self.declare_partials(
            'drag_loss_due_to_shielded_wing_area', [
                'CROOT',
                'CBODYWF',
                'WBODYWF',
            ])

    def compute_partials(self, inputs, J):
        WBODYWF, CROOT, TCBODYWF, CBODYWF, ZW_RF, SWF, XWQLF = inputs.values()

        KVWF = .0194 - 0.14817*ZW_RF + 1.3515*ZW_RF**2
        dKVWF_dZWRF = -0.14817 + 2*1.3515*ZW_RF
        KLWF = 0.13077 + 1.9791*XWQLF + 3.3325*XWQLF**2 - \
            10.095*XWQLF**3 + 4.7229*XWQLF**4
        dKLWF_dXWQLF = 1.9791 + 2*3.3325*XWQLF - 3*10.095*XWQLF**2 + 4*4.7229*XWQLF**3
        KDTWF = 0.73543 + .028571*SWF/(TCBODYWF*CBODYWF)
        dKDTWF_dSWF = .028571/(TCBODYWF*CBODYWF)
        # dKDTWF_dTCBODYWF = -.028571*SWF/(TCBODYWF*CBODYWF)**2
        # dKDTWF_dCBODYWF = -.028571*SWF/(TCBODYWF*CBODYWF)**2

        J['interference_independent_of_shielded_area', 'TCBODYWF'] = \
            1.5*(TCBODYWF)*(CBODYWF**2)*KVWF*KLWF * \
            (3*0.73543*TCBODYWF + 2*.028571*SWF/CBODYWF)
        J['interference_independent_of_shielded_area', 'CBODYWF'] = \
            1.5*(TCBODYWF**3)*KVWF*KLWF*(2*0.73543*CBODYWF + .028571*SWF/TCBODYWF)
        J['interference_independent_of_shielded_area', 'ZW_RF'] = \
            1.5*(TCBODYWF**3)*(CBODYWF**2)*dKVWF_dZWRF*KLWF*KDTWF
        J['interference_independent_of_shielded_area', Aircraft.Wing.CENTER_DISTANCE] = \
            1.5*(TCBODYWF**3)*(CBODYWF**2)*KVWF*dKLWF_dXWQLF*KDTWF
        J['interference_independent_of_shielded_area', Aircraft.Fuselage.AVG_DIAMETER] = \
            1.5*(TCBODYWF**3)*(CBODYWF**2)*KVWF*KLWF*dKDTWF_dSWF

        J['drag_loss_due_to_shielded_wing_area', 'CROOT'] = 0.5*WBODYWF
        J['drag_loss_due_to_shielded_wing_area', 'CBODYWF'] = 0.5*WBODYWF
        J['drag_loss_due_to_shielded_wing_area', 'WBODYWF'] = 0.5*(CROOT + CBODYWF)


class WingFuselageInterferencePremission(om.Group):
    """
    Calculates the interference_independent_of_shielded_area and
    drag_loss_due_to_shielded_wing_area from static geometry parameters
    that will be fed to the dynamic portion during the mission.
    Separating out some of the basic quantities that are used in multiple places
    allowed for simplification of derivatives.
    """

    def setup(self):
        self.add_subsystem('root_chord', RootChord(),
                           promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('common_variables', CommonVariables(),
                           promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('top_and_bottom_width', TopAndBottomWidth(),
                           promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('body_ratios', BodyRatios(),
                           promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('interference_drag', InterferenceDrag(),
                           promotes_inputs=['*'], promotes_outputs=['*'])


class WingFuselageInterferenceMission(om.ExplicitComponent):
    """
    This calculates an additional flat plate drag area due to general aerodynamic interference for wing-fuselage interference
    (based on results from Hoerner's drag)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        add_aviary_input(self, Aircraft.Wing.FORM_FACTOR, 1.25)
        add_aviary_input(self, Aircraft.Wing.AVERAGE_CHORD)
        add_aviary_input(self, Dynamic.Mission.MACH, shape=nn)
        add_aviary_input(self, Dynamic.Mission.TEMPERATURE, shape=nn)
        add_aviary_input(self, Dynamic.Mission.KINEMATIC_VISCOSITY,
                         shape=nn)
        self.add_input('interference_independent_of_shielded_area')
        self.add_input('drag_loss_due_to_shielded_wing_area')

        self.add_output('wing_fuselage_interference_flat_plate_equivalent',
                        np.full(nn, 1.23456))

    def setup_partials(self):
        nn = self.options["num_nodes"]
        arange = np.arange(nn)
        self.declare_partials(
            'wing_fuselage_interference_flat_plate_equivalent', [
                Dynamic.Mission.MACH,
                Dynamic.Mission.TEMPERATURE,
                Dynamic.Mission.KINEMATIC_VISCOSITY],
            rows=arange, cols=arange)
        self.declare_partials(
            'wing_fuselage_interference_flat_plate_equivalent', [
                Aircraft.Wing.FORM_FACTOR,
                Aircraft.Wing.AVERAGE_CHORD,
                'interference_independent_of_shielded_area'],
            rows=arange, cols=np.zeros(nn))
        self.declare_partials('wing_fuselage_interference_flat_plate_equivalent', [
                              'drag_loss_due_to_shielded_wing_area'], val=1)

    def compute(self, inputs, outputs):
        CKW, CBARW, EM, T0, XKV, AREASHIELDWF, FEINTWF = inputs.values()

        # from gaspmain.f
        # reli = reynolds number per foot
        RELI = np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32) * EM * np.sqrt(T0)/XKV  # dynamic

        # from aero.f
        # CFIN CALCULATION FROM SCHLICHTING PG. 635-665
        # log10(10_000_000.) = 7
        CFIN = 0.455/7**2.58/(1. + 0.144*EM**2)**0.65  # dynamic
        CDWI = FCFWC*FCFWT*CFIN  # dynamic
        FEW_over_SW = CDWI * CKW * ((np.log10(RELI * CBARW)/7.)**(-2.6))  # dynamic
        # from interference.f
        FEIWF = FEINTWF - FEW_over_SW*AREASHIELDWF  # dynamic

        outputs['wing_fuselage_interference_flat_plate_equivalent'] = FEIWF

    def compute_partials(self, inputs, J):
        CKW, CBARW, EM, T0, XKV, AREASHIELDWF, FEINTWF = inputs.values()

        RELI = np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32) * EM * np.sqrt(T0)/XKV
        dRELI_dEM = np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32) * np.sqrt(T0)/XKV

        CFIN = 0.455/7**2.58/(1. + 0.144*EM**2)**0.65
        dCFIN_dEM = -.65*CFIN/(1. + 0.144*EM**2)*.288*EM
        CDWI = FCFWC*FCFWT*CFIN

        J['wing_fuselage_interference_flat_plate_equivalent', Aircraft.Wing.FORM_FACTOR] = \
            -CDWI * ((np.log10(RELI * CBARW)/7.)**(-2.6))*AREASHIELDWF
        J['wing_fuselage_interference_flat_plate_equivalent', Aircraft.Wing.AVERAGE_CHORD] = \
            2.6*CDWI * CKW * ((np.log10(RELI * CBARW)/7.)**(-3.6))*AREASHIELDWF \
            * 1/(np.log(10)*(CBARW)*7)
        J['wing_fuselage_interference_flat_plate_equivalent', Dynamic.Mission.MACH] = -CKW * AREASHIELDWF * (((np.log10(RELI * CBARW)/7.)**(-2.6)) * (
            FCFWC*FCFWT * dCFIN_dEM) + CFIN*(-2.6*((np.log10(RELI * CBARW)/7.)**(-3.6)) / (np.log(10)*(RELI)*7)*(dRELI_dEM)))
        J['wing_fuselage_interference_flat_plate_equivalent', Dynamic.Mission.TEMPERATURE] = \
            -CDWI * CKW * -2.6*((np.log10(RELI * CBARW)/7.)**(-3.6))*AREASHIELDWF \
            * 1/(np.log(10)*(RELI)*7) * np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32) \
            * EM * .5/(XKV*np.sqrt(T0))
        J['wing_fuselage_interference_flat_plate_equivalent', Dynamic.Mission.KINEMATIC_VISCOSITY] = \
            CDWI * CKW * -2.6*((np.log10(RELI * CBARW)/7.)**(-3.6))*AREASHIELDWF \
            * 1/(np.log(10)*(RELI)*7) * np.sqrt(1.4*GRAV_ENGLISH_GASP*53.32) \
            * EM * np.sqrt(T0) / XKV**2
        J['wing_fuselage_interference_flat_plate_equivalent',
            'interference_independent_of_shielded_area'] = \
            -CDWI * CKW * ((np.log10(RELI * CBARW)/7.)**(-2.6))
