"""
OpenMDAO component to compute the design mach number and design coefficient of lift,
based on the calculations used in FLOPS and LEAPS 1.0.
"""
import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class Design(om.ExplicitComponent):
    """
    Calculates the design mach number and coefficient of lift.

    Based on subroutines MDESN and CLDESN in FLOPS.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sub_sonic_coeff = [0.029, 0.1843, 57.2958, 1.0 / 10.0]
        self.super_sonic_coeff = [-0.06416, 0.530389, -0.214493, 0.0376684, 1.0 / 3.0]
        self.des_mach_coeff = [0.32, 57.2958, 0.144]

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        # Aircraft design inputs
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, 0.0)
        add_aviary_input(self, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.0)
        add_aviary_input(self, Aircraft.Wing.SWEEP, 0.0)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, 0.0)

        # Declare outputs
        add_aviary_output(self, Mission.Design.MACH, 0.0)
        add_aviary_output(self, Mission.Design.LIFT_COEFFICIENT, 0.0)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        AITEK = aviary_options.get_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY)
        VMAX = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        AR, CAM, SW25, TC = inputs.values()

        if TC.real > 0.065:  # subsonic
            a, b, c, d = self.sub_sonic_coeff
            CLDES = (a + b*AR) * np.cos(SW25/c) * (1.0 + d*CAM) / np.sqrt(AR)

        else:  # supersonic
            a, b, c, d, e = self.super_sonic_coeff
            FAR = AR * TC**e
            CLDES = a + b*FAR + c*FAR**2.0 + d*FAR**3.0

        outputs[Mission.Design.LIFT_COEFFICIENT] = CLDES

        if TC.real > 0.065 or VMAX < 1.0:  # subsonic
            TC23 = TC ** (2.0 / 3.0)

            x = np.array([CLDES[0], TC23[0]], dtype=CLDES.dtype)
            ANS1 = CMDEStable.interpolate(x)
            ANS2 = AMDEStable.interpolate(x)

            ANS = ANS1 * (2.0 - AITEK) + ANS2 * (AITEK - 1.0)
            DESM2D = np.sqrt(ANS + 1.0)

        else:  # supersonic
            x = np.array([CLDES[0], TC[0]])
            DESM2D, self.dDESM2D = HSMDEStable.interpolate(x, compute_derivative=True)

        # Calculate value of design Mach number
        a, b, c = self.des_mach_coeff

        DMDSWP = a * (1.0 - np.cos(SW25 / b))
        DMDAR = c / AR

        # Design Mach number
        outputs[Mission.Design.MACH] = DESM2D + DMDSWP + DMDAR

    def compute_partials(self, inputs, partials):
        aviary_options: AviaryValues = self.options['aviary_options']
        AITEK = aviary_options.get_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY)
        VMAX = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        AR, CAM, SW25, TC = inputs.values()

        if TC.real > 0.065:  # subsonic
            a, b, c, d = self.sub_sonic_coeff
            fact1 = a + b * AR
            fact2 = np.cos(SW25/c)
            fact3 = (1.0 + d*CAM)
            fact4 = 1.0 / np.sqrt(AR)
            CLDES = fact1 * fact2 * fact3 * fact4

            # Calculate derivative of design CL wrt Mach, CL, AR, TC, CAM, and SW25
            dCLDES_dAR = fact3 * fact2 * (b * fact4 - fact1 / (2*AR**1.5))
            dCLDES_dTC = 0.0
            dCLDES_dCAM = d * fact1 * fact2 * fact4
            dCLDES_dSW25 = -fact1 * fact3 * np.sin(SW25/c) * fact4/c

        else:  # supersonic
            a, b, c, d, e = self.super_sonic_coeff
            FAR = AR * TC**e
            CLDES = a + b*FAR + c*FAR**2.0 + d*FAR**3.0

            # Calculate derivative of design CL wrt AR, TC, SW25, and CAM
            dFAR_dAR = TC**e
            dFAR_dTC = e * AR * TC**(e - 1.0)
            dCLDES_dFAR = b + 2.0*c*FAR + 3.0*d*FAR**2.0
            dCLDES_dAR = dCLDES_dFAR * dFAR_dAR
            dCLDES_dTC = dCLDES_dFAR * dFAR_dTC
            dCLDES_dSW25 = 0.0
            dCLDES_dCAM = 0.0

        partials[Mission.Design.LIFT_COEFFICIENT, Aircraft.Wing.ASPECT_RATIO] = \
            dCLDES_dAR
        partials[Mission.Design.LIFT_COEFFICIENT, Aircraft.Wing.THICKNESS_TO_CHORD] = \
            dCLDES_dTC
        partials[
            Mission.Design.LIFT_COEFFICIENT, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN
        ] = dCLDES_dCAM
        partials[Mission.Design.LIFT_COEFFICIENT, Aircraft.Wing.SWEEP] = dCLDES_dSW25

        if TC.real > 0.065 or VMAX < 1.0:  # subsonic
            TC23 = TC ** (2.0 / 3.0)

            x = np.array([CLDES[0], TC23[0]])
            ANS1, dANS1 = CMDEStable.interpolate(x, compute_derivative=True)
            ANS2, dANS2 = AMDEStable.interpolate(x, compute_derivative=True)

            ANS = ANS1 * (2.0 - AITEK) + ANS2 * (AITEK - 1.0)

            dANS = dANS1 * (2.0 - AITEK) + dANS2 * (AITEK - 1.0)
            dANS_dCLDES = dANS[0, 0]
            dANS_dTC23 = dANS[0, 1]

            dDESM2D_dANS = 0.5 / np.sqrt(ANS + 1)

            # wrt CLDES
            dDESM2D_dCLDES = dDESM2D_dANS * dANS_dCLDES

            # wrt AR
            dTC23_dAR = 0.0
            dANS_dAR = dANS_dCLDES * dCLDES_dAR + dANS_dTC23 * dTC23_dAR
            dDESM2D_dAR = dDESM2D_dANS * dANS_dAR

            # wrt TC
            dTC23_dTC = (2.0/3.0) * TC**(-1.0/3.0)
            dANS_dTC = dANS_dCLDES * dCLDES_dTC + dANS_dTC23 * dTC23_dTC
            dDESM2D_dTC = dDESM2D_dANS * dANS_dTC

            # wrt CAM
            dTC23_dCAM = 0.0
            dANS_dCAM = dANS_dCLDES * dCLDES_dCAM + dANS_dTC23 * dTC23_dCAM
            dDESM2D_dCAM = dDESM2D_dANS * dANS_dCAM

            # wrt SW25
            dTC23_dSW25 = 0.0
            dANS_dSW25 = dANS_dCLDES * dCLDES_dSW25 + dANS_dTC23 * dTC23_dSW25
            dDESM2D_dSW25 = dDESM2D_dANS * dANS_dSW25

        else:  # supersonic
            dDESM2D = self.dDESM2D

            # wrt CLDES
            dDESM2D_dCLDES = dDESM2D[0, 0]

            # wrt AR
            dDESM2D_dAR = dDESM2D_dCLDES * dCLDES_dAR

            # wrt TC
            dDESM2D_dTC = dDESM2D[0, 1] + dDESM2D_dCLDES * dCLDES_dTC

            # wrt CAM
            dTC_dCAM = 0.0
            dDESM2D_dCAM = dDESM2D_dCLDES * dCLDES_dCAM + dDESM2D_dTC * dTC_dCAM

            # wrt SW25
            dTC_dSW25 = 0.0
            dDESM2D_dSW25 = dDESM2D_dCLDES * dCLDES_dSW25 + dDESM2D_dTC * dTC_dSW25

        # wrt AR
        a, b, c = self.des_mach_coeff
        dDMDAR_dAR = -c / AR**2
        dDESM_dAR = dDESM2D_dAR + dDMDAR_dAR

        # wrt TC
        dDESM_dTC = dDESM2D_dTC

        # wrt CAM
        dDESM_dCAM = dDESM2D_dCAM

        # wrt SW25
        dDMDSWP_dSW25 = (a / b) * np.sin(SW25 / b)
        dDESM_dSW25 = dDESM2D_dSW25 + dDMDSWP_dSW25

        partials[Mission.Design.MACH, Aircraft.Wing.ASPECT_RATIO] = dDESM_dAR
        partials[Mission.Design.MACH, Aircraft.Wing.THICKNESS_TO_CHORD] = dDESM_dTC
        partials[Mission.Design.MACH, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN] = \
            dDESM_dCAM
        partials[Mission.Design.MACH, Aircraft.Wing.SWEEP] = dDESM_dSW25


AMDES = np.array(
    [[8003,     .180,   .240,   .300],
     [.10,   -.2080, -.3330, -.4590],
     [.20,   -.2180, -.3430, -.4680],
     [.30,   -.2290, -.3530, -.4780],
     [.40,   -.2420, -.3650, -.4880],
     [.50,   -.2580, -.3770, -.4950],
     [.60,   -.2710, -.3880, -.5070],
     [.70,   -.2940, -.4130, -.5300],
     [.80,   -.3170, -.4310, -.5460]])

CMDES = np.array(
    [[7003,   0.180000,  0.240000,  0.300000],
     [0.100000,  -0.374000, -0.445000, -0.513000],
     [0.200000,  -0.385000, -0.461000, -0.537000],
     [0.300000,  -0.401000, -0.478000, -0.556000],
     [0.400000,  -0.416000, -0.490000, -0.564000],
     [0.500000,  -0.441000, -0.509000, -0.578000],
     [0.600000,  -0.474000, -0.532000, -0.591000],
     [0.700000,  -0.518000, -0.571000, -0.621000]])

HSMDES = np.array(
    [[6003,   0.020000,  0.040000,  0.060000],
     [0.000000,   0.844000,  0.822000,  0.801000],
     [0.100000,   0.836000,  0.815000,  0.793000],
     [0.200000,   0.829000,  0.807000,  0.786000],
     [0.300000,   0.820000,  0.799000,  0.778000],
     [0.400000,   0.811000,  0.791000,  0.770000],
     [0.500000,   0.802000,  0.781000,  0.759000]])

AMDEStable = InterpND(
    method='slinear', points=(AMDES[1:, 0], AMDES[0, 1:]), values=AMDES[1:, 1:],
    extrapolate=True)
CMDEStable = InterpND(
    method='slinear', points=(CMDES[1:, 0], CMDES[0, 1:]), values=CMDES[1:, 1:],
    extrapolate=True)
HSMDEStable = InterpND(
    method='slinear', points=(HSMDES[1:, 0], HSMDES[0, 1:]), values=HSMDES[1:, 1:],
    extrapolate=True)
