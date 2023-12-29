# %%
import numpy as np
import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LiftDependentDrag(om.ExplicitComponent):
    """
    Calculates lift dependent drag
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int,
                             desc="Number of nodes along mission segment")
        self.options.declare('gamma', default=1.4,
                             desc='Ratio of specific heats for air.')

    def setup(self):
        nn = self.options["num_nodes"]

        # Simulation inputs
        self.add_input(Dynamic.Mission.MACH, shape=(
            nn), units='unitless', desc="Mach number")
        self.add_input(Dynamic.Mission.LIFT, shape=(
            nn), units="lbf", desc="Lift magnitude")
        self.add_input(Dynamic.Mission.STATIC_PRESSURE, np.ones(nn), units='lbf/ft**2',
                       desc='Static pressure at each evaulation point.')

        # Aero design inputs
        add_aviary_input(self, Mission.Design.LIFT_COEFFICIENT, 0.0)
        add_aviary_input(self, Mission.Design.MACH, 0.0)

        # Aircraft design inputs
        add_aviary_input(self, Aircraft.Wing.AREA, 0.0)
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, 0.0)
        add_aviary_input(self, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.0)
        add_aviary_input(self, Aircraft.Wing.SWEEP, 0.0)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, 0.0)

        # Declare outputs
        self.add_output("CD", shape=(nn), units='unitless',
                        desc="Coefficient of lift dependent drag")

    def setup_partials(self):
        nn = self.options["num_nodes"]

        self.declare_partials('CD', [Dynamic.Mission.MACH, Dynamic.Mission.LIFT, Dynamic.Mission.STATIC_PRESSURE],
                              rows=np.arange(nn), cols=np.arange(nn))

        wrt = [Mission.Design.LIFT_COEFFICIENT,
               Mission.Design.MACH,
               Aircraft.Wing.AREA,
               Aircraft.Wing.ASPECT_RATIO,
               Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
               Aircraft.Wing.SWEEP,
               Aircraft.Wing.THICKNESS_TO_CHORD]

        self.declare_partials('CD', wrt)

    def edge_interp(self, A1, A2, FCDP1, FCDP2, dFCDP1, dFCDP2, A):

        den = 1.0 / ((A - A1)*FCDP1 - (A - A2)*FCDP2)
        FCDP = 2.0 * FCDP1 * FCDP2 * den

        # Derivative of FCDP w.r.t A variable at value of A
        dFCDP_dA = 2.0 * FCDP1 * FCDP2 * (FCDP2 - FCDP1) * den ** 2

        dFCDP_dDEL = 2.0 * den * (dFCDP1 * (FCDP2 - FCDP1 * FCDP2 * den * (A - A1)) +
                                  dFCDP2 * (FCDP1 + FCDP1 * FCDP2 * den * (A - A2)))

        return FCDP, dFCDP_dDEL[:, 0], dFCDP_dDEL[:, 1], dFCDP_dA

    def inner_interp(self, arrA, FCDP1, FCDP2, FCDP3, FCDP4, FCDP5, dFCDP1, dFCDP2, dFCDP3, dFCDP4, dFCDP5, A):

        # Derivative of FCDP w.r.t A variable at value of A
        arrFCDP = np.array([FCDP1, FCDP2, FCDP3, FCDP4, FCDP5])
        arrFCDP = np.reshape(arrFCDP, (5))
        interp_arrFCDP = InterpND(method='lagrange2', points=(arrA), values=arrFCDP)
        FCDP, deriv = interp_arrFCDP.interpolate(A, compute_derivative=True)  # at A
        dFCDP_dA = deriv[:, 0]  # at A

        # Derivative of FCDP w.r.t DELM variable at value of A
        arrdFCDP_dDELM = np.array(
            [dFCDP1[:, 0], dFCDP2[:, 0], dFCDP3[:, 0], dFCDP4[:, 0], dFCDP5[:, 0]])
        arrdFCDP_dDELM = np.reshape(arrdFCDP_dDELM, (5))
        interp_arrdFCDP_dDELM = InterpND(
            method='lagrange2', points=(arrA), values=arrdFCDP_dDELM)
        dFCDP_dDELM, deriv = interp_arrdFCDP_dDELM.interpolate(
            A, compute_derivative=True)  # at A

        # Derivative of FCDP w.r.t DELCL variable at value of A
        arrdFCDP_dDELCL = np.array(
            [dFCDP1[:, 1], dFCDP2[:, 1], dFCDP3[:, 1], dFCDP4[:, 1], dFCDP5[:, 1]])
        arrdFCDP_dDELCL = np.reshape(arrdFCDP_dDELCL, (5))
        interp_arrdFCDP_dDELCL = InterpND(
            method='lagrange2', points=(arrA), values=arrdFCDP_dDELCL)
        dFCDP_dDELCL, deriv = interp_arrdFCDP_dDELCL.interpolate(
            A, compute_derivative=True)  # at A

        return FCDP, dFCDP_dDELM, dFCDP_dDELCL, dFCDP_dA

    def compute(self, inputs, outputs):
        """
        Calculate lift-dependent drag.

        :param inputs: _description_
        :type inputs: _type_
        :param outputs: _description_
        :type outputs: _type_
        """
        nn = self.options["num_nodes"]
        gamma = self.options['gamma']
        mach, lift, P, CLDES, MDES, Sref, AR, CAM, SW25, TC = inputs.values()

        FCDP = np.empty(nn, dtype=mach.dtype)
        DCDP = np.empty(nn, dtype=mach.dtype)
        dFCDP_dDELM = np.empty(nn, dtype=mach.dtype)
        dFCDP_dDELCL = np.empty(nn, dtype=mach.dtype)
        dFCDP_dA = np.empty(nn, dtype=mach.dtype)

        CL = 2.0 * lift / (Sref * gamma * P * mach ** 2)

        DELCL = CL - CLDES
        DELM = mach - MDES
        A = self.A = AR * TC ** (1.0/3.0)

        for i in range(0, nn):

            x = np.array([DELM[i], DELCL[i]])

            if DELM[i].real <= 0.075:

                if A.real < 0.5:

                    FCDP1, dFCDP1 = AR05table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = AR1table.interpolate(x, compute_derivative=True)

                    A1 = 0.5
                    A2 = 1.0
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.edge_interp(
                        A1, A2, FCDP1, FCDP2, dFCDP1, dFCDP2, A
                    )

                elif 0.5 <= A.real < 6:

                    FCDP1, dFCDP1 = AR05table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = AR1table.interpolate(x, compute_derivative=True)
                    FCDP3, dFCDP3 = AR2table.interpolate(x, compute_derivative=True)
                    FCDP4, dFCDP4 = AR4table.interpolate(x, compute_derivative=True)
                    FCDP5, dFCDP5 = AR6table.interpolate(x, compute_derivative=True)

                    arrA = np.array([0.5, 1, 2, 4, 6])
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.inner_interp(
                        arrA, FCDP1, FCDP2, FCDP3, FCDP4, FCDP5, dFCDP1, dFCDP2, dFCDP3, dFCDP4, dFCDP5, A
                    )

                else:

                    FCDP1, dFCDP1 = AR4table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = AR6table.interpolate(x, compute_derivative=True)

                    A1 = 4.0
                    A2 = 6.0
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.edge_interp(
                        A1, A2, FCDP1, FCDP2, dFCDP1, dFCDP2, A
                    )

            else:

                if A.real < 0.7:

                    FCDP1, dFCDP1 = ARS07table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = ARS08table.interpolate(x, compute_derivative=True)

                    A1 = 0.7
                    A2 = 0.8
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.edge_interp(
                        A1, A2, FCDP1, FCDP2, dFCDP1, dFCDP2, A
                    )

                elif 0.7 <= A.real <= 1.4:

                    FCDP1, dFCDP1 = ARS07table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = ARS08table.interpolate(x, compute_derivative=True)
                    FCDP3, dFCDP3 = ARS10table.interpolate(x, compute_derivative=True)
                    FCDP4, dFCDP4 = ARS12table.interpolate(x, compute_derivative=True)
                    FCDP5, dFCDP5 = ARS14table.interpolate(x, compute_derivative=True)

                    arrA = np.array([0.7, 0.8, 1.0, 1.2, 1.4])
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.inner_interp(
                        arrA, FCDP1, FCDP2, FCDP3, FCDP4, FCDP5, dFCDP1, dFCDP2, dFCDP3, dFCDP4, dFCDP5, A
                    )

                elif 1.4 < A.real <= 2.0:

                    FCDP1, dFCDP1 = ARS12table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = ARS14table.interpolate(x, compute_derivative=True)
                    FCDP3, dFCDP3 = ARS16table.interpolate(x, compute_derivative=True)
                    FCDP4, dFCDP4 = ARS18table.interpolate(x, compute_derivative=True)
                    FCDP5, dFCDP5 = ARS20table.interpolate(x, compute_derivative=True)

                    arrA = np.array([1.2, 1.4, 1.6, 1.8, 2.0])
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.inner_interp(
                        arrA, FCDP1, FCDP2, FCDP3, FCDP4, FCDP5, dFCDP1, dFCDP2, dFCDP3, dFCDP4, dFCDP5, A
                    )

                else:

                    FCDP1, dFCDP1 = ARS18table.interpolate(x, compute_derivative=True)
                    FCDP2, dFCDP2 = ARS20table.interpolate(x, compute_derivative=True)

                    A1 = 1.8
                    A2 = 2.0
                    FCDP[i], dFCDP_dDELM[i], dFCDP_dDELCL[i], dFCDP_dA[i] = self.edge_interp(
                        A1, A2, FCDP1, FCDP2, dFCDP1, dFCDP2, A
                    )

        DCDP = FCDP * (1.0 + CAM/10.0) * A/AR
        self.clamp_indices = np.where(DCDP < 0)
        DCDP[DCDP < 0] = 0.0

        self.FCDP = FCDP
        self.dFCDP_dA = dFCDP_dA
        self.dFCDP_dDELM = dFCDP_dDELM
        self.dFCDP_dDELCL = dFCDP_dDELCL

        outputs["CD"] = DCDP

    def compute_partials(self, inputs, partials):
        """
        Calculate partials of lift-dependent drag.

        :param inputs: _description_
        :type inputs: _type_
        :param partials: _description_
        :type partials: _type_
        """
        gamma = self.options['gamma']
        mach, lift, P, CLDES, MDES, Sref, AR, CAM, SW25, TC = inputs.values()

        ddelCL_dL = 2.0 / (Sref * gamma * P * mach ** 2)
        ddelCL_dP = -2.0 * lift / (Sref * gamma * P * P * mach ** 2)
        ddelCL_dSref = -2.0 * lift / (Sref * Sref * gamma * P * mach ** 2)
        ddelCL_dmach = -4.0 * lift / (Sref * gamma * P * mach ** 3)

        A = self.A
        FCDP = self.FCDP
        dFCDP_dA = self.dFCDP_dA
        dFCDP_dDELM = self.dFCDP_dDELM
        dFCDP_dDELCL = self.dFCDP_dDELCL

        # A = AR * TC**(1.0 / 3.0)
        dA_dAR = TC**(1.0 / 3.0)
        dA_dTC = (1.0 / 3.0) * AR * TC**(-2.0 / 3.0)
        dA_dCAM = 0.0
        dA_dSW25 = 0.0

        dDCDP_dFCDP = (1 + CAM/10.0) * TC**(1.0 / 3.0)

        ddelCL_dL = 2.0 / (Sref * gamma * P * mach ** 2)

        # Derivative of lift-dependent drag w.r.t. Mach
        dCD_dmach = dDCDP_dFCDP * dFCDP_dDELM

        # Derivative of lift-dependent drag w.r.t. CL
        dCD_dCL = dDCDP_dFCDP * dFCDP_dDELCL

        # Derivative of lift-dependent drag w.r.t. wing aspect ratio
        dFCDP_dAR = dFCDP_dA * dA_dAR
        dCD_dAR = dDCDP_dFCDP * dFCDP_dAR

        # Derivative of lift-dependent drag w.r.t. thickness-to-chord ratio
        dDCDP_dTC = (1.0 / 3.0) * FCDP * (1 + CAM/10.0) * TC**(-2.0 / 3.0)
        dFCDP_dTC = dFCDP_dA * dA_dTC
        dCD_dTC = dDCDP_dFCDP * dFCDP_dTC + dDCDP_dTC

        # Derivative of lift-dependent drag w.r.t. wing camber
        dDCDP_dCAM = (FCDP / 10.0) * TC**(1.0 / 3.0)
        dFCDP_dCAM = dFCDP_dA * dA_dCAM
        dCD_dCAM = dDCDP_dFCDP * dFCDP_dCAM + dDCDP_dCAM

        # Derivative of lift-dependent drag w.r.t. wing quarted-chord sweep
        dFCDP_dSW25 = dFCDP_dA * dA_dSW25
        dCD_dSW25 = dDCDP_dFCDP * dFCDP_dSW25

        partials["CD", Dynamic.Mission.MACH] = dCD_dmach + dCD_dCL * ddelCL_dmach
        partials["CD", Dynamic.Mission.LIFT] = dCD_dCL * ddelCL_dL
        partials["CD", Dynamic.Mission.STATIC_PRESSURE] = dCD_dCL * ddelCL_dP
        partials["CD", Aircraft.Wing.AREA] = dCD_dCL * ddelCL_dSref
        partials["CD", Aircraft.Wing.ASPECT_RATIO] = dCD_dAR
        partials["CD", Aircraft.Wing.THICKNESS_TO_CHORD] = dCD_dTC
        partials["CD", Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN] = dCD_dCAM
        partials["CD", Aircraft.Wing.SWEEP] = dCD_dSW25
        partials["CD", Mission.Design.LIFT_COEFFICIENT] = -dCD_dCL
        partials["CD", Mission.Design.MACH] = -dCD_dmach

        if self.clamp_indices:
            partials["CD", Dynamic.Mission.MACH][self.clamp_indices] = 0.0
            partials["CD", Dynamic.Mission.LIFT][self.clamp_indices] = 0.0
            partials["CD", Dynamic.Mission.STATIC_PRESSURE][self.clamp_indices] = 0.0
            partials["CD", Aircraft.Wing.AREA][self.clamp_indices] = 0.0
            partials["CD", Aircraft.Wing.ASPECT_RATIO][self.clamp_indices] = 0.0
            partials["CD", Aircraft.Wing.THICKNESS_TO_CHORD][self.clamp_indices] = 0.0
            partials["CD", Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN][self.clamp_indices] = 0.0
            partials["CD", Aircraft.Wing.SWEEP][self.clamp_indices] = 0.0
            partials["CD", Mission.Design.LIFT_COEFFICIENT][self.clamp_indices] = 0.0
            partials["CD", Mission.Design.MACH][self.clamp_indices] = 0.0


# Tables
AR05 = np.array(
    [[11010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [-0.800000,   0.001500,  0.000400,  0.001500,  0.002400,  0.003500,
         0.006000,  0.014000,  0.028000,  0.082500,  0.150000],
     [-0.200000,   0.001500,  0.000400,  0.001500,  0.002400,  0.003500,
         0.006000,  0.014000,  0.028000,  0.082500,  0.150000],
     [-0.160000,   0.001500,  0.000400,  0.001500,  0.002400,  0.003500,
         0.006000,  0.014000,  0.028000,  0.082500,  0.150000],
     [-0.120000,   0.001500,  0.000400,  0.001500,  0.002400,  0.003500,
         0.006000,  0.014000,  0.028000,  0.082500,  0.150000],
     [-0.080000,   0.001500,  0.000400,  0.001500,  0.002400,  0.003500,
         0.006000,  0.014000,  0.028000,  0.082500,  0.150000],
     [-0.040000,   0.001500,  0.000400,  0.001500,  0.002600,  0.003500,
         0.007500,  0.014000,  0.028000,  0.082500,  0.152500],
     [-0.020000,   0.001600,  0.000400,  0.001600,  0.002800,  0.003500,
         0.008500,  0.014000,  0.029000,  0.084000,  0.156000],
     [0.000000,   0.001800,  0.000400,  0.001800,  0.003000,  0.003600,
         0.010000,  0.014000,  0.031000,  0.086000,  0.162000],
     [0.020000,   0.002050,  0.000400,  0.002100,  0.003100,  0.003900,
         0.010000,  0.016000,  0.033000,  0.089500,  0.168000],
     [0.040000,   0.002300,  0.000400,  0.002300,  0.003200,  0.004500,
         0.010000,  0.018000,  0.034000,  0.095500,  0.172500],
     [0.050000,   0.002400,  0.000400,  0.002400,  0.003200,  0.004900,  0.010000,  0.019000,  0.034000,  0.100000,  0.175000]])

AR1 = np.array(
    [[11010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [-0.800000,   0.001600,  0.000500,  0.001600,  0.002900,  0.005100,
         0.006500,  0.013000,  0.024000,  0.060000,  0.120000],
     [-0.200000,   0.001600,  0.000500,  0.001600,  0.002900,  0.005100,
         0.006500,  0.013000,  0.024000,  0.060000,  0.120000],
     [-0.160000,   0.001600,  0.000500,  0.001600,  0.002950,  0.005100,
         0.006500,  0.013000,  0.024000,  0.060000,  0.120000],
     [-0.120000,   0.001600,  0.000500,  0.001600,  0.003050,  0.005100,
         0.006500,  0.013000,  0.024000,  0.060000,  0.120000],
     [-0.080000,   0.001600,  0.000500,  0.001600,  0.003150,  0.005100,
         0.006500,  0.013000,  0.024000,  0.060000,  0.120000],
     [-0.040000,   0.001700,  0.000500,  0.001700,  0.003500,  0.005300,
         0.006500,  0.013500,  0.024500,  0.060000,  0.120000],
     [-0.020000,   0.001800,  0.000500,  0.001800,  0.003900,  0.005500,
         0.007500,  0.015000,  0.026000,  0.062000,  0.123500],
     [0.000000,   0.001900,  0.000500,  0.001900,  0.004250,  0.006000,
         0.009500,  0.016500,  0.027500,  0.064500,  0.127500],
     [0.020000,   0.002200,  0.000500,  0.002200,  0.004500,  0.006900,
         0.011500,  0.019000,  0.030000,  0.067000,  0.132500],
     [0.040000,   0.002600,  0.000500,  0.002600,  0.004650,  0.008400,
         0.014000,  0.023000,  0.034000,  0.071000,  0.138000],
     [0.050000,   0.003000,  0.000500,  0.003000,  0.004750,  0.009300,  0.016000,  0.026000,  0.037500,  0.074000,  0.141000]])

AR2 = np.array(
    [[11010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [-0.800000,   0.001500,  0.000500,  0.001500,  0.002800,  0.004700,
         0.007500,  0.011000,  0.018000,  0.035000,  0.073000],
     [-0.200000,   0.001500,  0.000500,  0.001500,  0.002800,  0.004700,
         0.007500,  0.011000,  0.018000,  0.035000,  0.073000],
     [-0.160000,   0.001500,  0.000500,  0.001500,  0.002900,  0.004700,
         0.007500,  0.011000,  0.018000,  0.035000,  0.073000],
     [-0.120000,   0.001600,  0.000500,  0.001600,  0.003100,  0.004700,
         0.007500,  0.011000,  0.018000,  0.035000,  0.073000],
     [-0.080000,   0.001600,  0.000500,  0.001600,  0.003400,  0.004700,
         0.007500,  0.011000,  0.018000,  0.035000,  0.073000],
     [-0.040000,   0.001800,  0.000500,  0.001800,  0.003700,  0.004900,
         0.007500,  0.012000,  0.018500,  0.035000,  0.073000],
     [-0.020000,   0.001900,  0.000500,  0.001900,  0.004100,  0.005300,
         0.008700,  0.014000,  0.020500,  0.036500,  0.073500],
     [0.000000,   0.002100,  0.000500,  0.002100,  0.004700,  0.006500,
         0.010600,  0.016500,  0.023500,  0.040000,  0.075000],
     [0.020000,   0.002200,  0.000600,  0.002600,  0.005700,  0.008400,
         0.014000,  0.019500,  0.028000,  0.046000,  0.078000],
     [0.040000,   0.002900,  0.000600,  0.003300,  0.007500,  0.011000,
         0.017000,  0.024500,  0.032500,  0.055000,  0.084000],
     [0.050000,   0.003700,  0.000700,  0.003700,  0.008800,  0.013500,  0.021000,  0.029000,  0.037000,  0.060000,  0.090000]])

AR4 = np.array(
    [[11010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [-0.800000,   0.001100,  0.000400,  0.001100,  0.002800,  0.003400,
         0.005500,  0.008000,  0.011500,  0.020000,  0.036000],
     [-0.200000,   0.001100,  0.000400,  0.001100,  0.002800,  0.003400,
         0.005500,  0.008000,  0.011500,  0.020000,  0.036000],
     [-0.160000,   0.001100,  0.000400,  0.001100,  0.002800,  0.003450,
         0.005500,  0.008000,  0.011500,  0.020000,  0.036000],
     [-0.120000,   0.001200,  0.000400,  0.001200,  0.002850,  0.003550,
         0.005500,  0.008000,  0.011500,  0.020000,  0.036000],
     [-0.080000,   0.001300,  0.000400,  0.001300,  0.002900,  0.003650,
         0.005500,  0.008000,  0.011500,  0.020000,  0.036000],
     [-0.040000,   0.001400,  0.000400,  0.001400,  0.003050,  0.003900,
         0.005500,  0.008000,  0.011500,  0.020000,  0.036000],
     [-0.020000,   0.001600,  0.000400,  0.001600,  0.003200,  0.004100,
         0.005800,  0.008100,  0.012300,  0.021000,  0.037000],
     [0.000000,   0.001900,  0.000400,  0.001900,  0.003550,  0.004600,
         0.007500,  0.011200,  0.015800,  0.026000,  0.040000],
     [0.020000,   0.002400,  0.000500,  0.002400,  0.004500,  0.006000,
         0.010000,  0.014000,  0.019500,  0.031000,  0.046000],
     [0.040000,   0.003200,  0.000600,  0.003500,  0.006800,  0.012200,
         0.018000,  0.024000,  0.030000,  0.043000,  0.056000],
     [0.050000,   0.004400,  0.000600,  0.004400,  0.011300,  0.018100,  0.024000,  0.031000,  0.038000,  0.051000,  0.064000]])

AR6 = np.array(
    [[11009,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000],
     [-0.800000,   0.000850,  0.000200,  0.000850,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007100,  0.010700],
     [-0.200000,   0.000850,  0.000200,  0.000850,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007100,  0.010700],
     [-0.160000,   0.000890,  0.000200,  0.000890,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007100,  0.010700],
     [-0.120000,   0.000960,  0.000200,  0.000960,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007100,  0.010700],
     [-0.080000,   0.001080,  0.000200,  0.001080,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007100,  0.010700],
     [-0.040000,   0.001200,  0.000260,  0.001200,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007100,  0.011700],
     [-0.020000,   0.001300,  0.000300,  0.001300,  0.002000,
         0.002700,  0.004100,  0.005800,  0.007700,  0.016700],
     [0.000000,   0.001480,  0.000340,  0.001480,  0.002300,
         0.002900,  0.004300,  0.006100,  0.011200,  0.024000],
     [0.020000,   0.001850,  0.000370,  0.001850,  0.003200,
         0.004500,  0.007200,  0.013000,  0.018500,  0.039500],
     [0.040000,   0.002900,  0.000450,  0.002900,  0.007000,
         0.010500,  0.014500,  0.024000,  0.033000,  0.054500],
     [0.050000,   0.003900,  0.000520,  0.003900,  0.011500,  0.016000,  0.027000,  0.037000,  0.047000,  0.068000]])

ARS07 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.001500,  0.000000,  0.001500,  0.004500,  0.008000,
         0.013500,  0.020000,  0.036000,  0.085000,  0.165000],
     [0.100000,   0.001700,  0.000000,  0.001700,  0.004500,  0.009000,
         0.015000,  0.023500,  0.041500,  0.091000,  0.200000],
     [0.150000,   0.001800,  0.000000,  0.001800,  0.004700,  0.010500,
         0.017000,  0.026500,  0.048000,  0.100000,  0.230000],
     [0.200000,   0.002000,  0.000000,  0.002000,  0.005000,  0.012000,
         0.019000,  0.030000,  0.055000,  0.112000,  0.260000],
     [0.300000,   0.003000,  0.000000,  0.003000,  0.006000,  0.015500,
         0.024000,  0.039500,  0.072000,  0.145000,  0.320000],
     [0.500000,   0.004000,  0.000000,  0.004000,  0.008300,  0.022500,
         0.036000,  0.067500,  0.112000,  0.210000,  0.440000],
     [0.700000,   0.004000,  0.000500,  0.004000,  0.012000,  0.029500,
         0.055000,  0.090000,  0.146000,  0.273000,  0.550000],
     [0.900000,   0.003600,  0.000700,  0.003600,  0.016000,  0.036500,
         0.065000,  0.110000,  0.174000,  0.330000,  0.630000],
     [1.100000,   0.003400,  0.000800,  0.003400,  0.020000,  0.042500,  0.074000,  0.130000,  0.200000,  0.380000,  0.700000]])

ARS08 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.001000,  0.002000,  0.004000,  0.005000,
         0.010000,  0.015000,  0.025000,  0.080000,  0.140000],
     [0.100000,   0.000000,  0.001000,  0.002500,  0.005000,  0.007500,
         0.014000,  0.018000,  0.029000,  0.086000,  0.158000],
     [0.150000,   0.000000,  0.001000,  0.003000,  0.006200,  0.009000,
         0.017000,  0.026000,  0.037000,  0.094000,  0.171000],
     [0.200000,   0.000000,  0.001000,  0.003500,  0.007500,  0.011500,
         0.020000,  0.032000,  0.051000,  0.103000,  0.185000],
     [0.300000,   0.000000,  0.001000,  0.004000,  0.009000,  0.016000,
         0.027500,  0.045000,  0.072000,  0.130000,  0.236000],
     [0.500000,   0.000000,  0.001000,  0.005000,  0.013000,  0.026500,
         0.044000,  0.069000,  0.106000,  0.190000,  0.335000],
     [0.700000,   0.000000,  0.001000,  0.006000,  0.016000,  0.035000,
         0.059000,  0.092000,  0.143000,  0.245000,  0.420000],
     [0.900000,   0.000000,  0.001000,  0.006500,  0.021000,  0.043000,
         0.071000,  0.113000,  0.175000,  0.290000,  0.480000],
     [1.100000,   0.000000,  0.001000,  0.007000,  0.026000,  0.051000,  0.080000,  0.135000,  0.200000,  0.320000,  0.520000]])

ARS10 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.000100,  0.003000,  0.006000,  0.008000,
         0.015500,  0.020000,  0.027000,  0.072000,  0.126000],
     [0.100000,   0.000000,  0.000200,  0.003700,  0.007500,  0.011500,
         0.018700,  0.026000,  0.036500,  0.076000,  0.135000],
     [0.150000,   0.000000,  0.000300,  0.004200,  0.009500,  0.013500,
         0.022000,  0.033000,  0.045000,  0.082000,  0.148000],
     [0.200000,   0.000000,  0.000400,  0.004800,  0.011500,  0.016000,
         0.025500,  0.040000,  0.053000,  0.089000,  0.160000],
     [0.300000,   0.000000,  0.000600,  0.006000,  0.014000,  0.020500,
         0.032500,  0.051000,  0.067000,  0.110000,  0.203000],
     [0.500000,   0.000000,  0.001000,  0.007500,  0.018500,  0.031000,
         0.048000,  0.070000,  0.098000,  0.160000,  0.280000],
     [0.700000,   0.000000,  0.001400,  0.009000,  0.024500,  0.042500,
         0.065000,  0.090000,  0.127000,  0.204000,  0.350000],
     [0.900000,   0.000000,  0.001800,  0.011000,  0.031500,  0.054000,
         0.082000,  0.112000,  0.152000,  0.240000,  0.405000],
     [1.100000,   0.000000,  0.002200,  0.012300,  0.038000,  0.065500,  0.100000,  0.136000,  0.176000,  0.270000,  0.450000]])

ARS12 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.000100,  0.003000,  0.008000,  0.011000,
         0.017500,  0.027500,  0.039000,  0.064000,  0.115000],
     [0.100000,   0.000000,  0.000200,  0.004000,  0.010000,  0.014500,
         0.022000,  0.032500,  0.045000,  0.068000,  0.120000],
     [0.150000,   0.000000,  0.000300,  0.005000,  0.012000,  0.016500,
         0.025000,  0.037500,  0.050000,  0.072000,  0.129000],
     [0.200000,   0.000000,  0.000400,  0.006000,  0.013500,  0.018000,
         0.032000,  0.043500,  0.055000,  0.077000,  0.138000],
     [0.300000,   0.000000,  0.000600,  0.007000,  0.016500,  0.022500,
         0.035500,  0.049500,  0.065000,  0.094000,  0.165000],
     [0.500000,   0.000000,  0.001000,  0.009000,  0.023500,  0.033000,
         0.048000,  0.067000,  0.088000,  0.140000,  0.250000],
     [0.700000,   0.000000,  0.001400,  0.015000,  0.030000,  0.046500,
         0.065000,  0.085500,  0.114000,  0.190000,  0.350000],
     [0.900000,   0.000000,  0.001800,  0.014000,  0.037500,  0.058000,
         0.083000,  0.108500,  0.140000,  0.240000,  0.460000],
     [1.100000,   0.000000,  0.002200,  0.017000,  0.045000,  0.073000,  0.101000,  0.133500,  0.165000,  0.300000,  0.600000]])

ARS14 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.001000,  0.003000,  0.010000,  0.014000,
         0.020000,  0.028000,  0.035500,  0.060000,  0.100000],
     [0.100000,   0.000000,  0.001000,  0.004000,  0.011500,  0.014500,
         0.021500,  0.030000,  0.038000,  0.062000,  0.105000],
     [0.150000,   0.000000,  0.001100,  0.005000,  0.013000,  0.018000,
         0.025000,  0.033000,  0.044000,  0.066000,  0.111000],
     [0.200000,   0.000000,  0.001200,  0.006000,  0.014000,  0.019000,
         0.025000,  0.035000,  0.048000,  0.071000,  0.122000],
     [0.300000,   0.000000,  0.001400,  0.007000,  0.017000,  0.023500,
         0.033500,  0.044000,  0.056000,  0.082000,  0.145000],
     [0.500000,   0.000000,  0.001700,  0.009500,  0.023500,  0.034500,
         0.048000,  0.063000,  0.081000,  0.116000,  0.235000],
     [0.700000,   0.000000,  0.002000,  0.012500,  0.031500,  0.047500,
         0.065000,  0.085000,  0.114000,  0.180000,  0.350000],
     [0.900000,   0.000000,  0.002200,  0.015000,  0.039500,  0.059000,
         0.081000,  0.108000,  0.135000,  0.240000,  0.450000],
     [1.100000,   0.000000,  0.002500,  0.018500,  0.047500,  0.074000,  0.101000,  0.129000,  0.155000,  0.300000,  0.550000]])

ARS16 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.001000,  0.003500,  0.009500,  0.013000,
         0.020000,  0.028000,  0.037000,  0.055000,  0.088000],
     [0.100000,   0.000000,  0.001000,  0.004000,  0.012000,  0.016000,
         0.022500,  0.030000,  0.040000,  0.061000,  0.092000],
     [0.150000,   0.000000,  0.001000,  0.005000,  0.013500,  0.018000,
         0.026000,  0.035000,  0.044500,  0.065000,  0.097000],
     [0.200000,   0.000000,  0.001000,  0.006000,  0.015000,  0.020000,
         0.029000,  0.039000,  0.048500,  0.072000,  0.104000],
     [0.300000,   0.000000,  0.001000,  0.007000,  0.018000,  0.024500,
         0.036000,  0.046000,  0.057500,  0.081000,  0.125000],
     [0.500000,   0.000000,  0.001500,  0.009500,  0.024000,  0.035000,
         0.050500,  0.063500,  0.078000,  0.108000,  0.190000],
     [0.700000,   0.000000,  0.002000,  0.012500,  0.032000,  0.047000,
         0.064500,  0.087000,  0.108000,  0.170000,  0.280000],
     [0.900000,   0.000000,  0.002500,  0.017000,  0.040000,  0.062500,
         0.083000,  0.106000,  0.130000,  0.240000,  0.380000],
     [1.100000,   0.000000,  0.003000,  0.022000,  0.048000,  0.076000,  0.102000,  0.129000,  0.154000,  0.300000,  0.490000]])

ARS18 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.000700,  0.003500,  0.011000,  0.014500,
         0.019000,  0.026000,  0.034000,  0.050000,  0.080000],
     [0.100000,   0.000000,  0.000800,  0.004200,  0.012000,  0.016500,
         0.024500,  0.031500,  0.040000,  0.060000,  0.082000],
     [0.150000,   0.000000,  0.000900,  0.005200,  0.013500,  0.018000,
         0.026500,  0.036500,  0.046500,  0.065500,  0.085000],
     [0.200000,   0.000000,  0.001000,  0.006000,  0.015000,  0.020000,
         0.030000,  0.039500,  0.049500,  0.070500,  0.090000],
     [0.300000,   0.000000,  0.001200,  0.007000,  0.018000,  0.024500,
         0.035500,  0.046500,  0.058500,  0.079000,  0.105000],
     [0.500000,   0.000000,  0.001600,  0.009500,  0.023500,  0.035000,
         0.049000,  0.063500,  0.079500,  0.108000,  0.150000],
     [0.700000,   0.000000,  0.002000,  0.012500,  0.031500,  0.047500,
         0.066500,  0.085500,  0.110000,  0.167000,  0.230000],
     [0.900000,   0.000000,  0.002500,  0.016400,  0.040000,  0.063000,
         0.086000,  0.108500,  0.131000,  0.232000,  0.330000],
     [1.100000,   0.000000,  0.003000,  0.022000,  0.048500,  0.076500,  0.104000,  0.130000,  0.155000,  0.302000,  0.440000]])

ARS20 = np.array(
    [[9010,  -0.400000, -0.300000, -0.200000, -0.100000, -0.050000,  0.000000,  0.050000,  0.100000,  0.200000,  0.300000],
     [0.050000,   0.000000,  0.000700,  0.003000,  0.011000,  0.014000,
         0.020000,  0.027000,  0.034000,  0.051000,  0.068000],
     [0.100000,   0.000000,  0.000800,  0.004500,  0.012000,  0.016000,
         0.024000,  0.030000,  0.039000,  0.059000,  0.080500],
     [0.150000,   0.000000,  0.000900,  0.005000,  0.013500,  0.018000,
         0.026000,  0.035000,  0.046000,  0.065000,  0.087000],
     [0.200000,   0.000000,  0.001000,  0.005700,  0.015000,  0.020000,
         0.030000,  0.039000,  0.049000,  0.069500,  0.091500],
     [0.300000,   0.000000,  0.001200,  0.007000,  0.018000,  0.024500,
         0.033000,  0.044000,  0.054000,  0.076000,  0.100000],
     [0.500000,   0.000000,  0.001600,  0.009500,  0.024000,  0.035000,
         0.047000,  0.058000,  0.069000,  0.093000,  0.134000],
     [0.700000,   0.000000,  0.002000,  0.012600,  0.031500,  0.048000,
         0.066000,  0.084000,  0.105000,  0.146000,  0.200000],
     [0.900000,   0.000000,  0.002800,  0.016000,  0.040000,  0.061500,
         0.084000,  0.108000,  0.131000,  0.210000,  0.290000],
     [1.100000,   0.000000,  0.003600,  0.022000,  0.048000,  0.075000,  0.102000,  0.128000,  0.155000,  0.269000,  0.375000]])

AR05table = InterpND(method='lagrange2', points=(
    AR05[1:, 0], AR05[0, 1:]), values=AR05[1:, 1:], extrapolate=True)
AR1table = InterpND(method='lagrange2', points=(
    AR1[1:, 0], AR1[0, 1:]), values=AR1[1:, 1:], extrapolate=True)
AR2table = InterpND(method='lagrange2', points=(
    AR2[1:, 0], AR2[0, 1:]), values=AR2[1:, 1:], extrapolate=True)
AR4table = InterpND(method='lagrange2', points=(
    AR4[1:, 0], AR4[0, 1:]), values=AR4[1:, 1:], extrapolate=True)
AR6table = InterpND(method='lagrange2', points=(
    AR6[1:, 0], AR6[0, 1:]), values=AR6[1:, 1:], extrapolate=True)
ARS07table = InterpND(method='lagrange2', points=(
    ARS07[1:, 0], ARS07[0, 1:]), values=ARS07[1:, 1:], extrapolate=True)
ARS08table = InterpND(method='lagrange2', points=(
    ARS08[1:, 0], ARS08[0, 1:]), values=ARS08[1:, 1:], extrapolate=True)
ARS10table = InterpND(method='lagrange2', points=(
    ARS10[1:, 0], ARS10[0, 1:]), values=ARS10[1:, 1:], extrapolate=True)
ARS12table = InterpND(method='lagrange2', points=(
    ARS12[1:, 0], ARS12[0, 1:]), values=ARS12[1:, 1:], extrapolate=True)
ARS14table = InterpND(method='lagrange2', points=(
    ARS14[1:, 0], ARS14[0, 1:]), values=ARS14[1:, 1:], extrapolate=True)
ARS16table = InterpND(method='lagrange2', points=(
    ARS16[1:, 0], ARS16[0, 1:]), values=ARS16[1:, 1:], extrapolate=True)
ARS18table = InterpND(method='lagrange2', points=(
    ARS18[1:, 0], ARS18[0, 1:]), values=ARS18[1:, 1:], extrapolate=True)
ARS20table = InterpND(method='lagrange2', points=(
    ARS20[1:, 0], ARS20[0, 1:]), values=ARS20[1:, 1:], extrapolate=True)
