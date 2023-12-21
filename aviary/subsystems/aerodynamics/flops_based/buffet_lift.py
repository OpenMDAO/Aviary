# %%
import numpy as np
import openmdao.api as om
import scipy.constants as _units
from openmdao.components.interp_util.interp import InterpND

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class BuffetLift(om.ExplicitComponent):
    """
    Computes lift margin before buffet onset.
    """

    def initialize(self):

        self.options.declare("num_nodes", default=1, types=int,
                             desc="Number of nodes along mission segment")

    def setup(self):
        nn = self.options["num_nodes"]

        # Simulation inputs
        self.add_input(
            Dynamic.Mission.MACH,
            shape=(nn),
            units='unitless',
            desc="Mach number")

        # Aero design inputs
        add_aviary_input(self, Mission.Design.MACH, 0.0)

        # Aircraft design inputs
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, 0.0)
        add_aviary_input(self, Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, 0.0)
        add_aviary_input(self, Aircraft.Wing.SWEEP, 0.0)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, 0.0)

        # Declare outputs
        self.add_output("DELCLB", shape=(nn), units='unitless',
                        desc="Delta lift coefficient before buffet onset")

    def setup_partials(self):
        nn = self.options["num_nodes"]

        self.declare_partials(
            'DELCLB',
            Dynamic.Mission.MACH,
            rows=np.arange(nn),
            cols=np.arange(nn))
        self.declare_partials('DELCLB', [Aircraft.Wing.ASPECT_RATIO,
                                         Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                                         Aircraft.Wing.SWEEP,
                                         Aircraft.Wing.THICKNESS_TO_CHORD,
                                         Mission.Design.MACH])

    def compute(self, inputs, outputs):
        """
        Computes lift margin before buffet onset.
        """
        mach, design_Mach, AR, CAM, SW25, TC = inputs.values()

        del_Mach = mach - design_Mach

        TC23 = TC**(2.0/3.0)
        TC23 = np.repeat(TC23, len(del_Mach))
        x = np.transpose(np.array([TC23, del_Mach]))
        # dFCLB = [dFCLB_dTC23, dFCLB_dDELM]
        FCLB, dFCLB = BUFTTable.interpolate(x, compute_derivative=True)
        DELCLB = FCLB * (AR * (1.0 + CAM/10.0) / np.cos(SW25 / _units.degree))

        self.FCLB = FCLB
        self.dFCLB = dFCLB
        outputs['DELCLB'] = DELCLB

    def compute_partials(self, inputs, partials):

        TC = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        SW25 = inputs[Aircraft.Wing.SWEEP]
        AR = inputs[Aircraft.Wing.ASPECT_RATIO]
        CAM = inputs[Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN]

        FCLB = self.FCLB
        dFCLB_dTC23 = self.dFCLB[:, 0]
        dFCLB_ddel_Mach = self.dFCLB[:, 1]

        a = _units.degree
        SW25_rad = SW25 / a
        cos_fact = 1.0 / np.cos(SW25_rad)

        dTC23_dTC = (2.0 / 3.0) * TC ** (-1.0 / 3.0)
        dDELCLB_dFCLB = AR * (1.0 + CAM/10.0) * cos_fact
        ddel_Mach_dMach = 1.0
        ddel_Mach_ddesign_Mach = -1.0

        # wrt Mach
        dCLB_dMach = dDELCLB_dFCLB * dFCLB_ddel_Mach * ddel_Mach_dMach

        # wrt design_Mach
        dCLB_ddesign_Mach = dDELCLB_dFCLB * dFCLB_ddel_Mach * ddel_Mach_ddesign_Mach

        # wrt AR
        dCLB_dAR = FCLB * (1.0 + CAM/10.0) * cos_fact

        # wrt TC
        dCLB_dTC = dDELCLB_dFCLB * dFCLB_dTC23 * dTC23_dTC

        # wrt SW25
        dCLB_dSW25 = (1.0 / a) * AR * FCLB * (1.0 + CAM/10.0) * \
            np.sin(SW25_rad) * cos_fact ** 2

        # wrt CAM
        dCLB_dCAM = FCLB * AR / 10.0 * cos_fact

        partials["DELCLB", Dynamic.Mission.MACH] = dCLB_dMach
        partials["DELCLB", Mission.Design.MACH] = dCLB_ddesign_Mach
        partials['DELCLB', Aircraft.Wing.ASPECT_RATIO] = dCLB_dAR
        partials['DELCLB', Aircraft.Wing.THICKNESS_TO_CHORD] = dCLB_dTC
        partials['DELCLB', Aircraft.Wing.SWEEP] = dCLB_dSW25
        partials['DELCLB', Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN] = dCLB_dCAM

        return


BUFT = np.array(
    [[8010,  -0.8,  -0.6,  -0.4,  -0.3,  -0.2,  -0.1,   0.0,   0.05,    0.1,   0.15],
     [0.1, 0.078, 0.076, 0.061, 0.050, 0.046, 0.051, 0.070,   0.09,   0.12,  0.156],
     [0.12, 0.078, 0.076, 0.061, 0.050, 0.043, 0.036, 0.039,  0.051,  0.073,  0.104],
     [0.14, 0.078, 0.076, 0.061, 0.050, 0.042, 0.034, 0.030,  0.031,  0.040,  0.057],
     [0.16, 0.078, 0.076, 0.061, 0.050, 0.041, 0.031, 0.023,  0.020,  0.016,  0.010],
     [0.18, 0.078, 0.076, 0.061, 0.050, 0.040, 0.030, 0.018,  0.010,  0.002, -0.009],
     [0.20, 0.078, 0.076, 0.061, 0.050, 0.040, 0.029, 0.015,  0.005, -0.007, -0.020],
     [0.24, 0.078, 0.076, 0.061, 0.050, 0.040, 0.028, 0.009, -0.005, -0.020, -0.036],
     [0.30, 0.078, 0.076, 0.061, 0.050, 0.040, 0.028, 0.004, -0.015, -0.037, -0.060]])


BUFTTable = InterpND(method='lagrange2', points=(BUFT[1:, 0], BUFT[0, 1:]), values=BUFT[1:, 1:],
                     extrapolate=True)
