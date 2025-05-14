import numpy as np
import openmdao.api as om
import scipy.constants as _units

from aviary.variable_info.functions import add_aviary_input, add_aviary_option
from aviary.variable_info.variables import Aircraft, Dynamic


class InducedDrag(om.ExplicitComponent):
    """Calculates induced drag."""

    def initialize(self):
        self.options.declare(
            'num_nodes', default=1, types=int, desc='Number of nodes along mission segment'
        )
        self.options.declare('gamma', default=1.4, desc='Ratio of specific heats for air.')

        add_aviary_option(self, Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION)

    def setup(self):
        nn = self.options['num_nodes']

        # Simulation inputs
        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')
        add_aviary_input(self, Dynamic.Vehicle.LIFT, shape=(nn), units='lbf')
        add_aviary_input(self, Dynamic.Atmosphere.STATIC_PRESSURE, shape=nn, units='lbf/ft**2')

        # Aero design inputs
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SWEEP, units='deg')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')

        # Declare outputs
        self.add_output(
            'induced_drag_coeff', shape=(nn), units='unitless', desc='Coefficient of induced drag'
        )

    def setup_partials(self):
        nn = self.options['num_nodes']

        row_col = np.arange(nn)
        self.declare_partials(
            'induced_drag_coeff',
            [
                Dynamic.Atmosphere.MACH,
                Dynamic.Vehicle.LIFT,
                Dynamic.Atmosphere.STATIC_PRESSURE,
            ],
            rows=row_col,
            cols=row_col,
        )

        wrt = [
            Aircraft.Wing.AREA,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.SPAN_EFFICIENCY_FACTOR,
            Aircraft.Wing.SWEEP,
            Aircraft.Wing.TAPER_RATIO,
        ]

        self.declare_partials('induced_drag_coeff', wrt, rows=row_col, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        options = self.options
        gamma = options['gamma']
        mach, lift, P, Sref, AR, span_efficiency_factor, SW25, TR = inputs.values()

        CL = 2.0 * lift / (Sref * gamma * P * mach**2)

        redux = self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]

        if redux:
            # Adjustment for extreme taper ratios.
            # Reference:
            # ----------
            # [1] DeYoung, John. "Advanced Supersonic Technology Concept Study Reference
            # Characteristics," NASA Contractor Report 132374.

            span_efficiency_0 = 1.0 + 0.1 * AR * (0.4226 * np.sqrt(AR) - 0.35 * TR - 0.143)
        else:
            span_efficiency_0 = 1.0

        if span_efficiency_factor <= 0.3:
            span_efficiency = span_efficiency_0 + span_efficiency_factor
        else:
            span_efficiency = span_efficiency_0 * span_efficiency_factor

        CDi = CL**2 / (np.pi * AR * span_efficiency)

        # If forward sweep, add Warner Robins Factor
        if SW25.real < 0.0:
            deg_to_rad = _units.degree

            TH = (1.0 - TR) / (1.0 + TR) / AR
            tan_sw = np.tan(SW25 / deg_to_rad)
            COSA = 1.0 / np.sqrt(1.0 + (tan_sw - 3.0 * TH) ** 2)
            COSB = 1.0 / np.sqrt(1.0 + (tan_sw + TH) ** 2)
            CAYT = (
                0.5
                * ((1.1 - 0.11 / (1.1 - mach * COSA)) / (1.1 - 0.11 / (1.1 - mach * COSB)) - 1.0)
                ** 2
            )

            CDi += CAYT * CL**2

        outputs['induced_drag_coeff'] = CDi

    def compute_partials(self, inputs, partials):
        options = self.options
        gamma = options['gamma']
        mach, lift, P, Sref, AR, span_efficiency_factor, SW25, TR = inputs.values()
        redux = self.options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION]

        if redux:
            sqrt_AR = np.sqrt(AR)
            span_efficiency_0 = 1.0 + 0.1 * AR * (0.4226 * sqrt_AR - 0.35 * TR - 0.143)
            dse0_dAR = 0.1 * (0.4226 * (sqrt_AR + 0.5 * AR / sqrt_AR) - 0.35 * TR - 0.143)
            dse0_dTR = -0.035 * AR

        else:
            span_efficiency_0 = 1.0

        if span_efficiency_factor <= 0.3:
            span_efficiency = span_efficiency_0 + span_efficiency_factor
        else:
            span_efficiency = span_efficiency_0 * span_efficiency_factor

        CL = 2.0 * lift / (Sref * gamma * P * mach**2)
        dCL_dL = 2.0 / (Sref * gamma * P * mach**2)
        dCL_dSref = -2.0 * lift / (Sref**2 * gamma * P * mach**2)
        dCL_dP = -2.0 * lift / (Sref * gamma * P**2 * mach**2)
        dCL_dmach = -4.0 * lift / (Sref * gamma * P * mach**3)

        dCDi_dCL = 2.0 * CL / (np.pi * AR * span_efficiency)
        dCDi_dAR = -(CL**2) / (np.pi * AR**2 * span_efficiency)
        dCDi_dspan = -(CL**2) / (np.pi * AR * span_efficiency**2)

        partials['induced_drag_coeff', Dynamic.Atmosphere.MACH] = dCDi_dCL * dCL_dmach
        partials['induced_drag_coeff', Dynamic.Vehicle.LIFT] = dCDi_dCL * dCL_dL
        partials['induced_drag_coeff', Dynamic.Atmosphere.STATIC_PRESSURE] = dCDi_dCL * dCL_dP
        partials['induced_drag_coeff', Aircraft.Wing.ASPECT_RATIO] = dCDi_dAR
        partials['induced_drag_coeff', Aircraft.Wing.SPAN_EFFICIENCY_FACTOR] = 0.0
        partials['induced_drag_coeff', Aircraft.Wing.SWEEP] = 0.0
        partials['induced_drag_coeff', Aircraft.Wing.TAPER_RATIO] = 0.0
        partials['induced_drag_coeff', Aircraft.Wing.AREA] = dCDi_dCL * dCL_dSref

        if span_efficiency_factor <= 0.3:
            partials['induced_drag_coeff', Aircraft.Wing.SPAN_EFFICIENCY_FACTOR] = dCDi_dspan
            if redux:
                partials['induced_drag_coeff', Aircraft.Wing.ASPECT_RATIO] += dCDi_dspan * dse0_dAR
                partials['induced_drag_coeff', Aircraft.Wing.TAPER_RATIO] = dCDi_dspan * dse0_dTR
        else:
            partials['induced_drag_coeff', Aircraft.Wing.SPAN_EFFICIENCY_FACTOR] = (
                dCDi_dspan * span_efficiency_0
            )
            if redux:
                partials['induced_drag_coeff', Aircraft.Wing.ASPECT_RATIO] += (
                    dCDi_dspan * dse0_dAR * span_efficiency_factor
                )
                partials['induced_drag_coeff', Aircraft.Wing.TAPER_RATIO] = (
                    dCDi_dspan * dse0_dTR * span_efficiency_factor
                )

        # If forward sweep, add Warner Robins Factor
        if SW25.real < 0.0:
            deg_to_rad = _units.degree

            fact1 = 1.0 - TR
            fact2 = 1.0 / (1.0 + TR)
            TH = fact1 * fact2 / AR
            dTH_dTR = -(fact2 + fact1 * fact2**2) / AR
            dTH_dAR = -fact1 * fact2 / AR**2

            tan_sw = np.tan(SW25 / deg_to_rad)
            dtansw_dsw = 1.0 / (deg_to_rad * np.cos(SW25 / deg_to_rad) ** 2)

            fact3 = 1.0 + (tan_sw - 3.0 * TH) ** 2
            COSA = 1.0 / np.sqrt(fact3)
            dCOSA_dtansw = -0.5 / fact3**1.5 * 2.0 * (tan_sw - 3.0 * TH)
            dCOSA_dTH = 0.5 / fact3**1.5 * 6.0 * (tan_sw - 3.0 * TH)

            fact4 = 1.0 + (tan_sw + TH) ** 2
            COSB = 1.0 / np.sqrt(fact4)
            dCOSB_dtansw = -0.5 / fact4**1.5 * 2.0 * (tan_sw + TH)
            dCOSB_dTH = -0.5 / fact4**1.5 * 2.0 * (tan_sw + TH)

            factA = 1.1 - mach * COSA
            factB = 1.1 - mach * COSB
            fact5 = 1.1 - 0.11 / factA
            fact6 = 1.1 - 0.11 / factB
            CAYT = 0.5 * (fact5 / fact6 - 1.0) ** 2
            dCAYT_dmach = (fact5 / fact6 - 1.0) * (
                -0.11 * COSA / (fact6 * factA**2) + 0.11 * fact5 * COSB / (factB**2 * fact6**2)
            )
            dCAYT_dCOSA = (fact5 / fact6 - 1.0) * (-0.11 * mach / (fact6 * factA**2))
            dCAYT_dCOSB = (fact5 / fact6 - 1.0) * (0.11 * fact5 * mach / (factB**2 * fact6**2))

            dCDi_dCAYT = CL**2
            dCDi_dCL = 2.0 * CAYT * CL

            partials['induced_drag_coeff', Dynamic.Atmosphere.MACH] += (
                dCDi_dCL * dCL_dmach + dCDi_dCAYT * dCAYT_dmach
            )
            partials['induced_drag_coeff', Dynamic.Vehicle.LIFT] += dCDi_dCL * dCL_dL
            partials['induced_drag_coeff', Aircraft.Wing.ASPECT_RATIO] += (
                dCDi_dCAYT * dTH_dAR * (dCAYT_dCOSA * dCOSA_dTH + dCAYT_dCOSB * dCOSB_dTH)
            )
            partials['induced_drag_coeff', Aircraft.Wing.SWEEP] += (
                dCDi_dCAYT * dtansw_dsw * (dCAYT_dCOSA * dCOSA_dtansw + dCAYT_dCOSB * dCOSB_dtansw)
            )
            partials['induced_drag_coeff', Dynamic.Atmosphere.STATIC_PRESSURE] += dCDi_dCL * dCL_dP
            partials['induced_drag_coeff', Aircraft.Wing.TAPER_RATIO] += (
                dCDi_dCAYT * dTH_dTR * (dCAYT_dCOSA * dCOSA_dTH + dCAYT_dCOSB * dCOSB_dTH)
            )
            partials['induced_drag_coeff', Aircraft.Wing.AREA] += dCDi_dCL * dCL_dSref
