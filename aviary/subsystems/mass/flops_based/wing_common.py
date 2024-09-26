import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class WingBendingMass(om.ExplicitComponent):
    '''
    Calculates the mass of wing bending material. The methodology is
    based on the FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, val=0.0)
        add_aviary_input(self, Aircraft.Wing.BENDING_FACTOR, val=0.0)
        add_aviary_input(self, Aircraft.Wing.BENDING_MASS_SCALER, val=1.0)
        add_aviary_input(self, Aircraft.Wing.COMPOSITE_FRACTION, val=0.0)
        add_aviary_input(self, Aircraft.Wing.ENG_POD_INERTIA_FACTOR, val=0.0)
        add_aviary_input(self, Aircraft.Wing.LOAD_FRACTION, val=0.0)
        add_aviary_input(self, Aircraft.Wing.MISC_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Wing.MISC_MASS_SCALER, val=1.0)
        add_aviary_input(self, Aircraft.Wing.SHEAR_CONTROL_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, val=1.0)
        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)
        add_aviary_input(self, Aircraft.Wing.SWEEP, val=0.0)
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.75)
        add_aviary_input(self, Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, val=0.0)

        add_aviary_output(self, Aircraft.Wing.BENDING_MASS, val=0.0)

        self.A1 = 8.80
        self.A2 = 6.25

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        bt = inputs[Aircraft.Wing.BENDING_FACTOR]
        ulf = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        span = inputs[Aircraft.Wing.SPAN]
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        faert = inputs[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR]
        varswp = inputs[Aircraft.Wing.VAR_SWEEP_MASS_PENALTY]
        pctl = inputs[Aircraft.Wing.LOAD_FRACTION]
        sweep = inputs[Aircraft.Wing.SWEEP]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        CAYE = inputs[Aircraft.Wing.ENG_POD_INERTIA_FACTOR]
        scaler = inputs[Aircraft.Wing.BENDING_MASS_SCALER]

        num_fuse = self.options['aviary_options'].get_val(
            Aircraft.Fuselage.NUM_FUSELAGES)

        # Note: Calculation requires weights prior to being scaled, so we need to divide
        # by the scale factor.
        W2 = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS] / \
            inputs[Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER] * GRAV_ENGLISH_LBM
        W3 = inputs[Aircraft.Wing.MISC_MASS] / \
            inputs[Aircraft.Wing.MISC_MASS_SCALER] * GRAV_ENGLISH_LBM

        vfact = 1.0 + varswp * (0.96 / np.cos(np.pi / 180. * sweep) - 1.0)
        cayf = 0.5 if num_fuse > 1 else 1.0

        W1NIR = self.A1 * bt * (1.0 + (self.A2 / span)**0.5) * ulf * span * \
            (1.0 - 0.4 * comp_frac) * (1.0 - 0.1 * faert) * cayf * vfact * pctl * 1.0e-6

        outputs[Aircraft.Wing.BENDING_MASS] = (
            (gross_weight * CAYE * W1NIR + W2 + W3) / (1.0 + W1NIR) - W2 - W3) \
            * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        bt = inputs[Aircraft.Wing.BENDING_FACTOR]
        ulf = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        span = inputs[Aircraft.Wing.SPAN]
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        faert = inputs[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR]
        varswp = inputs[Aircraft.Wing.VAR_SWEEP_MASS_PENALTY]
        pctl = inputs[Aircraft.Wing.LOAD_FRACTION]
        sweep = inputs[Aircraft.Wing.SWEEP]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        CAYE = inputs[Aircraft.Wing.ENG_POD_INERTIA_FACTOR]
        W2 = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS] * GRAV_ENGLISH_LBM
        W3 = inputs[Aircraft.Wing.MISC_MASS] * GRAV_ENGLISH_LBM
        W2scale = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER]
        W3scale = inputs[Aircraft.Wing.MISC_MASS_SCALER]
        scaler = inputs[Aircraft.Wing.BENDING_MASS_SCALER]

        num_fuse = self.options['aviary_options'].get_val(
            Aircraft.Fuselage.NUM_FUSELAGES)

        deg2rad = np.pi / 180.
        term = 0.96 / np.cos(deg2rad * sweep)
        vfact = 1.0 + varswp * (term - 1.0)
        dvfact_varswp = (term - 1.0)
        dvfact_sweep = varswp * deg2rad * term * np.tan(deg2rad * sweep)
        cayf = 0.5 if num_fuse > 1 else 1.0

        fact0 = (self.A2 / span)**0.5
        fact1 = 1.0 + fact0
        fact2 = 1.0 - 0.4 * comp_frac
        fact3 = 1.0 - 0.1 * faert
        W1NIR = (
            self.A1 * bt * fact1 * ulf * span * fact2 * fact3 * cayf * vfact * pctl
            * 1.0e-6)
        dW1NIR_bt = \
            self.A1 * fact1 * ulf * span * fact2 * fact3 * cayf * vfact * pctl * 1.0e-6
        dW1NIR_ulf = \
            self.A1 * bt * fact1 * span * fact2 * fact3 * cayf * vfact * pctl * 1.0e-6
        dW1NIR_pctl = \
            self.A1 * bt * fact1 * ulf * span * fact2 * fact3 * cayf * vfact * 1.0e-6
        dW1NIR_cayf = \
            self.A1 * bt * fact1 * ulf * span * fact2 * fact3 * vfact * pctl * 1.0e-6
        dW1NIR_compfrac = -self.A1 * bt * fact1 * ulf * \
            span * 0.4 * fact3 * cayf * vfact * pctl * 1.0e-6
        dW1NIR_faert = -self.A1 * bt * fact1 * ulf * \
            span * fact2 * 0.1 * cayf * vfact * pctl * 1.0e-6
        dW1NIR_varswp = self.A1 * bt * fact1 * ulf * span * \
            fact2 * fact3 * cayf * dvfact_varswp * pctl * 1.0e-6
        dW1NIR_sweep = self.A1 * bt * fact1 * ulf * span * \
            fact2 * fact3 * cayf * dvfact_sweep * pctl * 1.0e-6

        dfact1_span = -0.5 * self.A2 / (fact0 * span * span)
        dW1NIR_span = self.A1 * bt * \
            (dfact1_span * span + fact1) * ulf * \
            fact2 * fact3 * cayf * vfact * pctl * 1.0e-6

        fact1 = gross_weight * CAYE * W1NIR + W2/W2scale + W3/W3scale
        fact2 = 1.0 / (1.0 + W1NIR)
        dbend_w1nir = scaler * (gross_weight * CAYE * fact2 - fact1 * fact2**2)

        J[Aircraft.Wing.BENDING_MASS, Mission.Design.GROSS_MASS] = \
            CAYE * W1NIR * fact2 * scaler

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.ENG_POD_INERTIA_FACTOR] = \
            gross_weight * W1NIR * fact2 * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.SHEAR_CONTROL_MASS] = \
            (fact2 - 1.0) * scaler / W2scale

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER] = \
            -(fact2 - 1.0) * scaler * W2 / W2scale ** 2 / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.MISC_MASS] = \
            (fact2 - 1.0) * scaler / W3scale

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.MISC_MASS_SCALER] = \
            -(fact2 - 1.0) * scaler * W3 / W3scale ** 2 / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.BENDING_MASS_SCALER] = \
            (fact1 * fact2 - W2/W2scale - W3/W3scale) / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.BENDING_FACTOR] = \
            dbend_w1nir * dW1NIR_bt / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = \
            dbend_w1nir * dW1NIR_ulf / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.LOAD_FRACTION] = \
            dbend_w1nir * dW1NIR_pctl / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.COMPOSITE_FRACTION] = \
            dbend_w1nir * dW1NIR_compfrac / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR] = \
            dbend_w1nir * dW1NIR_faert / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.VAR_SWEEP_MASS_PENALTY] = \
            dbend_w1nir * dW1NIR_varswp / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.SWEEP] = \
            dbend_w1nir * dW1NIR_sweep / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.BENDING_MASS, Aircraft.Wing.SPAN] = \
            dbend_w1nir * dW1NIR_span / GRAV_ENGLISH_LBM


class WingShearControlMass(om.ExplicitComponent):
    '''
    Calculates the mass of wing shear control material. The methodology is
    based on the FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

        self.options.declare(
            'aircraft_type',
            default='Transport',
            values=['Transport', 'HWB', 'GA'],
            desc='Aircfaft type: Tranpsport, HWB, or GA')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.COMPOSITE_FRACTION, val=0.0)

        add_aviary_input(self, Aircraft.Wing.CONTROL_SURFACE_AREA, val=0.0)

        add_aviary_input(self, Mission.Design.GROSS_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.Wing.SHEAR_CONTROL_MASS, val=0.0)

        if (
            (self.options['aircraft_type'] == 'Transport')
            or (self.options['aircraft_type'] == 'HWB')
        ):
            self.A3 = 0.68
            self.A4 = 0.34
            self.A5 = 0.60
        elif self.options['aircraft_type'] == 'GA':
            self.A3 = 0.25
            self.A4 = 0.50
            self.A5 = 0.50

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        ctrl_area = inputs[Aircraft.Wing.CONTROL_SURFACE_AREA]
        scaler = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER]

        outputs[Aircraft.Wing.SHEAR_CONTROL_MASS] = \
            self.A3 * (1.0 - 0.17 * comp_frac) * ctrl_area ** self.A4 * \
            gross_weight ** self.A5 * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        ctrl_area = inputs[Aircraft.Wing.CONTROL_SURFACE_AREA]
        scaler = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER]

        comp_frac_term = self.A3 * (1.0 - 0.17 * comp_frac)

        J[Aircraft.Wing.SHEAR_CONTROL_MASS, Aircraft.Wing.COMPOSITE_FRACTION] = \
            -0.17 * self.A3 * ctrl_area ** self.A4 * \
            gross_weight ** self.A5 * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.SHEAR_CONTROL_MASS, Aircraft.Wing.CONTROL_SURFACE_AREA] = \
            comp_frac_term * self.A4 * \
            ctrl_area ** (self.A4-1.0) * gross_weight ** self.A5 * \
            scaler / GRAV_ENGLISH_LBM

        J[Aircraft.Wing.SHEAR_CONTROL_MASS, Mission.Design.GROSS_MASS] = \
            comp_frac_term * ctrl_area ** self.A4 * self.A5 * \
            gross_weight ** (self.A5-1.0) * scaler

        J[
            Aircraft.Wing.SHEAR_CONTROL_MASS,
            Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER] = \
            self.A3 * (1.0 - 0.17 * comp_frac) * \
            ctrl_area ** self.A4 * gross_weight ** self.A5 / GRAV_ENGLISH_LBM


class WingMiscMass(om.ExplicitComponent):
    '''
    Calculates the mass of wing miscellaneous material. The methodology is
    based on the FLOPS weight equations, modified to output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

        self.options.declare(
            'aircraft_type',
            default='Transport',
            values=['Transport', 'HWB', 'GA'],
            desc='Aircfaft type: Tranpsport, HWB, or GA')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.COMPOSITE_FRACTION, val=0.0)

        add_aviary_input(self, Aircraft.Wing.AREA, val=0.0)

        add_aviary_input(self, Aircraft.Wing.MISC_MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.Wing.MISC_MASS, val=0.0)

        if (
            (self.options['aircraft_type'] == 'Transport')
            or (self.options['aircraft_type'] == 'HWB')
        ):
            self.A6 = 0.035
            self.A7 = 1.50
        elif self.options['aircraft_type'] == 'GA':
            self.A6 = 0.16
            self.A7 = 1.2

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        area = inputs[Aircraft.Wing.AREA]
        scaler = inputs[Aircraft.Wing.MISC_MASS_SCALER]

        outputs[Aircraft.Wing.MISC_MASS] = self.A6 * \
            (1.0 - 0.3 * comp_frac) * area ** self.A7 * \
            scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        area = inputs[Aircraft.Wing.AREA]
        scaler = inputs[Aircraft.Wing.MISC_MASS_SCALER]

        J[Aircraft.Wing.MISC_MASS, Aircraft.Wing.COMPOSITE_FRACTION] = - \
            0.3 * self.A6 * area ** self.A7 * scaler / GRAV_ENGLISH_LBM
        J[Aircraft.Wing.MISC_MASS, Aircraft.Wing.AREA] = self.A6 * \
            (1.0 - 0.3 * comp_frac) * self.A7 * \
            area ** (self.A7-1) * scaler / GRAV_ENGLISH_LBM
        J[Aircraft.Wing.MISC_MASS, Aircraft.Wing.MISC_MASS_SCALER] = self.A6 * \
            (1.0 - 0.3 * comp_frac) * area ** self.A7 / GRAV_ENGLISH_LBM


class WingTotalMass(om.ExplicitComponent):
    """
    Computation of wing mass using FLOPS-based detailed wing mass equations.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.BENDING_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SHEAR_CONTROL_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Wing.MISC_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Wing.BWB_AFTBODY_MASS, val=0.0)

        add_aviary_input(self, Aircraft.Wing.MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.Wing.MASS, val=0)

    def setup_partials(self):
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        m1 = inputs[Aircraft.Wing.BENDING_MASS]
        m2 = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS]
        m3 = inputs[Aircraft.Wing.MISC_MASS]
        m4 = inputs[Aircraft.Wing.BWB_AFTBODY_MASS]
        m_scaler = inputs[Aircraft.Wing.MASS_SCALER]

        outputs[Aircraft.Wing.MASS] = (m1 + m2 + m3 + m4) * m_scaler

    def compute_partials(self, inputs, J):
        m1 = inputs[Aircraft.Wing.BENDING_MASS]
        m2 = inputs[Aircraft.Wing.SHEAR_CONTROL_MASS]
        m3 = inputs[Aircraft.Wing.MISC_MASS]
        m4 = inputs[Aircraft.Wing.BWB_AFTBODY_MASS]
        m_scaler = inputs[Aircraft.Wing.MASS_SCALER]

        J[Aircraft.Wing.MASS, Aircraft.Wing.BENDING_MASS] = m_scaler
        J[Aircraft.Wing.MASS, Aircraft.Wing.SHEAR_CONTROL_MASS] = m_scaler
        J[Aircraft.Wing.MASS, Aircraft.Wing.MISC_MASS] = m_scaler
        J[Aircraft.Wing.MASS, Aircraft.Wing.BWB_AFTBODY_MASS] = m_scaler
        J[Aircraft.Wing.MASS, Aircraft.Wing.MASS_SCALER] = m1 + m2 + m3 + m4
