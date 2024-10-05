import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class SimpleWingBendingFact(om.ExplicitComponent):
    """
    Simplified computation of wing bending factor and engine inertia relief factor
    for FLOPS-based mass.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SPAN, val=0.0)

        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, val=0.0)

        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.Wing.STRUT_BRACING_FACTOR, val=0.0)

        add_aviary_input(self, Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, val=0.0)

        add_aviary_input(self, Aircraft.Wing.ASPECT_RATIO, val=0.0)

        add_aviary_input(self, Aircraft.Wing.SWEEP, val=0.0)

        add_aviary_output(self, Aircraft.Wing.BENDING_FACTOR, val=0.0)

        add_aviary_output(self, Aircraft.Wing.ENG_POD_INERTIA_FACTOR, val=0.0)

    def setup_partials(self):
        self.declare_partials(of=Aircraft.Wing.BENDING_FACTOR,
                              wrt=[Aircraft.Wing.STRUT_BRACING_FACTOR,
                                   Aircraft.Wing.SPAN,
                                   Aircraft.Wing.TAPER_RATIO,
                                   Aircraft.Wing.AREA,
                                   Aircraft.Wing.THICKNESS_TO_CHORD,
                                   Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR,
                                   Aircraft.Wing.ASPECT_RATIO,
                                   Aircraft.Wing.SWEEP])

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_wing_eng = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)
        fstrt = inputs[Aircraft.Wing.STRUT_BRACING_FACTOR]
        span = inputs[Aircraft.Wing.SPAN]
        tr = inputs[Aircraft.Wing.TAPER_RATIO]
        area = inputs[Aircraft.Wing.AREA]
        tca = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        faert = inputs[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR]
        ar = inputs[Aircraft.Wing.ASPECT_RATIO]
        sweep = inputs[Aircraft.Wing.SWEEP]

        C4 = 1.0 - 0.5 * faert
        C6 = 0.5 * faert - 0.16 * fstrt

        if ar <= 5.0:
            caya = 0.0
        else:
            caya = ar - 5.0

        tlam = np.tan(np.pi / 180. * sweep) - 2 * (1 - tr) / (ar * (1 + tr))

        slam = tlam / (1.0 + tlam**2)**0.5

        cayl = (1.0 - slam**2) * (1.0 + C6 * slam**2 + 0.03 * caya * C4 * slam)

        ems = 1.0 - 0.25 * fstrt

        outputs[Aircraft.Wing.BENDING_FACTOR] = \
            0.215 * (0.37 + 0.7 * tr) * (span**2 / area)**ems / (cayl * tca)

        outputs[Aircraft.Wing.ENG_POD_INERTIA_FACTOR] = 1.0 - 0.03 * num_wing_eng

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_wing_eng = aviary_options.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)
        fstrt = inputs[Aircraft.Wing.STRUT_BRACING_FACTOR]
        span = inputs[Aircraft.Wing.SPAN]
        tr = inputs[Aircraft.Wing.TAPER_RATIO]
        area = inputs[Aircraft.Wing.AREA]
        tca = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
        faert = inputs[Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR]
        ar = inputs[Aircraft.Wing.ASPECT_RATIO]
        sweep = inputs[Aircraft.Wing.SWEEP]

        C4 = 1.0 - 0.5 * faert
        C6 = 0.5 * faert - 0.16 * fstrt

        if ar <= 5.0:
            caya = 0.0
            dcayl_ar = 0.0
        else:
            caya = ar - 5.0

        deg2rad = np.pi / 180.
        den = ar * (1 + tr)
        tlam = np.tan(deg2rad * sweep) - 2 * (1 - tr) / den
        dtlam_sweep = deg2rad / np.cos(deg2rad * sweep) ** 2
        dtlam_ar = 2.0 * (1 - tr) / (ar * den)
        dtlam_tr = 2.0 * (1.0 / den + (1 - tr) / den**2 * ar)

        den = 1.0 + tlam**2
        slam = tlam / den**0.5
        dslam = 1.0 / den**0.5 - tlam**2 / den**1.5

        term1 = 1.0 - slam**2
        term2 = 1.0 + C6 * slam**2 + 0.03 * caya * C4 * slam
        cayl = term1 * term2
        dcayl_slam = -2.0 * slam * term2 + term1 * (2.0 * slam * C6 + 0.03 * caya * C4)
        dcayl_faert = term1 * slam * 0.5 * (slam - 0.03 * caya)
        dcayl_fstrt = -term1 * 0.16 * slam**2
        if ar > 5.0:
            dcayl_ar = term1 * 0.03 * C4 * slam

        ems = 1.0 - 0.25 * fstrt

        # bend = 0.215 * (0.37 + 0.7 * tr) * (span**2 / area)**ems / (cayl * tca)
        term1 = (0.37 + 0.7 * tr)
        term2a = (span**2 / area)
        term2 = term2a ** ems
        term3 = 1.0 / (cayl * tca)
        dbend_exp = -0.215 * term1 * term2 * term3 * np.log(term2a) * 0.25
        dbend_tr = 0.215 * 0.7 * term2 * term3
        dbend_cayl = -0.215 * term1 * term2 * tca * term3**2

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.STRUT_BRACING_FACTOR] = \
            dbend_exp + dbend_cayl * dcayl_fstrt

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.SPAN] = \
            2.0 * 0.215 * term1 * ems * term2a**(ems - 1) * term3 * span / area

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.TAPER_RATIO] = \
            dbend_tr + dbend_cayl * (dcayl_slam * dslam * dtlam_tr)

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.AREA] = \
            -0.215 * term1 * ems * term2a**(ems - 1) * term3 * term2a / area

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.THICKNESS_TO_CHORD] = \
            -0.215 * term1 * term2 * cayl * term3**2

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR] = \
            dbend_cayl * dcayl_faert

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.ASPECT_RATIO] = \
            dbend_cayl * (dcayl_ar + dcayl_slam * dslam * dtlam_ar)

        J[Aircraft.Wing.BENDING_FACTOR, Aircraft.Wing.SWEEP] = \
            dbend_cayl * (dcayl_slam * dslam * dtlam_sweep)
