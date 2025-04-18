import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.flops_based.induced_drag import InducedDrag
from aviary.variable_info.variables import Aircraft, Dynamic


class InducedDragTest(unittest.TestCase):
    def test_derivs(self):
        P = 2.60239151
        Sref = 1370.0

        CL = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
        mach = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.85])
        lift = 0.5 * CL * Sref * 1.4 * P * mach**2

        nn = len(CL)

        prob = om.Problem()

        options = {}
        options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION] = False

        prob.model.add_subsystem(
            'induced_drag', InducedDrag(num_nodes=nn, **options), promotes=['*']
        )
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Atmosphere.MACH, val=mach)
        prob.set_val(Dynamic.Vehicle.LIFT, val=lift)
        prob.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P)
        prob.set_val(Aircraft.Wing.AREA, val=Sref)
        prob.set_val(Aircraft.Wing.SWEEP, val=-25.03)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.278)
        prob.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 0.7)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-12, rtol=8e-12)

        assert_near_equal(
            prob.get_val('induced_drag_coeff'),
            [0.00370367, 0.00504111, 0.0065843, 0.00833326, 0.01028797, 0.01244845],
            1e-6,
        )

    def test_derivs_span_eff_redux(self):
        P = 2.60239151
        Sref = 1370.0

        CL = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
        mach = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.85])
        lift = 0.5 * CL * Sref * 1.4 * P * mach**2

        nn = len(CL)

        # High factor

        prob = om.Problem()

        options = {}
        options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION] = True

        prob.model.add_subsystem('drag', InducedDrag(num_nodes=nn, **options), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Atmosphere.MACH, val=mach)
        prob.set_val(Dynamic.Vehicle.LIFT, val=lift)
        prob.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P)
        prob.set_val(Aircraft.Wing.AREA, val=Sref)
        prob.set_val(Aircraft.Wing.SWEEP, val=-25.10)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.312)
        prob.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 0.528)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-12, rtol=8e-12)

        assert_near_equal(
            prob.get_val('induced_drag_coeff'),
            [0.00216084, 0.00294208, 0.00384454, 0.00486925, 0.00601801, 0.00748097],
            1e-6,
        )

        # Low factor.

        prob = om.Problem(model=om.Group())

        options = {}
        options[Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION] = True

        prob.model.add_subsystem('drag', InducedDrag(num_nodes=nn, **options), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Atmosphere.MACH, val=mach)
        prob.set_val(Dynamic.Vehicle.LIFT, val=lift)
        prob.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P)
        prob.set_val(Aircraft.Wing.AREA, val=Sref)
        prob.set_val(Aircraft.Wing.SWEEP, val=-25.10)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.312)
        prob.set_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, 0.528)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(derivs, atol=1e-12, rtol=8e-12)

        assert_near_equal(
            prob.get_val('induced_drag_coeff'),
            [0.00216084, 0.00294208, 0.00384454, 0.00486925, 0.00601801, 0.00748097],
            1e-6,
        )


if __name__ == '__main__':
    unittest.main()
