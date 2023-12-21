import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.subsystems.aerodynamics.flops_based.lift_dependent_drag import \
    LiftDependentDrag
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class LiftDependentDragTest(unittest.TestCase):

    def test_derivs_edge_interp(self):

        # Pressure in lbf/in**2 at 41000 ft.
        P = 2.60239151
        Sref = 1370.0

        CL = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
        mach = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.85])
        lift = 0.5 * CL * Sref * 1.4 * P * mach ** 2

        nn = len(CL)

        prob = om.Problem(model=om.Group())
        prob.model.add_subsystem('drag', LiftDependentDrag(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Mission.MACH, val=mach)
        prob.set_val(Dynamic.Mission.LIFT, val=lift)
        prob.set_val(Dynamic.Mission.STATIC_PRESSURE, val=P)
        prob.set_val(Aircraft.Wing.AREA, val=Sref)
        prob.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, val=1.0)
        prob.set_val(Aircraft.Wing.SWEEP, val=25.03)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05)

        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.123)
        prob.set_val(Mission.Design.LIFT_COEFFICIENT, val=1.28)
        prob.set_val(Mission.Design.MACH, val=0.765)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method="cs")

        # TODO: need to test outputs too
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)

    def test_derivs_inner_interp(self):

        # Pressure in lbf/in**2 at 41000 ft.
        P = 2.60239151
        Sref = 1370.0

        CL = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
        mach = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.85])
        lift = 0.5 * CL * Sref * 1.4 * P * mach ** 2

        nn = len(CL)

        prob = om.Problem(model=om.Group())
        prob.model.add_subsystem('drag', LiftDependentDrag(num_nodes=nn), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Dynamic.Mission.MACH, val=mach)
        prob.set_val(Dynamic.Mission.LIFT, val=lift)
        prob.set_val(Dynamic.Mission.STATIC_PRESSURE, val=P)
        prob.set_val(Aircraft.Wing.AREA, val=Sref)
        prob.set_val(Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN, val=1.0)
        prob.set_val(Aircraft.Wing.SWEEP, val=25.07)

        prob.set_val(Aircraft.Wing.ASPECT_RATIO, val=11.05 * 0.5)

        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, val=0.132)
        prob.set_val(Mission.Design.LIFT_COEFFICIENT, val=0.1234)
        prob.set_val(Mission.Design.MACH, val=0.4321)

        prob.run_model()

        derivs = prob.check_partials(out_stream=None, method="cs")

        # TODO: need to test outputs too
        assert_check_partials(derivs, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
