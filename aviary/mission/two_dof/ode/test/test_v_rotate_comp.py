import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.two_dof.ode.v_rotate_comp import VRotateComp
from aviary.variable_info.variables import Aircraft


class TestVRotateComp(unittest.TestCase):
    """Test the computation of the speed at which takeoff rotation should be initiated."""

    def test_partials(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'vrot_comp', VRotateComp(), promotes_inputs=['*'], promotes_outputs=['*']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val('dV1', val=10, units='kn')
        prob.set_val('dVR', val=5, units='kn')
        prob.set_val(Aircraft.Wing.AREA, val=1370, units='ft**2')
        prob.set_val('density', val=0.0023769, units='slug/ft**3')
        prob.set_val('CL_max', val=2.1886, units='unitless')
        prob.set_val('mass', val=175_000, units='lbm')

        prob.run_model()

        # print(prob.get_val("Vrot", units="kn"))

        partials = prob.check_partials(method='cs', out_stream=None)
        assert_check_partials(partials, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
