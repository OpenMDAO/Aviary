import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.variable_info.variables import Aircraft, Dynamic


class TestVRotateComp(unittest.TestCase):
    """
    Test the computation of the speed at which takeoff rotation should be initiated
    """

    def test_partials(self):
        prob = om.Problem()

        prob.model.add_subsystem("vrot_comp", VRotateComp(), promotes_inputs=[
                                 "*"], promotes_outputs=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("dV1", val=10, units="kn")
        prob.set_val("dVR", val=5, units="kn")
        prob.set_val(Aircraft.Wing.AREA, val=1370, units="ft**2")
        prob.set_val(
            Dynamic.Mission.DENSITY, val=RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3"
        )
        prob.set_val("CL_max", val=2.1886, units="unitless")
        prob.set_val("mass", val=175_000, units="lbm")

        prob.run_model()

        # print(prob.get_val("Vrot", units="kn"))

        prob.check_partials(method='cs')


class TestVRotateComp2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.mission.gasp_based.phases.v_rotate_comp as vr
        vr.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.mission.gasp_based.phases.v_rotate_comp as vr
        vr.GRAV_ENGLISH_LBM = 1.0

    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem("vrot_comp", VRotateComp(), promotes_inputs=[
                                 "*"], promotes_outputs=["*"])
        prob.setup(force_alloc_complex=True)
        prob.set_val("dV1", val=10, units="kn")
        prob.set_val("dVR", val=5, units="kn")
        prob.set_val(Aircraft.Wing.AREA, val=1370, units="ft**2")
        prob.set_val(
            Dynamic.Mission.DENSITY, val=RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3"
        )
        prob.set_val("CL_max", val=2.1886, units="unitless")
        prob.set_val("mass", val=175_000, units="lbm")

        partial_data = prob.check_partials(method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)
