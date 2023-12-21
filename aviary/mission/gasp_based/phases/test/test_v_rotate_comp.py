import unittest

import openmdao.api as om

from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.mission.gasp_based.phases.v_rotate_comp import VRotateComp
from aviary.variable_info.variables import Aircraft


class TestVRotateComp(unittest.TestCase):

    def test_partials(self):
        prob = om.Problem()

        prob.model.add_subsystem("vrot_comp", VRotateComp(), promotes_inputs=[
                                 "*"], promotes_outputs=["*"])

        prob.setup(force_alloc_complex=True)

        prob.set_val("dV1", val=10, units="kn")
        prob.set_val("dVR", val=5, units="kn")
        prob.set_val(Aircraft.Wing.AREA, val=1370, units="ft**2")
        prob.set_val("rho", val=RHO_SEA_LEVEL_ENGLISH, units="slug/ft**3")
        prob.set_val("CL_max", val=2.1886, units="unitless")
        prob.set_val("mass", val=175_000, units="lbm")

        prob.run_model()

        # print(prob.get_val("Vrot", units="kn"))

        prob.check_partials(method='cs')
