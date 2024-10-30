import unittest

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.interference import RootChord, \
    CommonVariables, TopAndBottomWidth, BodyRatios, InterferenceDrag, \
    WingFuselageInterferencePremission, WingFuselageInterferenceMission
from aviary.variable_info.variables import Aircraft, Dynamic


tol = 1e-6


class TestPreMissionSubComponents(unittest.TestCase):
    def test_root_chord(self):
        prob = om.Problem()
        prob.model.add_subsystem("comp", RootChord(), promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.AREA, 1400)
        prob.set_val(Aircraft.Wing.SPAN, 118)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, .9)

        prob.run_model()

        assert_near_equal(prob.get_val('CROOT'), [12.48884924], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)

    def test_common_vars(self):
        prob = om.Problem()
        prob.model.add_subsystem("comp", CommonVariables(), promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val('CROOT', 12)
        prob.set_val(Aircraft.Wing.MOUNTING_TYPE, .1)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, .12)
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 10)

        prob.run_model()

        assert_near_equal(prob.get_val('ZW_RF'), [-.8], tol)
        assert_near_equal(prob.get_val('wtofd'), [.144], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)

    def test_top_and_bottom(self):
        prob = om.Problem()
        prob.model.add_subsystem("comp", TopAndBottomWidth(), promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val('wtofd', .14)
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 10)
        for z in (-1, 1):
            prob.set_val('ZW_RF', z)

            prob.run_model()

            assert_near_equal(prob.get_val('WBODYWF'), [2.55147016], tol)

            partial_data = prob.check_partials(method="cs", out_stream=None)
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)

    def test_body_ratios(self):
        prob = om.Problem()
        prob.model.add_subsystem("comp", BodyRatios(), promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val('WBODYWF', 2.5)
        prob.set_val('CROOT', 12)
        prob.set_val(Aircraft.Wing.SPAN, 118)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, .9)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, .12)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, .1)

        prob.run_model()

        assert_near_equal(prob.get_val('TCBODYWF'), [0.11957627], tol)
        assert_near_equal(prob.get_val('CBODYWF'), [11.974576], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)

    def test_interference_drag(self):
        prob = om.Problem()
        prob.model.add_subsystem("comp", InterferenceDrag(), promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val('WBODYWF', 2.5)
        prob.set_val('CROOT', 12)
        prob.set_val('TCBODYWF', .1)
        prob.set_val('CBODYWF', 12)
        prob.set_val('ZW_RF', .5)
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 10)
        prob.set_val(Aircraft.Wing.CENTER_DISTANCE, .6)

        prob.run_model()

        assert_near_equal(prob.get_val(
            'interference_independent_of_shielded_area'), [0.05654201], tol)
        assert_near_equal(prob.get_val('drag_loss_due_to_shielded_wing_area'), [30], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)


class TestPreMission(unittest.TestCase):
    def test_complete_group(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            "comp", WingFuselageInterferencePremission(),
            promotes=["aircraft:*",
                      'interference_independent_of_shielded_area',
                      'drag_loss_due_to_shielded_wing_area'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.AREA, 1400)
        prob.set_val(Aircraft.Wing.SPAN, 118)
        prob.set_val(Aircraft.Wing.TAPER_RATIO, .9)
        prob.set_val(Aircraft.Wing.MOUNTING_TYPE, .1)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, .12)
        prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, .1)
        prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 12)
        prob.set_val(Aircraft.Wing.CENTER_DISTANCE, .6)

        prob.run_model()

        assert_near_equal(prob.get_val(
            'interference_independent_of_shielded_area'), [0.35794891], tol)
        assert_near_equal(prob.get_val(
            'drag_loss_due_to_shielded_wing_area'), [83.53366], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)


class TestMission(unittest.TestCase):
    def test_complete_group(self):
        nn = 2
        prob = om.Problem()
        prob.model.add_subsystem(
            "atmos",
            USatm1976Comp(num_nodes=nn),
            promotes_inputs=[("h", Dynamic.Mission.ALTITUDE)],
            promotes_outputs=['rho', "viscosity",
                              ("temp", Dynamic.Mission.TEMPERATURE)],
        )
        prob.model.add_subsystem(
            "kin_visc",
            om.ExecComp(
                "nu = viscosity / rho",
                viscosity={"units": "lbf*s/ft**2", "shape": nn},
                rho={"units": "slug/ft**3", "shape": nn},
                nu={"units": "ft**2/s", "shape": nn},
                has_diag_partials=True,
            ),
            promotes=["*", ('nu', Dynamic.Mission.KINEMATIC_VISCOSITY)],
        )
        prob.model.add_subsystem(
            "comp", WingFuselageInterferenceMission(num_nodes=nn),
            promotes=["*"])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.FORM_FACTOR, 1.25)
        prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12)
        prob.set_val(Dynamic.Mission.MACH, (.6, .65))
        prob.set_val(Dynamic.Mission.ALTITUDE, (30000, 30000))
        prob.set_val('interference_independent_of_shielded_area', 0.35794891)
        prob.set_val('drag_loss_due_to_shielded_wing_area', 83.53366)

        prob.run_model()

        assert_near_equal(prob.get_val('wing_fuselage_interference_flat_plate_equivalent'), [
                          83.53249732, 83.53251792], tol)

        partial_data = prob.check_partials(method="cs", out_stream=None)
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
