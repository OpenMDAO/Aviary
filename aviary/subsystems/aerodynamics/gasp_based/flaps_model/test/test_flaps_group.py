import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.flaps_model import \
    FlapsGroup
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.enums import FlapType
from aviary.variable_info.variables import Aircraft, Dynamic

"""
All data is from validation files using standalone flaps model
"""


class FlapsGroupTestCaseTripleSlotted(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE,
                        val=FlapType.TRIPLE_SLOTTED, units='unitless')

        self.prob.model = FCC = FlapsGroup(aviary_options=options)

        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.67, units="degR")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.406626, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.25725)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 1.0000)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444322)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03512)
        self.prob.set_val("VLAM14", 0.99124)

        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 0.62455)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)

        self.prob.set_val("flap_defl_ratio", 0.72727273)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)

        self.prob.set_val("slat_defl_ratio", 0.5)
        self.prob.set_val("body_to_span_ratio", 0.09240447)
        self.prob.set_val("chord_to_body_ratio", 0.12679)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759553)

    def test_case(self):

        self.prob.run_model()
        tol = 6e-4  # checked. high tol for lack of precision in GASP data.

        reg_data = 2.8155
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.17522
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 157.1111
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.0406
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.0293
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(method="fd", out_stream=None)
        assert_check_partials(
            data, atol=6315, rtol=7e-3
        )  # tolerance at this value for special cases. All other values check out.


class FlapsGroupTestCaseSplit(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE, val=FlapType.SPLIT, units='unitless')

        self.prob.model = FCC = FlapsGroup(aviary_options=options)

        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.67, units="degR")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.406626, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.19742)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 0.8)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444322)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03209)
        self.prob.set_val("VLAM14", 0.99082)

        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 0.55667)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)

        self.prob.set_val("flap_defl_ratio", 0.6666667)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)

        self.prob.set_val("slat_defl_ratio", 0.5)
        self.prob.set_val("body_to_span_ratio", 0.09240447)
        self.prob.set_val("chord_to_body_ratio", 0.12679)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759553)

    def test_case(self):

        self.prob.run_model()
        tol = 9e-4  # checked. high tol for lack of precision in GASP data.

        reg_data = 2.56197
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.18368
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 164.78406
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.0362
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.7816
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(method="fd", out_stream=None)
        assert_check_partials(
            data, atol=6630, rtol=0.007
        )  # tolerance at this value for special cases. All other values check out.


class FlapsGroupTestCaseSingleSlotted(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE,
                        val=FlapType.SINGLE_SLOTTED, units='unitless')

        self.prob.model = FCC = FlapsGroup(aviary_options=options)

        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.67, units="degR")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION, 55.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.406626, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.25725)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 1.0000)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444322)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03512)
        self.prob.set_val("VLAM14", 0.99124)

        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 0.62455)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)

        self.prob.set_val("flap_defl_ratio", 0.72727273)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)

        self.prob.set_val("slat_defl_ratio", 0.5)
        self.prob.set_val("body_to_span_ratio", 0.09240447)
        self.prob.set_val("chord_to_body_ratio", 0.12679)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759553)

    def test_case(self):

        self.prob.run_model()
        tol = 6e-4  # checked. high tol for lack of precision in GASP data.

        reg_data = 2.8155
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.17522
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 157.1111
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.0406
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.0293
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(method="fd", out_stream=None)
        assert_check_partials(
            data, atol=6315, rtol=0.007
        )  # tolerance at this value for special cases. All other values check out.


class FlapsGroupTestCasePlain(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE, val=FlapType.PLAIN, units='unitless')

        self.prob.model = FCC = FlapsGroup(aviary_options=options)

        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.67, units="degR")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.406626, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.19742)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 0.8)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444322)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03209)
        self.prob.set_val("VLAM14", 0.99082)

        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 0.55667)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)

        self.prob.set_val("flap_defl_ratio", 0.6666667)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)

        self.prob.set_val("slat_defl_ratio", 0.5)
        self.prob.set_val("body_to_span_ratio", 0.09240447)
        self.prob.set_val("chord_to_body_ratio", 0.12679)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759553)

    def test_case(self):

        self.prob.run_model()
        tol = 9e-4  # checked. high tol for lack of precision in GASP data.

        reg_data = 2.56197
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.18368
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 164.78406
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.0362
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.7816
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(method="fd", out_stream=None)
        assert_check_partials(
            data, atol=6630, rtol=0.007
        )  # tolerance at this value for special cases. All other values check out.


class FlapsGroupTestCaseFowler(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE, val=FlapType.FOWLER, units='unitless')

        self.prob.model = FCC = FlapsGroup(aviary_options=options)

        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.67, units="degR")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.406626, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.25725)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 1.1100)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444322)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03639)
        self.prob.set_val("VLAM14", 0.99142)

        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 0.64667)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)

        self.prob.set_val("flap_defl_ratio", 1.333333)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)

        self.prob.set_val("slat_defl_ratio", 0.5)
        self.prob.set_val("body_to_span_ratio", 0.09240447)
        self.prob.set_val("chord_to_body_ratio", 0.12679)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759553)

    def test_case(self):

        self.prob.run_model()
        tol = 6e-4  # checked. high tol for lack of precision in GASP data.

        reg_data = 2.93271
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.17168
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 154.02686
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.1070
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.1441
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(method="fd", out_stream=None)
        assert_check_partials(
            data, atol=6200, rtol=0.007
        )  # tolerance at this value for special cases. All other values check out.


class FlapsGroupTestCaseDoubleFowler(unittest.TestCase):
    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FLAP_TYPE,
                        val=FlapType.DOUBLE_SLOTTED_FOWLER, units='unitless')

        self.prob.model = FCC = FlapsGroup(aviary_options=options)

        self.prob.setup()

        self.prob.set_val(Aircraft.Wing.SWEEP, 25.0, units="deg")
        self.prob.set_val(Dynamic.Mission.TEMPERATURE, 518.67, units="degR")
        self.prob.set_val(Aircraft.Wing.ASPECT_RATIO, 10.13)
        self.prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3)
        self.prob.set_val(Aircraft.Wing.TAPER_RATIO, 0.33)
        self.prob.set_val(Aircraft.Wing.CENTER_CHORD, 17.48974, units="ft")
        self.prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15)
        self.prob.set_val(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.set_val(Aircraft.Wing.SLAT_CHORD_RATIO, 0.15)
        self.prob.set_val("slat_defl", 10.0, units="deg")
        self.prob.set_val(Aircraft.Wing.OPTIMUM_SLAT_DEFLECTION, 20.0, units="deg")
        self.prob.set_val(Aircraft.Wing.ROOT_CHORD, 16.406626, units="ft")
        self.prob.set_val(Aircraft.Fuselage.LENGTH, 129.4, units="ft")
        self.prob.set_val(Aircraft.Wing.LEADING_EDGE_SWEEP, 0.47639, units="rad")

        self.prob.set_val("VLAM1", 0.97217)
        self.prob.set_val("VLAM2", 1.09948)
        self.prob.set_val("VLAM3", 0.97217)
        self.prob.set_val("VLAM4", 1.25725)
        self.prob.set_val("VLAM5", 1.0000)
        self.prob.set_val("VLAM6", 1.1100)
        self.prob.set_val("VLAM7", 0.735)
        self.prob.set_val("VLAM8", 0.74444322)
        self.prob.set_val("VLAM9", 0.9975)
        self.prob.set_val("VLAM10", 0.74)
        self.prob.set_val("VLAM11", 0.84232)
        self.prob.set_val("VLAM12", 0.79208)
        self.prob.set_val("VLAM13", 1.03639)
        self.prob.set_val("VLAM14", 0.99142)

        self.prob.set_val("VDEL1", 1.0)
        self.prob.set_val("VDEL2", 0.64667)
        self.prob.set_val("VDEL3", 0.765)
        self.prob.set_val("VDEL4", 0.93578)
        self.prob.set_val("VDEL5", 0.90761)

        self.prob.set_val(Dynamic.Mission.SPEED_OF_SOUND, 1118.21948771, units="ft/s")
        self.prob.set_val(Aircraft.Wing.LOADING, 128.0, units="lbf/ft**2")
        self.prob.set_val(Dynamic.Mission.STATIC_PRESSURE,
                          (14.696 * 144), units="lbf/ft**2")
        self.prob.set_val(Aircraft.Wing.AVERAGE_CHORD, 12.61, units="ft")
        self.prob.set_val("kinematic_viscosity", 0.15723e-3, units="ft**2/s")
        self.prob.set_val(Aircraft.Wing.MAX_LIFT_REF, 1.150)
        self.prob.set_val(Aircraft.Wing.SLAT_LIFT_INCREMENT_OPTIMUM, 0.930)
        self.prob.set_val("fus_lift", 0.05498)
        self.prob.set_val(Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM, 1.500)
        self.prob.set_val(Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM, 0.1)

        self.prob.set_val("flap_defl_ratio", 1.333333)
        self.prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13966)
        self.prob.set_val("flap_defl", 40.0, units="deg")
        self.prob.set_val(Aircraft.Wing.FLAP_SPAN_RATIO, 0.65)

        self.prob.set_val("slat_defl_ratio", 0.5)
        self.prob.set_val("body_to_span_ratio", 0.09240447)
        self.prob.set_val("chord_to_body_ratio", 0.12679)
        self.prob.set_val(Aircraft.Wing.SLAT_SPAN_RATIO, 0.89759553)

    def test_case(self):

        self.prob.run_model()
        tol = 6e-4  # checked. high tol for lack of precision in GASP data.

        reg_data = 2.93271
        ans = self.prob["CL_max"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.17168
        ans = self.prob[Dynamic.Mission.MACH]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 154.02686
        ans = self.prob["reynolds"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 0.1070
        ans = self.prob["delta_CD"]
        assert_near_equal(ans, reg_data, tol)

        reg_data = 1.1441
        ans = self.prob["delta_CL"]
        assert_near_equal(ans, reg_data, tol)

        data = self.prob.check_partials(method="fd", out_stream=None)
        assert_check_partials(
            data, atol=6200, rtol=0.007
        )  # tolerance at this value for special cases. All other values check out.


if __name__ == "__main__":
    unittest.main()
