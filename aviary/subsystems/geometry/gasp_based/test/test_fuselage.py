import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.fuselage import (FuselageGroup,
                                                            FuselageParameters,
                                                            FuselageSize)
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft


class FuselageParametersTestCase1(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "parameters",
            FuselageParameters(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(self.prob["cabin_height"], 13.1, tol)
        assert_near_equal(self.prob["cabin_len"], 72.1, tol)
        assert_near_equal(self.prob["nose_height"], 8.6, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageParametersTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "parameters",
            FuselageParameters(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case2(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol)
        assert_near_equal(self.prob["cabin_height"], 9.183, tol)  # not actual GASP value
        assert_near_equal(self.prob["cabin_len"], 72.5, tol)  # not actual GASP value
        assert_near_equal(self.prob["nose_height"], 4.683, tol)  # not actual GASP value

        partial_data2 = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data2, atol=1e-8, rtol=1e-8)


class FuselageSizeTestCase1(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "size", FuselageSize(aviary_options=get_option_defaults()), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults("nose_height", 8.6, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")
        self.prob.model.set_input_defaults("cabin_len", 72.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults("cabin_height", 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, tol)
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageSizeTestCase2(
    unittest.TestCase
):  # this is the GASP test case for V3.6 advanced tube and wing
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Fuselage.PROVIDE_SURFACE_AREA,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "parameters", FuselageSize(aviary_options=options), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults("nose_height", 8.6, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")
        self.prob.model.set_input_defaults("cabin_len", 61.6, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults("cabin_height", 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 1, units="unitless")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case2(self):

        self.prob.run_model()

        tol = 3e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 119.03, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 4209, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 119.03, tol
        )  # not actual GASP value

        partial_data2 = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data2, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase1(
    unittest.TestCase
):  # this is the GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units="inch")
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units="inch")
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units="inch")

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FuselageGroup(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft")

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless")

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft")

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4
        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, tol)
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # note: this is the actual GASP value, but for version 3.5. Version 3 has 129.4

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Fuselage.PROVIDE_SURFACE_AREA,
                        val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24,
                        units="inch")  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29,
                        units="inch")  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2,
                        units="inch")  # not actual GASP value

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FuselageGroup(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 129.5, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 18558260.55555555, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 129.5, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Fuselage.PROVIDE_SURFACE_AREA,
                        val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24,
                        units="inch")  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29,
                        units="inch")  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2,
                        units="inch")  # not actual GASP value

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FuselageGroup(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 114.23, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 1.1790053e7, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 114.23, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class FuselageGroupTestCase4(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=30, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24,
                        units="inch")  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 1)  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29,
                        units="inch")  # not actual GASP value
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2,
                        units="inch")  # not actual GASP value

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            FuselageGroup(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units="ft"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.NOSE_FINENESS, 1, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.TAIL_FINENESS, 3, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_FACTOR, 4000, units="unitless"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units="ft"
        )  # not actual GASP value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Fuselage.AVG_DIAMETER], 56.2, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.LENGTH], 114.23, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.TailBoom.LENGTH], 114.23, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
