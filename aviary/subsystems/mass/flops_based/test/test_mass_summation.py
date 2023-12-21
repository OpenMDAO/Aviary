import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.mass_summation import \
    MassSummation
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (Version,
                                                      flops_validation_test,
                                                      get_flops_case_names,
                                                      get_flops_inputs,
                                                      print_case)
from aviary.variable_info.variables import Aircraft, Mission


class TotalSummationTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "tot",
            MassSummation(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.AirConditioning.MASS,
                        Aircraft.AntiIcing.MASS,
                        Aircraft.APU.MASS,
                        Aircraft.Avionics.MASS,
                        Aircraft.Canard.MASS,
                        Aircraft.CrewPayload.PASSENGER_MASS,
                        Aircraft.CrewPayload.BAGGAGE_MASS,
                        Aircraft.CrewPayload.CARGO_MASS,
                        Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
                        Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
                        Aircraft.CrewPayload.FLIGHT_CREW_MASS,
                        Aircraft.Design.EMPTY_MASS_MARGIN,
                        Aircraft.Design.EMPTY_MASS_MARGIN_SCALER,
                        Aircraft.Electrical.MASS,
                        Aircraft.Fins.MASS,
                        Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
                        Aircraft.Fuel.FUEL_SYSTEM_MASS,
                        Aircraft.Furnishings.MASS,
                        Aircraft.Fuselage.MASS,
                        Aircraft.HorizontalTail.MASS,
                        Aircraft.Hydraulics.MASS,
                        Aircraft.Instruments.MASS,
                        Aircraft.LandingGear.MAIN_GEAR_MASS,
                        Aircraft.LandingGear.NOSE_GEAR_MASS,
                        Aircraft.Nacelle.MASS,
                        Aircraft.Paint.MASS,
                        Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                        Aircraft.Wing.SURFACE_CONTROL_MASS,
                        Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
                        Aircraft.Fuel.UNUSABLE_FUEL_MASS,
                        Aircraft.VerticalTail.MASS,
                        Aircraft.Wing.MASS,
                        Mission.Design.GROSS_MASS,
                        Aircraft.Propulsion.TOTAL_ENGINE_MASS,
                        Aircraft.Propulsion.TOTAL_MISC_MASS],
            output_keys=[Aircraft.Design.STRUCTURE_MASS,
                         Aircraft.Propulsion.MASS,
                         Aircraft.Design.SYSTEMS_EQUIP_MASS,
                         Aircraft.Design.EMPTY_MASS,
                         Aircraft.Design.OPERATING_MASS,
                         Aircraft.Design.ZERO_FUEL_MASS,
                         Mission.Design.FUEL_MASS],
            version=Version.TRANSPORT,
            atol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltTotalSummationTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(),
                          name_func=print_case)
    def test_case(self, case_name):

        prob = self.prob

        prob.model.add_subsystem(
            "tot",
            MassSummation(aviary_options=get_flops_inputs(case_name)),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[Aircraft.AirConditioning.MASS,
                        Aircraft.AntiIcing.MASS,
                        Aircraft.APU.MASS,
                        Aircraft.Avionics.MASS,
                        Aircraft.Canard.MASS,
                        Aircraft.CrewPayload.PASSENGER_MASS,
                        Aircraft.CrewPayload.BAGGAGE_MASS,
                        Aircraft.CrewPayload.CARGO_MASS,
                        Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
                        Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
                        Aircraft.CrewPayload.FLIGHT_CREW_MASS,
                        Aircraft.Design.EMPTY_MASS_MARGIN,
                        Aircraft.Design.EMPTY_MASS_MARGIN_SCALER,
                        Aircraft.Electrical.MASS,
                        Aircraft.Fins.MASS,
                        Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
                        Aircraft.Fuel.FUEL_SYSTEM_MASS,
                        Aircraft.Furnishings.MASS_BASE,
                        Aircraft.Fuselage.MASS,
                        Aircraft.HorizontalTail.MASS,
                        Aircraft.Hydraulics.MASS,
                        Aircraft.Instruments.MASS,
                        Aircraft.LandingGear.MAIN_GEAR_MASS,
                        Aircraft.LandingGear.NOSE_GEAR_MASS,
                        Aircraft.Propulsion.TOTAL_MISC_MASS,
                        Aircraft.Nacelle.MASS,
                        Aircraft.Paint.MASS,
                        Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                        Aircraft.Wing.SURFACE_CONTROL_MASS,
                        Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
                        Aircraft.Fuel.UNUSABLE_FUEL_MASS,
                        Aircraft.VerticalTail.MASS,
                        Aircraft.Wing.MASS,
                        Mission.Design.GROSS_MASS,
                        Aircraft.Propulsion.TOTAL_ENGINE_MASS],
            output_keys=[Aircraft.Design.STRUCTURE_MASS,
                         Aircraft.Propulsion.MASS,
                         Aircraft.Design.SYSTEMS_EQUIP_MASS,
                         Aircraft.Design.EMPTY_MASS,
                         Aircraft.Design.OPERATING_MASS,
                         Aircraft.Design.ZERO_FUEL_MASS,
                         Mission.Design.FUEL_MASS],
            version=Version.ALTERNATE,
            atol=1e-10)

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == "__main__":
    unittest.main()
