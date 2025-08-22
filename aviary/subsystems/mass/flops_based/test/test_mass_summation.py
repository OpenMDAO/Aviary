import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.mass_summation import MassSummation, StructureMass
from aviary.subsystems.propulsion.engine_deck import EngineDeck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    Version,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings


class TotalSummationTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        prob.model.add_subsystem(
            'tot',
            MassSummation(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.AirConditioning.MASS,
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
                Aircraft.Propulsion.TOTAL_MISC_MASS,
            ],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS,
                Mission.Summary.OPERATING_MASS,
                Mission.Summary.ZERO_FUEL_MASS,
                Mission.Summary.FUEL_MASS,
            ],
            version=Version.TRANSPORT,
            atol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class AltTotalSummationTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(), name_func=print_case)
    def test_case(self, case_name):
        prob = self.prob

        inputs = get_flops_inputs(case_name, preprocess=True)

        options = {
            Aircraft.Design.USE_ALT_MASS: inputs.get_val(Aircraft.Design.USE_ALT_MASS),
        }

        prob.model.add_subsystem(
            'tot',
            MassSummation(**options),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.AirConditioning.MASS,
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
                Aircraft.Propulsion.TOTAL_ENGINE_MASS,
            ],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS,
                Mission.Summary.OPERATING_MASS,
                Mission.Summary.ZERO_FUEL_MASS,
                Mission.Summary.FUEL_MASS,
            ],
            version=Version.ALTERNATE,
            atol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


class StructureMassTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    def test_case_multiengine(self):
        prob = om.Problem()

        options = AviaryValues()

        options.set_val(Aircraft.Engine.NUM_ENGINES, 4)
        options.set_val(Aircraft.Engine.DATA_FILE, 'models/engines/turbofan_28k.csv')
        # suppress some warning messages about required option for EngineDecks
        options.set_val(Settings.VERBOSITY, 0)
        engineModel1 = EngineDeck(options=options)
        options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
        engineModel2 = EngineDeck(options=options)
        engineModel3 = EngineDeck(options=options)

        preprocess_propulsion(options, [engineModel1, engineModel2, engineModel3])

        comp_options = {
            Aircraft.Engine.NUM_ENGINES: options.get_val(Aircraft.Engine.NUM_ENGINES),
        }

        prob.model.add_subsystem('structure_mass', StructureMass(**comp_options), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Canard.MASS, val=10.0, units='lbm')
        prob.set_val(Aircraft.Fins.MASS, val=20.0, units='lbm')
        prob.set_val(Aircraft.Fuselage.MASS, val=30.0, units='lbm')
        prob.set_val(Aircraft.HorizontalTail.MASS, val=40.0, units='lbm')
        prob.set_val(Aircraft.LandingGear.MAIN_GEAR_MASS, val=50.0, units='lbm')
        prob.set_val(Aircraft.LandingGear.NOSE_GEAR_MASS, val=60.0, units='lbm')
        prob.set_val(Aircraft.Nacelle.MASS, val=np.array([1000.0, 500.0, 1500.0]), units='lbm')
        prob.set_val(Aircraft.Paint.MASS, val=70.0, units='lbm')
        prob.set_val(Aircraft.VerticalTail.MASS, val=80.0, units='lbm')
        prob.set_val(Aircraft.Wing.MASS, val=90.0, units='lbm')

        prob.run_model()

        structure_mass = prob.get_val(Aircraft.Design.STRUCTURE_MASS, 'lbm')
        structure_mass_expected = np.array([3450])
        assert_near_equal(structure_mass, structure_mass_expected, tolerance=1e-8)

        partial_data = prob.check_partials(
            out_stream=None,
            compact_print=True,
            show_only_incorrect=True,
            form='central',
            method='fd',
        )
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
