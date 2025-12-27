import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import (
    get_default_premission_subsystems,
    get_geom_and_mass_subsystems,
)
from aviary.validation_cases.validation_tests import (
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    get_flops_outputs,
    print_case,
    Version,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings

bwb_cases = ['BWBsimpleFLOPS', 'BWBdetailedFLOPS']


@use_tempdirs
class PreMissionGroupTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(omit=bwb_cases), name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)
        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES),
        )
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

        prob = self.prob

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        # prob.model.set_input_defaults(
        #     Aircraft.Engine.SCALE_FACTOR,
        #     flops_inputs.get_val(
        #         Aircraft.Engine.SCALE_FACTOR))

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(2)

        # Initial guess for gross weight.
        # We set it to an unconverged value to test convergence.
        prob.set_val(Mission.Design.GROSS_MASS, val=1000.0)

        set_aviary_initial_values(prob, flops_inputs)

        if case_name in ['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPSdw']:
            # We set these so that their derivatives are defined.
            # The ref values are not set in our test models.
            prob[Aircraft.Wing.ASPECT_RATIO_REF] = prob[Aircraft.Wing.ASPECT_RATIO]
            prob[Aircraft.Wing.THICKNESS_TO_CHORD_REF] = prob[Aircraft.Wing.THICKNESS_TO_CHORD]

        prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm'
        )
        prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, units='lbm'
        )

        flops_validation_test(
            prob,
            case_name,
            input_keys=[],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS,
                Mission.Summary.OPERATING_MASS,
                Mission.Summary.ZERO_FUEL_MASS,
                Mission.Summary.FUEL_MASS,
            ],
            step=1.01e-40,
            atol=1e-8,
            rtol=1e-10,
        )

    def test_diff_configuration_mass(self):
        # This standalone test provides coverage for some features unique to this
        # model.

        prob = om.Problem()

        flops_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')
        flops_outputs = get_flops_outputs('LargeSingleAisle2FLOPS')
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        prob.setup(check=False)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            prob,
            'LargeSingleAisle2FLOPS',
            input_keys=[],
            output_keys=[Mission.Summary.FUEL_MASS],
            atol=1e-4,
            rtol=1e-4,
            check_partials=False,
            flops_inputs=flops_inputs,
            flops_outputs=flops_outputs,
        )

    def test_mass_aero_only(self):
        # tests geom, mass, aero only, similar to IANAL=2 mode in FLOPS
        prob = om.Problem()

        flops_inputs = get_flops_inputs('LargeSingleAisle2FLOPS')
        flops_outputs = get_flops_outputs('LargeSingleAisle2FLOPS')
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]

        preprocess_options(flops_inputs, engine_models=engines)

        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines=engines)[
            1:
        ]

        prob.model.add_subsystem(
            'mass_and_aero_premission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes=['*'],
        )

        setup_model_options(prob, flops_inputs)
        prob.setup()
        set_aviary_initial_values(prob, flops_inputs)

        prob.set_val(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 'lbf'),
            'lbf',
        )
        prob.run_model()

        flops_validation_test(
            prob,
            'LargeSingleAisle2FLOPS',
            input_keys=[],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                Aircraft.Design.EMPTY_MASS,
                Mission.Summary.OPERATING_MASS,
                Mission.Summary.ZERO_FUEL_MASS,
                Mission.Summary.FUEL_MASS,
            ],
            atol=1e-4,
            rtol=1e-4,
            check_partials=False,
            flops_inputs=flops_inputs,
            flops_outputs=flops_outputs,
        )


@use_tempdirs
class BWBPreMissionGroupTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case_all_subsystems(self, case_name):
        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)

        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES),
        )
        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES),
        )
        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES),
        )
        flops_inputs.set_val(
            Aircraft.Fuel.TOTAL_CAPACITY,
            flops_outputs.get_val(Aircraft.Fuel.TOTAL_CAPACITY, units='lbm'),
            units='lbm',
        )
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)

        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)

        prob = self.prob

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        # prob.model.set_input_defaults(
        #     Aircraft.Engine.SCALE_FACTOR,
        #     flops_inputs.get_val(
        #         Aircraft.Engine.SCALE_FACTOR))

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(2)

        # Initial guess for gross weight.
        # We set it to an unconverged value to test convergence.
        # prob.set_val(Mission.Design.GROSS_MASS, val=1000.0)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[],
            output_keys=[
                # Geometry
                # BWBSimpleCabinLayout
                Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.Fuselage.MAX_HEIGHT,
                Aircraft.BWB.NUM_BAYS,
                # BWBFuselagePrelim
                Aircraft.Fuselage.REF_DIAMETER,
                Aircraft.Fuselage.PLANFORM_AREA,
                # BWBWingPrelim
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.ASPECT_RATIO_REF,
                Aircraft.Wing.LOAD_FRACTION,
                # _BWBWing
                Aircraft.Wing.WETTED_AREA,
                # _Tail
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.VerticalTail.WETTED_AREA,
                # _FuselageRatios
                Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
                Aircraft.Fuselage.LENGTH_TO_DIAMETER,
                # Nacelles
                Aircraft.Nacelle.TOTAL_WETTED_AREA,
                Aircraft.Nacelle.WETTED_AREA,
                # Canard
                Aircraft.Canard.WETTED_AREA,
                # BWBWingCharacteristicLength
                Aircraft.Wing.CHARACTERISTIC_LENGTH,
                Aircraft.Wing.FINENESS,
                # OtherCharacteristicLengths
                Aircraft.Canard.CHARACTERISTIC_LENGTH,
                Aircraft.Canard.FINENESS,
                Aircraft.Fuselage.CHARACTERISTIC_LENGTH,
                Aircraft.Fuselage.FINENESS,
                Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH,
                Aircraft.HorizontalTail.FINENESS,
                Aircraft.Nacelle.CHARACTERISTIC_LENGTH,
                Aircraft.Nacelle.FINENESS,
                Aircraft.VerticalTail.CHARACTERISTIC_LENGTH,
                Aircraft.VerticalTail.FINENESS,
                # TotalWettedArea
                Aircraft.Design.TOTAL_WETTED_AREA,
                # Mass
                # CargoMass
                Aircraft.CrewPayload.PASSENGER_MASS,
                Aircraft.CrewPayload.BAGGAGE_MASS,
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
                Aircraft.CrewPayload.CARGO_MASS,
                Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS,
                # TransportCargoContainersMass
                Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
                # TransportEngineCtrlsMass
                Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                # TransportAvionicsMass
                Aircraft.Avionics.MASS,
                # FuelCapacityGroup
                Aircraft.Fuel.WING_FUEL_CAPACITY,
                Aircraft.Fuel.TOTAL_CAPACITY,
                # EngineMass
                Aircraft.Engine.MASS,
                Aircraft.Engine.ADDITIONAL_MASS,
                Aircraft.Propulsion.TOTAL_ENGINE_MASS,
                # TransportFuelSystemMass
                Aircraft.Fuel.FUEL_SYSTEM_MASS,
                # TransportAirCondMass
                Aircraft.AirConditioning.MASS,
                # TransportEngineOilMass
                Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
                # BWBFurnishingsGroupMass
                Aircraft.Furnishings.MASS,
                # TransportHydraulicsGroupMass
                Aircraft.Hydraulics.MASS,
                # PassengerServiceMass
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
                # ElectricalMass
                Aircraft.Electrical.MASS,
                # AntiIcingMass
                Aircraft.AntiIcing.MASS,
                # TransportAPUMass
                Aircraft.APU.MASS,
                # NonFlightCrewMass
                Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
                # FlightCrewMass
                Aircraft.CrewPayload.FLIGHT_CREW_MASS,
                # TransportInstrumentMass
                Aircraft.Instruments.MASS,
                # EngineMiscMass
                Aircraft.Propulsion.TOTAL_MISC_MASS,
                # NacelleMass
                Aircraft.Nacelle.MASS,
                # PaintMass
                Aircraft.Paint.MASS,
                # ThrustReverserMass
                Aircraft.Engine.THRUST_REVERSERS_MASS,
                Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
                # LandingMassGroup
                Aircraft.Design.TOUCHDOWN_MASS,
                # SurfaceControlMass
                Aircraft.Wing.SURFACE_CONTROL_MASS,
                Aircraft.Wing.CONTROL_SURFACE_AREA,
                # BWBFuselageMass
                Aircraft.Fuselage.MASS,
                # HorizontalTailMass
                Aircraft.HorizontalTail.MASS,
                # VerticalTailMass
                Aircraft.VerticalTail.MASS,
                # CanardMass
                Aircraft.Canard.MASS,
                # FinMass
                Aircraft.Fins.MASS,
                # WingMassGroup
                # BWBDetailedWingBendingFact
                Aircraft.Wing.BENDING_MATERIAL_FACTOR,
                Aircraft.Wing.ENG_POD_INERTIA_FACTOR,
                # BWBWingMiscMass
                Aircraft.Wing.MISC_MASS,
                # WingShearControlMass
                Aircraft.Wing.SHEAR_CONTROL_MASS,
                # WingBendingMass
                Aircraft.Wing.BENDING_MATERIAL_MASS,
                # BWBAftBodyMass
                Aircraft.Fuselage.AFTBODY_MASS,
                Aircraft.Wing.BWB_AFTBODY_MASS,
                # MassSummation
                # StructureMass
                Aircraft.Design.STRUCTURE_MASS,
                # PropulsionMass
                Aircraft.Propulsion.MASS,
                # SystemsEquipMass
                Aircraft.Design.SYSTEMS_EQUIP_MASS,
                # EmptyMass
                Aircraft.Design.EMPTY_MASS,
                # OperatingMass
                Mission.Summary.OPERATING_MASS,
                # ZeroFuelMass
                Mission.Summary.ZERO_FUEL_MASS,
                # FuelMass
                Mission.Summary.FUEL_MASS,
            ],
            version=Version.BWB,
            step=1.01e-40,
            atol=1e-6,
            rtol=1e-6,
            check_values=True,
            check_partials=True,
        )

    def test_case_geom(self):
        case_name = 'BWBsimpleFLOPS'
        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)
        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES),
        )
        flops_inputs.set_val(
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES,
            flops_outputs.get_val(Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES),
        )
        flops_inputs.set_val(Settings.VERBOSITY, 0)

        preprocess_options(flops_inputs)
        default_premission_subsystems = get_geom_and_mass_subsystems('FLOPS')[0:1]

        prob = self.prob

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=flops_inputs, subsystems=default_premission_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(2)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[],
            output_keys=[
                Aircraft.Fuselage.PLANFORM_AREA,
            ],
            version=Version.BWB,
            step=1.01e-40,
            atol=1e-6,
            rtol=1e-6,
            check_partials=True,
            excludes=['*detailed_wing.*'],  # does not work?
        )


@use_tempdirs
class BWBPreMissionGroupCSVTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = AviaryProblem()

        csv_path = 'models/aircraft/blended_wing_body/bwb_simple_FLOPS.csv'
        self.flops_inputs = prob.load_inputs(csv_path)
        prob.check_and_preprocess_inputs()

    def test_case_geom(self):
        """
        premission: geometry
        """
        prob = self.prob

        preprocess_options(self.flops_inputs)
        geom_subsystem = get_geom_and_mass_subsystems('FLOPS')[0:1]

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=self.flops_inputs, subsystems=geom_subsystem),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.run_model()

        tol = 1e-5
        # Geometry
        # BWBSimpleCabinLayout
        assert_near_equal(prob[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH], 96.25, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 63.96, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 5173.187202504683, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MAX_HEIGHT], 15.125, tol)
        assert_near_equal(prob[Aircraft.BWB.NUM_BAYS], 5.0, tol)
        # BWBFuselagePrelim
        assert_near_equal(prob[Aircraft.Fuselage.REF_DIAMETER], 39.8525, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 7390.267432149546, tol)
        # BWBWingPrelim
        assert_near_equal(prob[Aircraft.Wing.AREA], 16555.972297926455, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO_REF], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.LOAD_FRACTION], 0.53107166, tol)
        # _BWBWing
        assert_near_equal(prob[Aircraft.Wing.WETTED_AREA], 33816.732336575638, tol)
        # _Tail
        assert_near_equal(prob[Aircraft.HorizontalTail.WETTED_AREA], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.WETTED_AREA], 0.0, tol)
        # _FuselageRatios
        assert_near_equal(prob[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN], 0.16739117852998228, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH_TO_DIAMETER], 3.4502226961922089, tol)
        # Nacelles
        assert_near_equal(prob[Aircraft.Nacelle.WETTED_AREA], 498.26822066, tol)
        assert_near_equal(prob[Aircraft.Nacelle.TOTAL_WETTED_AREA], 3 * 498.26822066, tol)
        # Canard
        assert_near_equal(prob[Aircraft.Canard.WETTED_AREA], 0.0, tol)
        # BWBWingCharacteristicLength
        assert_near_equal(prob[Aircraft.Wing.CHARACTERISTIC_LENGTH], 69.53953418, tol)
        assert_near_equal(prob[Aircraft.Wing.FINENESS], 0.11, tol)
        # OtherCharacteristicLengths
        assert_near_equal(prob[Aircraft.Canard.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.Canard.FINENESS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CHARACTERISTIC_LENGTH], 137.5, tol)
        assert_near_equal(prob[Aircraft.Fuselage.FINENESS], 3.4502227, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.FINENESS], 0.11, tol)
        assert_near_equal(prob[Aircraft.Nacelle.CHARACTERISTIC_LENGTH], [15.68611614], tol)
        assert_near_equal(prob[Aircraft.Nacelle.FINENESS], [1.38269353], tol)
        # TotalWettedArea
        assert_near_equal(prob[Aircraft.Design.TOTAL_WETTED_AREA], 35311.53118076, tol)

    def test_case_geom_mass(self):
        """
        premission: geometry + mass
        """
        prob = self.prob

        preprocess_options(self.flops_inputs)
        geom_mass_subsystems = get_geom_and_mass_subsystems('FLOPS')

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=self.flops_inputs, subsystems=geom_mass_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=70000.0 * 3, units='lbf')

        prob.run_model()

        tol = 1e-4
        # Geometry
        # BWBSimpleCabinLayout
        assert_near_equal(prob[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH], 96.25, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 63.96, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 5173.187202504683, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MAX_HEIGHT], 15.125, tol)
        assert_near_equal(prob[Aircraft.BWB.NUM_BAYS], 5.0, tol)
        # BWBFuselagePrelim
        assert_near_equal(prob[Aircraft.Fuselage.REF_DIAMETER], 39.8525, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 7390.267432149546, tol)
        # BWBWingPrelim
        assert_near_equal(prob[Aircraft.Wing.AREA], 16555.972297926455, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO_REF], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.LOAD_FRACTION], 0.53107166, tol)
        # _BWBWing
        assert_near_equal(prob[Aircraft.Wing.WETTED_AREA], 33816.732336575638, tol)
        # _Tail
        assert_near_equal(prob[Aircraft.HorizontalTail.WETTED_AREA], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.WETTED_AREA], 0.0, tol)
        # _FuselageRatios
        assert_near_equal(prob[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN], 0.16739117852998228, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH_TO_DIAMETER], 3.4502226961922089, tol)
        # Nacelles
        assert_near_equal(prob[Aircraft.Nacelle.WETTED_AREA], 498.26822066, tol)
        assert_near_equal(prob[Aircraft.Nacelle.TOTAL_WETTED_AREA], 3 * 498.26822066, tol)
        # Canard
        assert_near_equal(prob[Aircraft.Canard.WETTED_AREA], 0.0, tol)
        # BWBWingCharacteristicLength
        assert_near_equal(prob[Aircraft.Wing.CHARACTERISTIC_LENGTH], 69.53953418, tol)
        assert_near_equal(prob[Aircraft.Wing.FINENESS], 0.11, tol)
        # OtherCharacteristicLengths
        assert_near_equal(prob[Aircraft.Canard.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.Canard.FINENESS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CHARACTERISTIC_LENGTH], 137.5, tol)
        assert_near_equal(prob[Aircraft.Fuselage.FINENESS], 3.4502227, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.FINENESS], 0.11, tol)
        assert_near_equal(prob[Aircraft.Nacelle.CHARACTERISTIC_LENGTH], [15.68611614], tol)
        assert_near_equal(prob[Aircraft.Nacelle.FINENESS], [1.38269353], tol)
        assert_near_equal(prob[Aircraft.VerticalTail.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.FINENESS], 0.11, tol)
        # TotalWettedArea
        assert_near_equal(prob[Aircraft.Design.TOTAL_WETTED_AREA], 35311.53118076, tol)
        # Mass
        # CargoMass
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_MASS], 77220.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.BAGGAGE_MASS], 20592.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 97812.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 97812.0, tol)
        # TransportCargoContainersMass
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 3850.0, tol)
        # TransportEngineCtrlsMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS], 206.36860226, tol)
        # TransportAvionicsMass
        assert_near_equal(prob[Aircraft.Avionics.MASS], 2896.223816950469, tol)
        # FuelCapacityGroup
        assert_near_equal(prob[Aircraft.Fuel.WING_FUEL_CAPACITY], 2385712.4988316689, tol)
        assert_near_equal(prob[Aircraft.Fuel.TOTAL_CAPACITY], 2385712.4988316689, tol)
        # EngineMass
        assert_near_equal(prob[Aircraft.Engine.MASS], 17825.63336233, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 53476.90008698, tol)
        # TransportFuelSystemMass
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 8120.2023807944415, tol)
        # TransportAirCondMass
        assert_near_equal(prob[Aircraft.AirConditioning.MASS], 4383.96064972, tol)
        # TransportEngineOilMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 346.93557352, tol)
        # BWBFurnishingsGroupMass
        assert_near_equal(prob[Aircraft.Furnishings.MASS], 61482.097969438299, tol)
        # TransportHydraulicsGroupMass
        assert_near_equal(prob[Aircraft.Hydraulics.MASS], 7368.5077321194321, tol)
        # PassengerServiceMass
        assert_near_equal(
            prob[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS], 10806.675950702213, tol
        )
        # ElectricalMass
        assert_near_equal(prob[Aircraft.Electrical.MASS], 4514.28869169, tol)
        # AntiIcingMass
        assert_near_equal(prob[Aircraft.AntiIcing.MASS], 519.37038003, tol)
        # TransportAPUMass
        assert_near_equal(prob[Aircraft.APU.MASS], 2148.13002234, tol)
        # NonFlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS], 3810.0, tol)
        # FlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 450.0, tol)
        # TransportInstrumentMass
        assert_near_equal(prob[Aircraft.Instruments.MASS], 1383.9538229392606, tol)
        # EngineMiscMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_MISC_MASS], 0.0, tol)
        # NacelleMass
        assert_near_equal(prob[Aircraft.Nacelle.MASS], 0.0, tol)
        # PaintMass
        assert_near_equal(prob[Aircraft.Paint.MASS], 0.0, tol)
        # ThrustReverserMass
        assert_near_equal(prob[Aircraft.Engine.THRUST_REVERSERS_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS], 0.0, tol)
        # LandingMassGroup
        assert_near_equal(prob[Aircraft.Design.TOUCHDOWN_MASS], 699279.2, tol)
        # SurfaceControlMass
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 14152.3734702, tol)
        assert_near_equal(prob[Aircraft.Wing.CONTROL_SURFACE_AREA], 5513.13877521, tol)
        # BWBFuselageMass
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 152790.66300003964, tol)
        # HorizontalTailMass
        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 0.0, tol)
        # VerticalTailMass
        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 0.0, tol)
        # CanardMass
        assert_near_equal(prob[Aircraft.Canard.MASS], 0.0, tol)
        # FinMass
        assert_near_equal(prob[Aircraft.Fins.MASS], 3159.3781042368792, tol)
        # WingMassGroup
        # BWBDetailedWingBendingFact
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_FACTOR], 2.68745091, tol)
        assert_near_equal(prob[Aircraft.Wing.ENG_POD_INERTIA_FACTOR], 1.0, tol)
        # BWBWingMiscMass
        assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 21498.83307778, tol)
        # WingShearControlMass
        assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 38779.21499739, tol)
        # WingBendingMass
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 6313.44762977, tol)
        # BWBAftBodyMass
        assert_near_equal(prob[Aircraft.Fuselage.AFTBODY_MASS], 24278.05868511, tol)
        assert_near_equal(prob[Aircraft.Wing.BWB_AFTBODY_MASS], 20150.78870864, tol)
        # MassSummation
        # StructureMass
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 273591.31917826, tol)
        # PropulsionMass
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 61597.102467771889, tol)
        # SystemsEquipMass
        assert_near_equal(prob[Aircraft.Design.SYSTEMS_EQUIP_MASS], 98848.9061107412710, tol)
        # EmptyMass
        assert_near_equal(prob[Aircraft.Design.EMPTY_MASS], 434037.32820147, tol)
        # OperatingMass
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 455464.65969526308, tol)
        # ZeroFuelMass
        assert_near_equal(prob[Mission.Summary.ZERO_FUEL_MASS], 553276.65969526302, tol)
        # FuelMass
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 320822.34030473698, tol)

    def test_case_all_subsystems(self):
        """
        premission: propulsion + geometry + aerodynamics + mass
        Note: not checking propulsion and aerodynamics
        """
        prob = self.prob

        engines = [build_engine_deck(self.flops_inputs)]
        preprocess_options(self.flops_inputs)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines=engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.flops_inputs, subsystems=default_premission_subsystems
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST], 70000.0 * 3, tol)
        # Geometry
        # BWBSimpleCabinLayout
        assert_near_equal(prob[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH], 96.25, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 63.96, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 5173.187202504683, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MAX_HEIGHT], 15.125, tol)
        assert_near_equal(prob[Aircraft.BWB.NUM_BAYS], 5.0, tol)
        # BWBFuselagePrelim
        assert_near_equal(prob[Aircraft.Fuselage.REF_DIAMETER], 39.8525, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 7390.267432149546, tol)
        # BWBWingPrelim
        assert_near_equal(prob[Aircraft.Wing.AREA], 16555.972297926455, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO_REF], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.LOAD_FRACTION], 0.53107166, tol)
        # _BWBWing
        assert_near_equal(prob[Aircraft.Wing.WETTED_AREA], 33816.732336575638, tol)
        # _Tail
        assert_near_equal(prob[Aircraft.HorizontalTail.WETTED_AREA], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.WETTED_AREA], 0.0, tol)
        # _FuselageRatios
        assert_near_equal(prob[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN], 0.16739117852998228, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH_TO_DIAMETER], 3.4502226961922089, tol)
        # Nacelles
        assert_near_equal(prob[Aircraft.Nacelle.WETTED_AREA], 498.26822066, tol)
        assert_near_equal(prob[Aircraft.Nacelle.TOTAL_WETTED_AREA], 3 * 498.26822066, tol)
        # Canard
        assert_near_equal(prob[Aircraft.Canard.WETTED_AREA], 0.0, tol)
        # BWBWingCharacteristicLength
        assert_near_equal(prob[Aircraft.Wing.CHARACTERISTIC_LENGTH], 69.53953418, tol)
        assert_near_equal(prob[Aircraft.Wing.FINENESS], 0.11, tol)
        # OtherCharacteristicLengths
        assert_near_equal(prob[Aircraft.Canard.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.Canard.FINENESS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CHARACTERISTIC_LENGTH], 137.5, tol)
        assert_near_equal(prob[Aircraft.Fuselage.FINENESS], 3.4502227, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.FINENESS], 0.11, tol)
        assert_near_equal(prob[Aircraft.Nacelle.CHARACTERISTIC_LENGTH], [15.68611614], tol)
        assert_near_equal(prob[Aircraft.Nacelle.FINENESS], [1.38269353], tol)
        assert_near_equal(prob[Aircraft.VerticalTail.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.FINENESS], 0.11, tol)
        # TotalWettedArea
        assert_near_equal(prob[Aircraft.Design.TOTAL_WETTED_AREA], 35311.53118076, tol)
        # Mass
        # CargoMass
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_MASS], 77220.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.BAGGAGE_MASS], 20592.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 97812.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 97812.0, tol)
        # TransportCargoContainersMass
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 3850.0, tol)
        # TransportEngineCtrlsMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS], 206.36860226, tol)
        # TransportAvionicsMass
        assert_near_equal(prob[Aircraft.Avionics.MASS], 2896.223816950469, tol)
        # FuelCapacityGroup
        assert_near_equal(prob[Aircraft.Fuel.WING_FUEL_CAPACITY], 2385712.4988316689, tol)
        assert_near_equal(prob[Aircraft.Fuel.TOTAL_CAPACITY], 2385712.4988316689, tol)
        # EngineMass
        assert_near_equal(prob[Aircraft.Engine.MASS], 17825.63336233, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 53476.90008698, tol)
        # TransportFuelSystemMass
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 8120.2023807944415, tol)
        # TransportAirCondMass
        assert_near_equal(prob[Aircraft.AirConditioning.MASS], 4383.96064972, tol)
        # TransportEngineOilMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 346.93557352, tol)
        # BWBFurnishingsGroupMass
        assert_near_equal(prob[Aircraft.Furnishings.MASS], 61482.097969438299, tol)
        # TransportHydraulicsGroupMass
        assert_near_equal(prob[Aircraft.Hydraulics.MASS], 7368.5077321194321, tol)
        # PassengerServiceMass
        assert_near_equal(
            prob[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS], 10806.675950702213, tol
        )
        # ElectricalMass
        assert_near_equal(prob[Aircraft.Electrical.MASS], 4514.28869169, tol)
        # AntiIcingMass
        assert_near_equal(prob[Aircraft.AntiIcing.MASS], 519.37038003, tol)
        # TransportAPUMass
        assert_near_equal(prob[Aircraft.APU.MASS], 2148.13002234, tol)
        # NonFlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS], 3810.0, tol)
        # FlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 450.0, tol)
        # TransportInstrumentMass
        assert_near_equal(prob[Aircraft.Instruments.MASS], 1383.9538229392606, tol)
        # EngineMiscMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_MISC_MASS], 0.0, tol)
        # NacelleMass
        assert_near_equal(prob[Aircraft.Nacelle.MASS], 0.0, tol)
        # PaintMass
        assert_near_equal(prob[Aircraft.Paint.MASS], 0.0, tol)
        # ThrustReverserMass
        assert_near_equal(prob[Aircraft.Engine.THRUST_REVERSERS_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS], 0.0, tol)
        # LandingMassGroup
        assert_near_equal(prob[Aircraft.Design.TOUCHDOWN_MASS], 699279.2, tol)
        # SurfaceControlMass
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 14152.3734702, tol)
        assert_near_equal(prob[Aircraft.Wing.CONTROL_SURFACE_AREA], 5513.13877521, tol)
        # BWBFuselageMass
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 152790.66300003964, tol)
        # HorizontalTailMass
        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 0.0, tol)
        # VerticalTailMass
        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 0.0, tol)
        # CanardMass
        assert_near_equal(prob[Aircraft.Canard.MASS], 0.0, tol)
        # FinMass
        assert_near_equal(prob[Aircraft.Fins.MASS], 3159.3781042368792, tol)
        # WingMassGroup
        # BWBDetailedWingBendingFact
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_FACTOR], 2.68745091, tol)
        assert_near_equal(prob[Aircraft.Wing.ENG_POD_INERTIA_FACTOR], 1.0, tol)
        # BWBWingMiscMass
        assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 21498.83307778, tol)
        # WingShearControlMass
        assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 38779.21499739, tol)
        # WingBendingMass
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 6313.44762977, tol)
        # BWBAftBodyMass
        assert_near_equal(prob[Aircraft.Fuselage.AFTBODY_MASS], 24278.05868511, tol)
        assert_near_equal(prob[Aircraft.Wing.BWB_AFTBODY_MASS], 20150.78870864, tol)
        # MassSummation
        # StructureMass
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 273591.31917826, tol)
        # PropulsionMass
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 61597.102467771889, tol)
        # SystemsEquipMass
        assert_near_equal(prob[Aircraft.Design.SYSTEMS_EQUIP_MASS], 98848.9061107412710, tol)
        # EmptyMass
        assert_near_equal(prob[Aircraft.Design.EMPTY_MASS], 434037.32820147, tol)
        # OperatingMass
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 455464.65969526308, tol)
        # ZeroFuelMass
        assert_near_equal(prob[Mission.Summary.ZERO_FUEL_MASS], 553276.65969526302, tol)
        # FuelMass
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 320822.34030473698, tol)


if __name__ == '__main__':
    unittest.main()
