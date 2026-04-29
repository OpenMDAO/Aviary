import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.core.aviary_problem import AviaryProblem
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
            CorePreMission(
                aviary_options=flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
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
        prob.set_val(Aircraft.Design.GROSS_MASS, val=1000.0)

        set_aviary_initial_values(prob, flops_inputs)

        if case_name in ['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPSdw']:
            # We set these so that their derivatives are defined.
            # The ref values are not set in our test models.
            prob[Aircraft.Wing.ASPECT_RATIO_REFERENCE] = prob[Aircraft.Wing.ASPECT_RATIO]
            prob[Aircraft.Wing.THICKNESS_TO_CHORD_REFERENCE] = prob[
                Aircraft.Wing.THICKNESS_TO_CHORD
            ]

        prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_MASS, units='lbm'
        )
        prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = flops_outputs.get_val(
            Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, units='lbm'
        )

        flops_validation_test(
            self,
            prob,
            case_name,
            input_keys=[],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
                Aircraft.Design.EMPTY_MASS,
                Mission.OPERATING_MASS,
                Mission.ZERO_FUEL_MASS,
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
            CorePreMission(
                aviary_options=flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        prob.setup(check=False)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            self,
            prob,
            'LargeSingleAisle2FLOPS',
            input_keys=[],
            output_keys=[],
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
            CorePreMission(
                aviary_options=flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
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
            self,
            prob,
            'LargeSingleAisle2FLOPS',
            input_keys=[],
            output_keys=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
                Aircraft.Design.EMPTY_MASS,
                Mission.OPERATING_MASS,
                Mission.ZERO_FUEL_MASS,
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

    # @parameterized.expand(get_flops_case_names(only=bwb_cases), name_func=print_case)
    def test_case_all_subsystems(
        self,
    ):
        case_name = 'BWBdetailedFLOPS'
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
            CorePreMission(
                aviary_options=flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
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
        # prob.set_val(Aircraft.Design.GROSS_MASS, val=1000.0)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            self,
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
                Aircraft.Wing.ASPECT_RATIO_REFERENCE,
                Aircraft.Wing.LOAD_FRACTION,
                # BWBWingWettedArea
                Aircraft.Wing.WETTED_AREA,
                # TailWettedArea
                Aircraft.HorizontalTail.WETTED_AREA,
                Aircraft.VerticalTail.WETTED_AREA,
                # _FuselageRatios
                Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
                Aircraft.Fuselage.LENGTH_TO_DIAMETER,
                # NacelleWettedArea
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
                Aircraft.CrewPayload.PASSENGER_MASS_TOTAL,
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
                Aircraft.CrewPayload.CABIN_CREW_MASS,
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
                Aircraft.Design.TOUCHDOWN_MASS_MAX,
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
                Aircraft.Wing.MASS,
                # BWBAftBodyMass
                Aircraft.Fuselage.AFTBODY_MASS,
                Aircraft.Wing.BWB_AFTBODY_MASS,
                # MassSummation
                # StructureMass
                Aircraft.Design.STRUCTURE_MASS,
                # PropulsionMass
                Aircraft.Propulsion.MASS,
                # SystemsEquipMass
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
                # EmptyMass
                Aircraft.Design.EMPTY_MASS,
                # OperatingMass
                Mission.OPERATING_MASS,
                # ZeroFuelMass
                Mission.ZERO_FUEL_MASS,
            ],
            version=Version.BWB,
            step=1.01e-40,
            atol=1e-6,
            rtol=1e-6,
            check_values=True,
            check_partials=True,
            excludes=['*bending_material_factor*'],
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

        engines = [build_engine_deck(flops_inputs)]
        preprocess_options(flops_inputs, engine_models=engines)

        default_premission_subsystems = get_geom_and_mass_subsystems('FLOPS')[0:1]

        prob = self.prob

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(2)

        set_aviary_initial_values(prob, flops_inputs)

        flops_validation_test(
            self,
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
class BWBPreMissionGroupCSVTest1(unittest.TestCase):
    """
    testing using bwb_simple_FLOPS.csv
    """

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
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=geom_subsystem,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.run_model()

        tol = 1e-5
        # Geometry
        # BWBComputeDetailedWingDist
        assert_near_equal(prob[Aircraft.Wing.SPAN], 238.08, tol)
        # BWBSimpleCabinLayout
        assert_near_equal(prob[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH], 96.25, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 63.96, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 5173.187202504683, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MAX_HEIGHT], 15.125, tol)
        assert_near_equal(prob[Aircraft.BWB.NUM_BAYS], 5.0, 1e-4)
        # BWBFuselagePrelim
        assert_near_equal(prob[Aircraft.Fuselage.REF_DIAMETER], 39.8525, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 7390.267432149546, tol)
        # BWBWingPrelim
        assert_near_equal(prob[Aircraft.Wing.AREA], 16555.972297926455, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO_REFERENCE], 3.4488813, tol)
        assert_near_equal(prob[Aircraft.Wing.LOAD_FRACTION], 0.53107166, tol)
        # BWBWingWettedArea
        assert_near_equal(prob[Aircraft.Wing.WETTED_AREA], 33816.732336575638, tol)
        # TailWettedArea
        assert_near_equal(prob[Aircraft.HorizontalTail.WETTED_AREA], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.WETTED_AREA], 0.0, tol)
        # _FuselageRatios
        assert_near_equal(prob[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN], 0.16739117852998228, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH_TO_DIAMETER], 3.4502226961922089, tol)
        # NacelleWettedArea
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
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=geom_mass_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=70000.0 * 3, units='lbf')

        prob.run_model()

        tol = 1e-4
        # Mass
        # CargoMass
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_MASS_TOTAL], 77220.0, tol)
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
        assert_near_equal(prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 3810.0, tol)
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
        assert_near_equal(prob[Aircraft.Design.TOUCHDOWN_MASS_MAX], 699279.2, tol)
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
        assert_near_equal(
            prob[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS], 98848.9061107412710, tol
        )
        # EmptyMass
        assert_near_equal(prob[Aircraft.Design.EMPTY_MASS], 434037.32820147, tol)
        # OperatingMass
        assert_near_equal(prob[Mission.OPERATING_MASS], 455464.65969526308, tol)
        # ZeroFuelMass
        assert_near_equal(prob[Mission.ZERO_FUEL_MASS], 553276.65969526302, tol)

    def test_case_all_subsystems(self):
        """
        premission: propulsion + geometry + aerodynamics + mass
        """
        prob = self.prob

        engines = [build_engine_deck(self.flops_inputs)]
        preprocess_options(self.flops_inputs)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines=engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
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
        # Aerodynamics
        # Design
        assert_near_equal(prob[Aircraft.Design.MACH], 0.91589163, tol)
        assert_near_equal(prob[Aircraft.Design.LIFT_COEFFICIENT], 0.3487563, tol)


@use_tempdirs
class BWBPreMissionGroupCSVTest2(unittest.TestCase):
    """
    testing using bwb_detailed_FLOPS.csv
    """

    def setUp(self):
        prob = self.prob = AviaryProblem()

        csv_path = 'models/aircraft/blended_wing_body/bwb_detailed_FLOPS.csv'
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
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=geom_subsystem,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.run_model()

        tol = 1e-5
        # Geometry
        # BWBComputeDetailedWingDist
        assert_near_equal(prob[Aircraft.Wing.SPAN], 253.720756, tol)
        # DetailedCabinLayout
        assert_near_equal(prob[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH], 78.23465729, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 38.5, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 4638.43963915, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MAX_HEIGHT], 12.29401757, tol)
        assert_near_equal(prob[Aircraft.BWB.NUM_BAYS], 7.0, tol)
        # BWBFuselagePrelim
        assert_near_equal(prob[Aircraft.Fuselage.REF_DIAMETER], 45.88190626, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 6626.34234164, tol)
        # BWBWingPrelim
        assert_near_equal(prob[Aircraft.Wing.AREA], 12059.52621246, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO], 5.39216403, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO_REFERENCE], 5.39216403, tol)
        assert_near_equal(prob[Aircraft.Wing.LOAD_FRACTION], 0.47167013, tol)
        # BWBWingWettedArea
        assert_near_equal(prob[Aircraft.Wing.WETTED_AREA], 24610.64920629, tol)
        # TailWettedArea
        assert_near_equal(prob[Aircraft.HorizontalTail.WETTED_AREA], 0.0, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.WETTED_AREA], 0.0, tol)
        # _FuselageRatios
        assert_near_equal(prob[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN], 0.18083624, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH_TO_DIAMETER], 2.43590132, tol)
        # NacelleWettedArea
        assert_near_equal(prob[Aircraft.Nacelle.WETTED_AREA], 498.26822066, tol)
        assert_near_equal(prob[Aircraft.Nacelle.TOTAL_WETTED_AREA], 3 * 498.26822066, tol)
        # Canard
        assert_near_equal(prob[Aircraft.Canard.WETTED_AREA], 0.0, tol)
        # BWBWingCharacteristicLength
        assert_near_equal(prob[Aircraft.Wing.CHARACTERISTIC_LENGTH], 47.53070424, tol)
        assert_near_equal(prob[Aircraft.Wing.FINENESS], 0.11, tol)
        # OtherCharacteristicLengths
        assert_near_equal(prob[Aircraft.Canard.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.Canard.FINENESS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CHARACTERISTIC_LENGTH], 111.76379613, tol)
        assert_near_equal(prob[Aircraft.Fuselage.FINENESS], 2.43590132, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.FINENESS], 0.11, tol)
        assert_near_equal(prob[Aircraft.Nacelle.CHARACTERISTIC_LENGTH], [15.68611614], tol)
        assert_near_equal(prob[Aircraft.Nacelle.FINENESS], [1.38269353], tol)
        # TotalWettedArea
        assert_near_equal(prob[Aircraft.Design.TOTAL_WETTED_AREA], 26105.45386827, tol)

    def test_case_geom_mass(self):
        """
        premission: geometry + mass
        """
        prob = self.prob

        preprocess_options(self.flops_inputs)
        geom_mass_subsystems = get_geom_and_mass_subsystems('FLOPS')

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=geom_mass_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=70000.0 * 3, units='lbf')

        prob.run_model()

        tol = 1e-4
        # Mass
        # CargoMass
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_MASS_TOTAL], 77220.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.BAGGAGE_MASS], 20592.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 97812.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 97812.0, tol)
        # TransportCargoContainersMass
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 3850.0, tol)
        # TransportEngineCtrlsMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS], 206.36860226, tol)
        # TransportAvionicsMass
        assert_near_equal(prob[Aircraft.Avionics.MASS], 2763.47804131, tol)
        # FuelCapacityGroup
        assert_near_equal(prob[Aircraft.Fuel.WING_FUEL_CAPACITY], 1187780.58456809, tol)
        assert_near_equal(prob[Aircraft.Fuel.TOTAL_CAPACITY], 1187780.58456809, tol)
        # EngineMass
        assert_near_equal(prob[Aircraft.Engine.MASS], 17825.63336233, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 53476.90008698, tol)
        # TransportFuelSystemMass
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 5418.70316, tol)
        # TransportAirCondMass
        assert_near_equal(prob[Aircraft.AirConditioning.MASS], 3871.27698278, tol)
        # TransportEngineOilMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 346.93557352, tol)
        # BWBFurnishingsGroupMass
        assert_near_equal(prob[Aircraft.Furnishings.MASS], 57402.19908931, tol)
        # TransportHydraulicsGroupMass
        assert_near_equal(prob[Aircraft.Hydraulics.MASS], 6139.65550426, tol)
        # PassengerServiceMass
        assert_near_equal(
            prob[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS], 10806.675950702213, tol
        )
        # ElectricalMass
        assert_near_equal(prob[Aircraft.Electrical.MASS], 4277.63057018, tol)
        # AntiIcingMass
        assert_near_equal(prob[Aircraft.AntiIcing.MASS], 560.96510604, tol)
        # TransportAPUMass
        assert_near_equal(prob[Aircraft.APU.MASS], 2122.95946345, tol)
        # NonFlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 3810.0, tol)
        # FlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 450.0, tol)
        # TransportInstrumentMass
        assert_near_equal(prob[Aircraft.Instruments.MASS], 1300.50317605, tol)
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
        assert_near_equal(prob[Aircraft.Design.TOUCHDOWN_MASS_MAX], 699279.2, tol)
        # SurfaceControlMass
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 11701.85233893, tol)
        assert_near_equal(prob[Aircraft.Wing.CONTROL_SURFACE_AREA], 4015.82222875, tol)
        # BWBFuselageMass
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 136102.89191481, tol)
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
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_FACTOR], 4.00585433, tol)
        assert_near_equal(prob[Aircraft.Wing.ENG_POD_INERTIA_FACTOR], 1.0, tol)
        # BWBWingMiscMass
        assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 9811.77743845, tol)
        # WingShearControlMass
        assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 34818.23098605, tol)
        # WingBendingMass
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 9010.03041908, tol)
        # BWBAftBodyMass
        assert_near_equal(prob[Aircraft.Fuselage.AFTBODY_MASS], 18545.58195235, tol)
        assert_near_equal(prob[Aircraft.Wing.BWB_AFTBODY_MASS], 15392.83302045, tol)
        # MassSummation
        # StructureMass
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 239194.13868898, tol)
        # PropulsionMass
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 58895.60324675, tol)
        # SystemsEquipMass
        assert_near_equal(prob[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS], 90140.52027232, tol)
        # EmptyMass
        assert_near_equal(prob[Aircraft.Design.EMPTY_MASS], 388230.26220806, tol)
        # OperatingMass
        assert_near_equal(prob[Mission.OPERATING_MASS], 409221.81531426, tol)
        # ZeroFuelMass
        assert_near_equal(prob[Mission.ZERO_FUEL_MASS], 507033.81531426, tol)

    def test_case_all_subsystems(self):
        """
        premission: propulsion + geometry + aerodynamics + mass
        """
        prob = self.prob

        engines = [build_engine_deck(self.flops_inputs)]
        preprocess_options(self.flops_inputs)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines=engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
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
        # Aerodynamics
        # Design
        assert_near_equal(prob[Aircraft.Design.MACH], 0.89471085, tol)
        assert_near_equal(prob[Aircraft.Design.LIFT_COEFFICIENT], 0.42922096, tol)


@use_tempdirs
class BWB300PreMissionGroupCSVTest(unittest.TestCase):
    """
    testing using bwb_detailed_FLOPS.csv
    """

    def setUp(self):
        prob = self.prob = AviaryProblem()

        csv_path = 'models/aircraft/blended_wing_body/bwb300_baseline_FLOPS.csv'
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
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=geom_subsystem,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.run_model()

        tol = 1e-5
        # Geometry
        # BWBComputeDetailedWingDist
        assert_near_equal(prob[Aircraft.Wing.SPAN], 186.631829293424, tol)
        # DetailedCabinLayout
        assert_near_equal(prob[Aircraft.Fuselage.MAX_WIDTH], 50.17622787, tolerance=1e-9)
        assert_near_equal(prob[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH], 81.9534836, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 38.5, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CABIN_AREA], 3021.9507202, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MAX_HEIGHT], 20.9800918, tol)
        assert_near_equal(prob[Aircraft.BWB.NUM_BAYS], 4.0, tol)
        # BWBFuselagePrelim
        assert_near_equal(prob[Aircraft.Fuselage.REF_DIAMETER], 35.57815983, tol)
        assert_near_equal(prob[Aircraft.Fuselage.PLANFORM_AREA], 4317.07245743, tol)
        # BWBWingPrelim
        assert_near_equal(prob[Aircraft.Wing.AREA], 8454.35056774, tol)
        assert_near_equal(prob[Aircraft.Wing.ASPECT_RATIO], 4.82172761, tol)
        assert_near_equal(
            prob['AIRCRAFT_DATA_OVERRIDE:aircraft:wing:aspect_ratio_reference'], 4.82172761, tol
        )
        # assert_near_equal(prob[Aircraft.Wing.LOAD_FRACTION], 0.46761341784858923, tol)
        # _BWBWing
        assert_near_equal(prob[Aircraft.Wing.WETTED_AREA], 17370.05552974, tol)
        # _Tail
        assert_near_equal(prob[Aircraft.HorizontalTail.WETTED_AREA], 983.26501, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.WETTED_AREA], 125.0, tol)
        # _FuselageRatios
        assert_near_equal(prob[Aircraft.Fuselage.DIAMETER_TO_WING_SPAN], 0.19063286, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH_TO_DIAMETER], 3.29068186, tol)
        # Nacelles
        assert_near_equal(prob[Aircraft.Nacelle.WETTED_AREA], 613.74211034217353, tol)
        assert_near_equal(prob[Aircraft.Nacelle.TOTAL_WETTED_AREA], 2 * 613.7421103421735, tol)
        # Canard
        assert_near_equal(prob[Aircraft.Canard.WETTED_AREA], 0.0, tol)
        # BWBWingCharacteristicLength
        assert_near_equal(prob[Aircraft.Wing.CHARACTERISTIC_LENGTH], 45.29961797, tol)
        assert_near_equal(prob[Aircraft.Wing.FINENESS], 0.11, tol)
        # OtherCharacteristicLengths
        assert_near_equal(prob[Aircraft.Canard.CHARACTERISTIC_LENGTH], 0.0, tol)
        assert_near_equal(prob[Aircraft.Canard.FINENESS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Fuselage.CHARACTERISTIC_LENGTH], 117.07640514, tol)
        assert_near_equal(prob[Aircraft.Fuselage.FINENESS], 3.29068186, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH], 26.45751311065, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.FINENESS], 0.1, tol)
        assert_near_equal(prob[Aircraft.Nacelle.CHARACTERISTIC_LENGTH], [17.367966592444596], tol)
        assert_near_equal(prob[Aircraft.Nacelle.FINENESS], [1.3761635770546583], tol)
        # TotalWettedArea
        assert_near_equal(prob[Aircraft.Design.TOTAL_WETTED_AREA], 19705.80476042, tol)

    def test_case_geom_mass(self):
        """
        premission: geometry + mass
        """
        prob = self.prob

        preprocess_options(self.flops_inputs)
        geom_mass_subsystems = get_geom_and_mass_subsystems('FLOPS')

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=geom_mass_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.set_val(Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, val=87500.0 * 2, units='lbf')

        prob.run_model()

        # Only masses are checked because geometry is checked in test_case_geom() already.
        tol = 1e-4
        # Mass
        # CargoMass
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_MASS_TOTAL], 49500.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.BAGGAGE_MASS], 13200.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 62700.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 62700.0, tol)
        # TransportCargoContainersMass
        assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 23500.0, tol)
        # TransportEngineCtrlsMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS], 153.81807436, tol)
        # TransportAvionicsMass
        assert_near_equal(prob[Aircraft.Avionics.MASS], 2290.95007773, tol)
        # FuelCapacityGroup
        assert_near_equal(prob[Aircraft.Fuel.WING_FUEL_CAPACITY], 793608.88038917, tol)
        assert_near_equal(prob[Aircraft.Fuel.TOTAL_CAPACITY], 793608.88038917, tol)
        # EngineMass
        assert_near_equal(prob[Aircraft.Engine.MASS], 44541.857940875525 / 2, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 44541.857940875525, tol)
        # TransportFuelSystemMass
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 3673.16899646, tol)
        # TransportAirCondMass
        assert_near_equal(prob[Aircraft.AirConditioning.MASS], 3807.20118967, tol)
        # TransportEngineOilMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 267.39241429019251, tol)
        # BWBFurnishingsGroupMass
        assert_near_equal(prob[Aircraft.Furnishings.MASS], 52410.94244865, tol)
        # TransportHydraulicsGroupMass
        assert_near_equal(prob[Aircraft.Hydraulics.MASS], 3996.6384617, tol)
        # PassengerServiceMass
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS], 7029.593528180887, tol)
        # ElectricalMass
        assert_near_equal(prob[Aircraft.Electrical.MASS], 2654.06975307, tol)
        # AntiIcingMass
        assert_near_equal(prob[Aircraft.AntiIcing.MASS], 400.99917573, tol)
        # TransportAPUMass
        assert_near_equal(prob[Aircraft.APU.MASS], 1581.00217211, tol)
        # NonFlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.CABIN_CREW_MASS], 1640.0, tol)
        # FlightCrewMass
        assert_near_equal(prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 450.0, tol)
        # TransportInstrumentMass
        assert_near_equal(prob[Aircraft.Instruments.MASS], 967.59344999, tol)
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
        assert_near_equal(prob[Aircraft.Design.TOUCHDOWN_MASS_MAX], 420000.0, tol)
        # SurfaceControlMass
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 8112.00284687, tol)
        assert_near_equal(prob[Aircraft.Wing.CONTROL_SURFACE_AREA], 2536.30517032, tol)
        # BWBFuselageMass
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 81157.44843467, tol)
        # HorizontalTailMass
        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 6444.9988831532046, tol)
        # VerticalTailMass
        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 0.0, tol)
        # CanardMass
        assert_near_equal(prob[Aircraft.Canard.MASS], 0.0, tol)
        # FinMass
        assert_near_equal(prob[Aircraft.Fins.MASS], 2822.1415450307886, tol)
        # WingMassGroup
        # BWBDetailedWingBendingFact, In FLOPS run, 6.7996347825592336
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_FACTOR], 6.75079525, tol)
        assert_near_equal(prob[Aircraft.Wing.ENG_POD_INERTIA_FACTOR], 1.0, tol)
        # BWBWingMiscMass
        assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 6938.99205472, tol)
        # WingShearControlMass
        assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 24493.35004333, tol)
        # WingBendingMass
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 17876.82972382, tol)
        # BWBAftBodyMass
        assert_near_equal(prob[Aircraft.Fuselage.AFTBODY_MASS], 10478.08779866, tol)
        assert_near_equal(prob[Aircraft.Wing.BWB_AFTBODY_MASS], 8964.00411176, tol)
        # MassSummation
        # StructureMass 158921.83401643133
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 167990.52160543, tol)
        # PropulsionMass
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 48215.02693733, tol)
        # SystemsEquipMass
        assert_near_equal(prob[Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS], 76221.39957552, tol)
        # EmptyMass
        assert_near_equal(prob[Aircraft.Design.EMPTY_MASS], 292426.94811828, tol)
        # OperatingMass
        assert_near_equal(prob[Mission.OPERATING_MASS], 326632.14480333, tol)
        # ZeroFuelMass
        assert_near_equal(prob[Mission.ZERO_FUEL_MASS], 389332.14480333, tol)
        # FinMass
        assert_near_equal(prob[Aircraft.Fins.MASS], 2822.14154503, tol)

    def test_case_all_subsystems(self):
        """
        premission: propulsion + geometry + aerodynamics + mass
        """
        prob = self.prob

        engines = [build_engine_deck(self.flops_inputs)]
        preprocess_options(self.flops_inputs)
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines=engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.flops_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.flops_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.flops_inputs)

        prob.run_model()

        # Only aero parameters are checked because geometry and mass are checked in test_case_geom() already.
        tol = 1e-4
        # Design
        assert_near_equal(prob[Aircraft.Design.MACH], 0.8995951, tol)
        assert_near_equal(prob[Aircraft.Design.LIFT_COEFFICIENT], 0.40724438, tol)


if __name__ == '__main__':
    unittest.main()
