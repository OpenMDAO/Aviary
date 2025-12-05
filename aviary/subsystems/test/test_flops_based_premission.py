import unittest

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

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
                Aircraft.Fuselage.AVG_DIAMETER,
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
                Aircraft.Fuselage.PLANFORM_AREA,
            ],
            version=Version.BWB,
            step=1.01e-40,
            atol=1e-6,
            rtol=1e-6,
            check_partials=True,
            excludes=['*detailed_wing.*'],  # does not work?
        )


if __name__ == '__main__':
    unittest.main()
