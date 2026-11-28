import unittest

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import get_default_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class PreMissionGroupTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = AviaryProblem()

        csv_path = 'validation_cases/validation_data/test_models/aircraft_for_bench_GwGm.csv'
        self.gasp_inputs = prob.load_inputs(csv_path)
        self.gasp_inputs.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )
        prob.check_and_preprocess_inputs()

    def test_case1(self):
        """premission: geometry + mass."""
        prob = self.prob
        preprocess_options(self.gasp_inputs)
        geom_and_mass_subsystems = [
            get_default_subsystems('GASP')[k] for k in ['geometry', 'mass', 'energy']
        ]

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs,
                subsystems=geom_and_mass_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False, force_alloc_complex=True)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.3648, units='unitless')
        prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf')

        prob.run_model()

        tol = 1e-5
        expected_values = {
            # geometry subsystem
            Aircraft.Fuselage.AVG_DIAMETER: 157.2,
            Aircraft.Fuselage.LENGTH: 129.497,
            Aircraft.Fuselage.WETTED_AREA: 4000.0,
            Aircraft.Wing.AREA: 1370.3125,
            Aircraft.Wing.SPAN: 117.81878299,
            Aircraft.Wing.CENTER_CHORD: 17.48974356,
            Aircraft.Wing.AVERAGE_CHORD: 12.61453233,
            Aircraft.Wing.ROOT_CHORD: 16.40711451,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.139656,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 1114.0056,
            Aircraft.HorizontalTail.AREA: 375.8798,
            Aircraft.HorizontalTail.SPAN: 42.2543,
            Aircraft.HorizontalTail.ROOT_CHORD: 13.1592,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 9.5768,
            Aircraft.HorizontalTail.MOMENT_ARM: 54.6793,
            Aircraft.VerticalTail.AREA: 469.3183,
            Aircraft.VerticalTail.SPAN: 27.9957,
            Aircraft.VerticalTail.ROOT_CHORD: 18.6162,
            Aircraft.VerticalTail.AVERAGE_CHORD: 16.8321,
            Aircraft.VerticalTail.MOMENT_ARM: 49.8809,
            Aircraft.Nacelle.AVG_DIAMETER: 7.25,
            Aircraft.Nacelle.AVG_LENGTH: 14.5,
            Aircraft.Nacelle.WETTED_AREA: 330.2599,
            # mass subsystem
            Aircraft.Design.LIFT_CURVE_SLOPE: 6.39471,
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR: 3.75,
            Aircraft.Wing.MATERIAL_FACTOR: 1.22129,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 36000.0,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: 36000.0,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12605.94,
            Aircraft.Nacelle.MASS: 990.7798,
            Aircraft.HorizontalTail.MASS: 2275.99370556,
            Aircraft.VerticalTail.MASS: 2297.83041065,
            Aircraft.Wing.HIGH_LIFT_MASS: 4740.1241,
            Aircraft.Controls.MASS: 3817.57054461,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 3680.31304736,
            Aircraft.LandingGear.TOTAL_MASS: 7489.8343,
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6366.3591,
            # GASP original 21078.3911, added controls mass
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS: 24895.76217942,
            Mission.OPERATING_ITEMS_MASS: 5341.4317956,
            Aircraft.Engine.ADDITIONAL_MASS: 850.90095,
            Aircraft.Wing.MASS: 16206.8122,
            Aircraft.Fuel.FUEL_SYSTEM_MASS: 1740.39915666,
            Aircraft.Design.STRUCTURE_MASS: 50665.90504969,
            Aircraft.Fuselage.MASS: 18671.74985618,
            Aircraft.Propulsion.MASS: 16048.0025,
            Aircraft.Fuel.WING_VOLUME_DESIGN: 848.5976721,
            Mission.OPERATING_MASS: 96951.24008137,
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY: 0,
        }

        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=3e-9)

    def test_case2(self):
        """premission: propulsion + geometry + aerodynamics + mass."""
        prob = self.prob
        engines = [build_engine_deck(self.gasp_inputs)]
        preprocess_options(self.gasp_inputs, engine_models=engines)
        default_premission_subsystems = get_default_subsystems('GASP', engines).values()

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.run_model()

        tol = 1e-5
        expected_values = {
            # propulsion subsystem
            Aircraft.Engine.SCALED_SLS_THRUST: 28690.0,
            # geometry subsystem
            Aircraft.Fuselage.AVG_DIAMETER: 157.2,
            Aircraft.Fuselage.LENGTH: 129.497,
            Aircraft.Fuselage.WETTED_AREA: 4000.0,
            Aircraft.Wing.AREA: 1370.3125,
            Aircraft.Wing.SPAN: 117.81878299,
            Aircraft.Wing.CENTER_CHORD: 17.48974356,
            Aircraft.Wing.AVERAGE_CHORD: 12.61453233,
            Aircraft.Wing.ROOT_CHORD: 16.40711451,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.139656,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 1114.0056,
            Aircraft.HorizontalTail.AREA: 375.8798,
            Aircraft.HorizontalTail.SPAN: 42.2543,
            Aircraft.HorizontalTail.ROOT_CHORD: 13.1592,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 9.5768,
            Aircraft.HorizontalTail.MOMENT_ARM: 54.6793,
            Aircraft.VerticalTail.AREA: 469.3183,
            Aircraft.VerticalTail.SPAN: 27.9957,
            Aircraft.VerticalTail.ROOT_CHORD: 18.6162,
            Aircraft.VerticalTail.AVERAGE_CHORD: 16.8321,
            Aircraft.VerticalTail.MOMENT_ARM: 49.8809,
            Aircraft.Nacelle.AVG_DIAMETER: 7.25,
            Aircraft.Nacelle.AVG_LENGTH: 14.5,
            Aircraft.Nacelle.WETTED_AREA: 330.2599,
            # aerodynamics subsystem
            Mission.Landing.LIFT_COEFFICIENT_MAX: 2.8179491,
            # mass subsystem
            Aircraft.Design.LIFT_CURVE_SLOPE: 6.39471,
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR: 3.75,
            Aircraft.Wing.MATERIAL_FACTOR: 1.22129,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 36000.0,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: 36000.0,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12259.8108,
            Aircraft.Nacelle.MASS: 990.7798,
            Aircraft.HorizontalTail.MASS: 2275.99370556,
            Aircraft.VerticalTail.MASS: 2297.83041065,
            Aircraft.Wing.HIGH_LIFT_MASS: 4161.22777613,
            Aircraft.Controls.MASS: 3817.57054461,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 3680.31304736,
            Aircraft.LandingGear.TOTAL_MASS: 7489.8343,
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6366.3591,
            # GASP original 21078.3911, added controls mass
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS: 24895.76217942,
            Mission.OPERATING_ITEMS_MASS: 5332.684,
            Aircraft.Engine.ADDITIONAL_MASS: 827.5372,
            Aircraft.Wing.MASS: 15651.64198957,
            Aircraft.Fuel.FUEL_SYSTEM_MASS: 1779.20577868,
            Aircraft.Design.STRUCTURE_MASS: 50082.20009363,
            Aircraft.Fuselage.MASS: 18673.74849782,
            Aircraft.Propulsion.MASS: 15694.0515,
            Aircraft.Fuel.WING_VOLUME_DESIGN: 867.51931371,
            Mission.OPERATING_MASS: 96004.73710533,
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY: 0,
        }

        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=3e-9)


@use_tempdirs
class BWBPreMissionGroupTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = AviaryProblem()

        csv_path = 'models/aircraft/blended_wing_body/generic_BWB_GASP.csv'
        self.gasp_inputs = prob.load_inputs(csv_path)
        prob.check_and_preprocess_inputs()

    def test_case1(self):
        """
        premission: propulsion + geometry + aerodynamics + mass
        Testing GASP data case:
        Aircraft.Design.LIFT_CURVE_SLOPE -- CLALPH = 6.515
          Note: In GASP, CLALPH is first calculated in CLA() and get 5.9485 and
                later updated in CLIFT() and get 6.515.
        Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS -- WPES = 2055
        Aircraft.Wing.ULTIMATE_LOAD_FACTOR -- ULF = 3.7734
        Aircraft.Wing.MATERIAL_FACTOR -- SKNO = 1.19461238
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS -- WPL = 33750
        Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS -- not in GASP
        Aircraft.Propulsion.TOTAL_ENGINE_MASS -- WEP = 7005.
        Aircraft.Nacelle.MASS -- WNAC = 303.6144075 by hand computation
        Aircraft.HorizontalTail.MASS -- WHT = 1
        Aircraft.VerticalTail.MASS -- WVT = 864
        Aircraft.Wing.HIGH_LIFT_MASS -- WHLDEV = 974.0
        Aircraft.Controls.MASS -- WFC = 2115
        Aircraft.Wing.SURFACE_CONTROL_MASS -- not in GASP
        Aircraft.LandingGear.TOTAL_MASS -- WLG = 7800
        Aircraft.LandingGear.MAIN_GEAR_MASS -- WMG = 6630
        Aircraft.Avionics.MASS -- CW(5) = 3225.0
        Aircraft.AirConditioning.MASS -- WAC = 1301.57
        Aircraft.Furnishings.MASS -- 11269.88
        Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS -- WFE = 20876.
        Mission.OPERATING_ITEMS_MASS -- WFUL = 5775.
        Aircraft.Engine.ADDITIONAL_MASS -- not in GASP
        Aircraft.Wing.FOLD_MASS -- WWFOLD = 107.9
        Aircraft.Wing.MASS -- WW = 7645.
        Aircraft.Fuel.FUEL_SYSTEM_MASS -- WFSS = 1281.
          Note: In GASP, fuel related masses are based on sized engine.
                See the computation of Aircraft.Proplusion.TOTAL_ENGINE_POD_MASS
                in FuelMassGroup closure loop.
        Aircraft.Design.STRUCTURE_MASS -- WST = 45623.
        Aircraft.Fuselage.MASS -- WB = 27160
        FUEL_MASS_REQUIRED tol -- WFAREQ = 36595.0
        Aircraft.Propulsion.MASS tol -- WP = 8592.
        FUEL_MASS -- WFADES = 33268.2
        Aircraft.Fuel.WING_VOLUME_DESIGN -- FVOLREQ = 731.6
        Mission.OPERATING_MASS tol -- OWE = 82982.
        Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY -- not in GASP.
        """
        prob = self.prob

        engines = [build_engine_deck(self.gasp_inputs)]
        preprocess_options(self.gasp_inputs, engine_models=engines)
        default_premission_subsystems = get_default_subsystems('GASP', engines=engines).values()

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs,
                subsystems=default_premission_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False, force_alloc_complex=True)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.run_model()

        tol = 1e-5
        expected_values = {
            # geometry subsystem
            Aircraft.Fuselage.AVG_DIAMETER: 38,
            Aircraft.Fuselage.LENGTH: 71.52455,
            Aircraft.Fuselage.WETTED_AREA: 4573.882,
            Aircraft.Wing.AREA: 2142.857,
            Aircraft.Wing.SPAN: 146.385,
            Aircraft.Wing.CENTER_CHORD: 22.97244,
            Aircraft.Wing.AVERAGE_CHORD: 16.22,
            Aircraft.Wing.ROOT_CHORD: 20.3337,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.135966,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 605.9078,
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT: 1.0e-06,
            Aircraft.HorizontalTail.AREA: 0.00117064,
            Aircraft.HorizontalTail.SPAN: 0.04467601,
            Aircraft.HorizontalTail.ROOT_CHORD: 0.0383645,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 0.0280845,
            Aircraft.HorizontalTail.MOMENT_ARM: 29.6907,
            Aircraft.VerticalTail.VOLUME_COEFFICIENT: 0.015,
            Aircraft.VerticalTail.AREA: 169.1196,
            Aircraft.VerticalTail.SPAN: 16.98084,
            Aircraft.VerticalTail.ROOT_CHORD: 14.5819,
            Aircraft.VerticalTail.AVERAGE_CHORD: 10.6746,
            Aircraft.VerticalTail.MOMENT_ARM: 27.8219,
            Aircraft.Nacelle.AVG_DIAMETER: 5.33382,
            Aircraft.Nacelle.AVG_LENGTH: 7.2476,
            Aircraft.Nacelle.WETTED_AREA: 121.4458,
            # mass subsystem
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 1686.6256,
            Aircraft.Design.LIFT_CURVE_SLOPE: 5.948,
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR: 3.77181925,
            Aircraft.Wing.MATERIAL_FACTOR: 1.194612,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 33750.0,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: 33750.0,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 7005.15475,
            Aircraft.Nacelle.MASS: 303.6144,
            Aircraft.HorizontalTail.MASS: 1.02395879,
            Aircraft.VerticalTail.MASS: 864.1216559,
            Aircraft.Wing.HIGH_LIFT_MASS: 974.436,
            Aircraft.Controls.MASS: 2113.53974468,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 1984.80927303,
            Aircraft.LandingGear.TOTAL_MASS: 7800.0,
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6630.0,
            Aircraft.Avionics.MASS: 3225.0,
            Aircraft.AirConditioning.MASS: 1301.573,
            Aircraft.Furnishings.MASS: 11269.876,
            # GASP original 20876.453, added controls mass
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS: 22989.83829075,
            Mission.OPERATING_ITEMS_MASS: 5971.7946,
            Aircraft.Engine.ADDITIONAL_MASS: 153.1677,
            Aircraft.Wing.FOLD_MASS: 107.84571776,
            Aircraft.Wing.MASS: 6960.51388998,
            Aircraft.Fuel.FUEL_SYSTEM_MASS: 1316.26186968,
            Aircraft.Design.STRUCTURE_MASS: 44471.97975434,
            Aircraft.Fuselage.MASS: 27159.693,
            Aircraft.Propulsion.MASS: 8627.76690337,
            Aircraft.Fuel.WING_VOLUME_DESIGN: 751.81517038,
            Mission.OPERATING_MASS: 82061.38000832,
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY: 3879.75122799,
        }

        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=3e-9)

    def test_case_geom(self):
        """premission: geometry."""
        prob = self.prob

        preprocess_options(self.gasp_inputs)
        geom_subsystem = [get_default_subsystems('GASP')['geometry']]

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs,
                subsystems=geom_subsystem,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False, force_alloc_complex=True)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.run_model()

        tol = 1e-5
        expected_values = {
            # geometry subsystem
            Aircraft.Fuselage.AVG_DIAMETER: 38,
            Aircraft.Fuselage.LENGTH: 71.52455,
            Aircraft.Fuselage.WETTED_AREA: 4573.882,
            Aircraft.Wing.AREA: 2142.857,
            Aircraft.Wing.SPAN: 146.385,
            Aircraft.Wing.CENTER_CHORD: 22.97244,
            Aircraft.Wing.AVERAGE_CHORD: 16.22,
            Aircraft.Wing.ROOT_CHORD: 20.3337,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.135966,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 605.9078,
            Aircraft.HorizontalTail.AREA: 0.00117064,
            Aircraft.HorizontalTail.SPAN: 0.04467601,
            Aircraft.HorizontalTail.ROOT_CHORD: 0.0383645,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 0.0280845,
            Aircraft.HorizontalTail.MOMENT_ARM: 29.6907,
            Aircraft.VerticalTail.AREA: 169.1196,
            Aircraft.VerticalTail.SPAN: 16.98084,
            Aircraft.VerticalTail.ROOT_CHORD: 14.5819,
            Aircraft.VerticalTail.AVERAGE_CHORD: 10.6746,
            Aircraft.VerticalTail.MOMENT_ARM: 27.8219,
            Aircraft.Nacelle.AVG_DIAMETER: 5.33382,
            Aircraft.Nacelle.AVG_LENGTH: 7.2476,
            Aircraft.Nacelle.WETTED_AREA: 121.4458,
        }

        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=3e-9)

    def test_case_geom_mass(self):
        """premission: geometry + mass."""
        prob = self.prob

        preprocess_options(self.gasp_inputs)
        geom_and_mass_subsystems = [
            get_default_subsystems('GASP')[k] for k in ['geometry', 'mass', 'energy']
        ]

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs,
                subsystems=geom_and_mass_subsystems,
                subsystem_options={},
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False, force_alloc_complex=True)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.set_val(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.13421583, units='unitless'
        )  # 2.13421583 for landing, 1.94302452 for takeoff
        prob.set_val(
            Aircraft.Engine.SCALED_SLS_THRUST, val=19580.1602, units='lbf'
        )  # not 37451.0 as in .dat file. It is computed in propulsion_premission.py
        prob.set_val(
            Aircraft.Wing.SLAT_SPAN_RATIO, 0.827296853, units='unitless'
        )  # It is computed in basic_calculations.py

        prob.run_model()

        tol = 1e-5
        expected_values = {
            # geometry subsystem
            Aircraft.Fuselage.AVG_DIAMETER: 38,
            Aircraft.Fuselage.LENGTH: 71.52455,
            Aircraft.Fuselage.WETTED_AREA: 4573.882,
            Aircraft.Wing.AREA: 2142.857,
            Aircraft.Wing.SPAN: 146.385,
            Aircraft.Wing.CENTER_CHORD: 22.97244,
            Aircraft.Wing.AVERAGE_CHORD: 16.22,
            Aircraft.Wing.ROOT_CHORD: 20.3337,
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: 0.135966,
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX: 605.9078,
            Aircraft.HorizontalTail.AREA: 0.00117064,
            Aircraft.HorizontalTail.SPAN: 0.04467601,
            Aircraft.HorizontalTail.ROOT_CHORD: 0.0383645,
            Aircraft.HorizontalTail.AVERAGE_CHORD: 0.0280845,
            Aircraft.HorizontalTail.MOMENT_ARM: 29.6907,
            Aircraft.VerticalTail.AREA: 169.1196,
            Aircraft.VerticalTail.SPAN: 16.98084,
            Aircraft.VerticalTail.ROOT_CHORD: 14.5819,
            Aircraft.VerticalTail.AVERAGE_CHORD: 10.6746,
            Aircraft.VerticalTail.MOMENT_ARM: 27.8219,
            Aircraft.Nacelle.AVG_DIAMETER: 5.33382,
            Aircraft.Nacelle.AVG_LENGTH: 7.2476,
            Aircraft.Nacelle.WETTED_AREA: 121.4458,
            # mass subsystem
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 1686.6256,
            Aircraft.Design.LIFT_CURVE_SLOPE: 5.948,
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR: 3.77181925,
            Aircraft.Wing.MATERIAL_FACTOR: 1.194612,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 33750.0,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: 33750.0,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 7005.155,
            Aircraft.Nacelle.MASS: 303.6144,
            Aircraft.HorizontalTail.MASS: 1.02395879,
            Aircraft.VerticalTail.MASS: 864.1216559,
            Aircraft.Wing.HIGH_LIFT_MASS: 971.8248,
            Aircraft.Controls.MASS: 2113.53974468,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 1984.80927303,
            Aircraft.LandingGear.TOTAL_MASS: 7800.0,
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6630.0,
            Aircraft.Avionics.MASS: 3225.0,
            Aircraft.AirConditioning.MASS: 1301.573,
            Aircraft.Furnishings.MASS: 11269.876,
            # GASP original 20876.453, added controls mass
            Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS: 22989.83829075,
            Mission.OPERATING_ITEMS_MASS: 5971.7946,
            Aircraft.Engine.ADDITIONAL_MASS: 153.1677,
            Aircraft.Wing.FOLD_MASS: 107.80567539,
            Aircraft.Wing.MASS: 6957.9294994,
            Aircraft.Fuel.FUEL_SYSTEM_MASS: 1316.35830201,
            Aircraft.Design.STRUCTURE_MASS: 44469.39387775,
            Aircraft.Fuselage.MASS: 27159.693,
            Aircraft.Propulsion.MASS: 8627.84847386,
            Aircraft.Fuel.WING_VOLUME_DESIGN: 751.87025007,
            Mission.OPERATING_MASS: 82058.87527237,
            Aircraft.Fuel.AUXILIARY_FUEL_MASS_CAPACITY: 3882.25596394,
        }

        for var_name, expected in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=3e-9)


if __name__ == '__main__':
    unittest.main()
