import unittest

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.utils.test_utils.default_subsystems import (
    get_default_premission_subsystems,
    get_geom_and_mass_subsystems,
)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission


@use_tempdirs
class PreMissionGroupTest(unittest.TestCase):
    def setUp(self):
        prob = self.prob = AviaryProblem()

        csv_path = 'models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv'
        self.gasp_inputs = prob.load_inputs(csv_path)
        self.gasp_inputs.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )
        prob.check_and_preprocess_inputs()

    def test_case1(self):
        """premission: geometry + mass."""
        prob = self.prob
        preprocess_options(self.gasp_inputs)
        geom_and_mass_subsystems = get_geom_and_mass_subsystems('GASP')

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=self.gasp_inputs, subsystems=geom_and_mass_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.set_val(Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.3648, units='unitless')
        prob.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf')

        prob.run_model()

        tol = 1e-5
        # propulsion subsystem
        # geometry subsystem
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 129.497, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4000.0, tol)
        assert_near_equal(prob[Aircraft.Wing.AREA], 1370.3125, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 117.81878299, tol)
        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 17.48974356, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 12.61453233, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 16.40711451, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.139656, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114.0056, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], 375.8798, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.SPAN], 42.2543, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.1592, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.5768, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.6793, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 469.3183, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.SPAN], 27.9957, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.ROOT_CHORD], 18.6162, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.8321, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MOMENT_ARM], 49.8809, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], 7.25, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], 14.5, tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], 330.2599, tol)
        # mass subsystem
        assert_near_equal(self.prob[Aircraft.Design.LIFT_CURVE_SLOPE], 6.39471, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.22129, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 36000.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 36000.0, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12605.94, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.MASS], 990.7798, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 2276.1316, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 2297.9697, tol)
        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 4740.1241, tol)
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 3819.3564, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 3682.099, tol)
        assert_near_equal(prob[Aircraft.LandingGear.TOTAL_MASS], 7489.8343, tol)
        assert_near_equal(prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6366.3591, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 21078.3911, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5341.4317956, tol)

        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 850.90095, tol)
        assert_near_equal(prob[Aircraft.Wing.MASS], 16206.8122, tol)
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1740.2606, tol)
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 50667.4376, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 18673.0352, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS_REQUIRED], 42445.3806, tol)
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 16048.0025, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 42445.3806, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 848.5301, tol)
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 96954.6194, tol)
        assert_near_equal(prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)

    def test_case2(self):
        """premission: propulsion + geometry + aerodynamics + mass"""

        prob = self.prob
        engines = [build_engine_deck(self.gasp_inputs)]
        preprocess_options(self.gasp_inputs, engine_models=engines)
        default_premission_subsystems = get_default_premission_subsystems('GASP', engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs, subsystems=default_premission_subsystems
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.run_model()

        tol = 1e-5
        # propulsion subsystem
        assert_near_equal(prob[Aircraft.Engine.SCALED_SLS_THRUST], 28690.0, tol)
        # geometry subsystem
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 157.2, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 129.497, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4000.0, tol)
        assert_near_equal(prob[Aircraft.Wing.AREA], 1370.3125, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 117.81878299, tol)
        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 17.48974356, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 12.61453233, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 16.40711451, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.139656, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114.0056, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], 375.8798, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.SPAN], 42.2543, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.ROOT_CHORD], 13.1592, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.5768, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MOMENT_ARM], 54.6793, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 469.3183, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.SPAN], 27.9957, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.ROOT_CHORD], 18.6162, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.8321, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MOMENT_ARM], 49.8809, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], 7.25, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], 14.5, tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], 330.2599, tol)
        # aerodynamics subsystem
        assert_near_equal(prob[Mission.Landing.LIFT_COEFFICIENT_MAX], 2.82008, tol)
        # mass subsystem
        assert_near_equal(self.prob[Aircraft.Design.LIFT_CURVE_SLOPE], 6.39471, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.75, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.22129, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 36000.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 36000.0, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12259.8108, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.MASS], 990.7798, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 2276.1316, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 2297.9697, tol)
        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 4158.5858, tol)
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 3819.3564, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 3682.099, tol)
        assert_near_equal(prob[Aircraft.LandingGear.TOTAL_MASS], 7489.8343, tol)
        assert_near_equal(prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6366.3591, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 21078.3911, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5332.684, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 827.5372, tol)
        assert_near_equal(prob[Aircraft.Wing.MASS], 15649.1082, tol)
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1779.1662, tol)
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 50081.2197, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 18675.0408, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS_REQUIRED], 43394.2976, tol)
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 15694.0515, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 43394.2976, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 867.5, tol)
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 96005.7024, tol)
        assert_near_equal(prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)


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
        Aircraft.Controls.TOTAL_MASS -- WFC = 2115
        Aircraft.Wing.SURFACE_CONTROL_MASS -- not in GASP
        Aircraft.LandingGear.TOTAL_MASS -- WLG = 7800
        Aircraft.LandingGear.MAIN_GEAR_MASS -- WMG = 6630
        Aircraft.Avionics.MASS -- CW(5) = 3225.0
        Aircraft.AirConditioning.MASS -- WAC = 1301.57
        Aircraft.Furnishings.MASS -- 11269.88
        Aircraft.Design.FIXED_EQUIPMENT_MASS -- WFE = 20876.
        Aircraft.Design.FIXED_USEFUL_LOAD -- WFUL = 5775.
        Aircraft.Engine.ADDITIONAL_MASS -- not in GASP
        Aircraft.Wing.FOLD_MASS -- WWFOLD = 107.9
        Aircraft.Wing.MASS -- WW = 7645.
        Aircraft.Fuel.FUEL_SYSTEM_MASS -- WFSS = 1281.
          Note: In GASP, fuel related masses are based on sized engine.
                See the computation of Aircraft.Proplusion.TOTAL_ENGINE_POD_MASS
                in FuelMassGroup closure loop.
        Aircraft.Design.STRUCTURE_MASS -- WST = 45623.
        Aircraft.Fuselage.MASS -- WB = 27160
        Mission.Summary.FUEL_MASS_REQUIRED tol -- WFAREQ = 36595.0
        Aircraft.Propulsion.MASS tol -- WP = 8592.
        Mission.Summary.FUEL_MASS -- WFADES = 33268.2
        Aircraft.Fuel.WING_VOLUME_DESIGN -- FVOLREQ = 731.6
        Mission.Summary.OPERATING_MASS tol -- OWE = 82982.
        Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY -- not in GASP
        """
        prob = self.prob

        engines = [build_engine_deck(self.gasp_inputs)]
        preprocess_options(self.gasp_inputs, engine_models=engines)
        default_premission_subsystems = get_default_premission_subsystems('GASP', engines=engines)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(
                aviary_options=self.gasp_inputs, subsystems=default_premission_subsystems
            ),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False)
        set_aviary_initial_values(prob, self.gasp_inputs)

        prob.run_model()

        tol = 1e-5
        # geometry subsystem
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 38, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 71.52455, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4573.882, tol)
        assert_near_equal(prob[Aircraft.Wing.AREA], 2142.857, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 146.385, tol)
        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 22.97244, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 16.22, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 20.3337, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.135966, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.9078, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], 0.00117064, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.SPAN], 0.04467601, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.ROOT_CHORD], 0.0383645, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 0.0280845, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MOMENT_ARM], 29.6907, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 169.1196, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.SPAN], 16.98084, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.ROOT_CHORD], 14.5819, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AVERAGE_CHORD], 10.6746, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MOMENT_ARM], 27.8219, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], 5.33382, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], 7.2476, tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], 121.4458, tol)
        # mass subsystem
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 1686.6256, tol)
        assert_near_equal(prob[Aircraft.Design.LIFT_CURVE_SLOPE], 5.948, tol)
        assert_near_equal(prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.77336, tol)
        assert_near_equal(prob[Aircraft.Wing.MATERIAL_FACTOR], 1.194612, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 7005.15475, tol)
        assert_near_equal(prob[Aircraft.Nacelle.MASS], 303.6144, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 1.02402, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 864.174, tol)
        assert_near_equal(prob[Aircraft.Wing.HIGH_LIFT_MASS], 973.10188, tol)
        assert_near_equal(prob[Aircraft.Controls.TOTAL_MASS], 2114.982, tol)
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 1986.251, tol)
        assert_near_equal(prob[Aircraft.LandingGear.TOTAL_MASS], 7800.0, tol)
        assert_near_equal(prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6630.0, tol)
        assert_near_equal(prob[Aircraft.Avionics.MASS], 3225.0, tol)
        assert_near_equal(prob[Aircraft.AirConditioning.MASS], 1301.573, tol)
        assert_near_equal(prob[Aircraft.Furnishings.MASS], 11269.876, tol)
        assert_near_equal(prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 20876.453, tol)
        assert_near_equal(prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5971.7946, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 153.1677, tol)
        assert_near_equal(prob[Aircraft.Wing.FOLD_MASS], 107.85313, tol)
        assert_near_equal(prob[Aircraft.Wing.MASS], 6960.992, tol)
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1316.1836, tol)
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 44472.509, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 27159.693, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS_REQUIRED], 34186.5881, tol)
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 8627.6738, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 34186.588, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 751.7705, tol)
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 82063.412, tol)
        assert_near_equal(prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 3877.719, tol)

    def test_case2(self):
        """
        premission: geometry + mass
        """
        prob = self.prob

        preprocess_options(self.gasp_inputs)
        geom_and_mass_subsystems = get_geom_and_mass_subsystems('GASP')

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=self.gasp_inputs, subsystems=geom_and_mass_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, self.gasp_inputs)
        prob.setup(check=False)
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
        # geometry subsystem
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 38, tol)
        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 71.52455, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4573.882, tol)
        assert_near_equal(prob[Aircraft.Wing.AREA], 2142.857, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 146.385, tol)
        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 22.97244, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 16.22, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 20.3337, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.135966, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 605.9078, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], 0.00117064, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.SPAN], 0.04467601, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.ROOT_CHORD], 0.0383645, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 0.0280845, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MOMENT_ARM], 29.6907, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 169.1196, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.SPAN], 16.98084, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.ROOT_CHORD], 14.5819, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AVERAGE_CHORD], 10.6746, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MOMENT_ARM], 27.8219, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], 5.33382, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], 7.2476, tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], 121.4458, tol)
        # mass subsystem
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 1686.6256, tol)
        assert_near_equal(prob[Aircraft.Design.LIFT_CURVE_SLOPE], 5.948, tol)
        assert_near_equal(prob[Aircraft.Wing.ULTIMATE_LOAD_FACTOR], 3.77336, tol)
        assert_near_equal(prob[Aircraft.Wing.MATERIAL_FACTOR], 1.194612, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 7005.155, tol)
        assert_near_equal(prob[Aircraft.Nacelle.MASS], 303.6144, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 1.02402, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 864.174, tol)
        assert_near_equal(prob[Aircraft.Wing.HIGH_LIFT_MASS], 971.8248, tol)
        assert_near_equal(prob[Aircraft.Controls.TOTAL_MASS], 2114.982, tol)
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 1986.251, tol)
        assert_near_equal(prob[Aircraft.LandingGear.TOTAL_MASS], 7800.0, tol)
        assert_near_equal(prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6630.0, tol)
        assert_near_equal(prob[Aircraft.Avionics.MASS], 3225.0, tol)
        assert_near_equal(prob[Aircraft.AirConditioning.MASS], 1301.573, tol)
        assert_near_equal(prob[Aircraft.Furnishings.MASS], 11269.876, tol)
        assert_near_equal(prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 20876.453, tol)
        assert_near_equal(prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5971.7946, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 153.1677, tol)
        assert_near_equal(prob[Aircraft.Wing.FOLD_MASS], 107.8335, tol)
        assert_near_equal(prob[Aircraft.Wing.MASS], 6959.7262, tol)
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1316.2306, tol)
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 44471.243, tol)
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 27159.693, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS_REQUIRED], 34187.8, tol)
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 8627.72, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 34187.807, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 751.7973, tol)
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 82062.193, tol)
        assert_near_equal(prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 3878.938, tol)


if __name__ == '__main__':
    unittest.main()
