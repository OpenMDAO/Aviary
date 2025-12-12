import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.models.aircraft.large_single_aisle_1.V3_bug_fixed_IO import (
    V3_bug_fixed_non_metadata,
    V3_bug_fixed_options,
)
from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.subsystems.mass.gasp_based.mass_premission import MassPremission
from aviary.utils.aviary_values import get_items
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults, is_option
from aviary.variable_info.variables import Aircraft, Mission


class MassSummationTestCase1(unittest.TestCase):
    """
    This is the large single aisle 1 V3 bug fixed test case.
    All values are from V3 bug fixed output (or hand calculated from output) unless
    otherwise specified.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'gasp_based_geom',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'total_mass',
            MassPremission(),
            promotes=['*'],
        )

        for key, (val, units) in get_items(V3_bug_fixed_options):
            if not is_option(key):
                self.prob.model.set_input_defaults(key, val=val, units=units)

        for key, (val, units) in get_items(V3_bug_fixed_non_metadata):
            self.prob.model.set_input_defaults(key, val=val, units=units)

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        # self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, V3_bug_fixed_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        # print(f'wetted_area: {self.prob[Aircraft.Fuselage.WETTED_AREA]}')

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['gasp_based_geom.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['gasp_based_geom.cabin_len'], 72.09722222222223, tol)
        assert_near_equal(self.prob['gasp_based_geom.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.63, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.54, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact GASP value from the output file, likely due to rounding error

        assert_near_equal(self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.6509873673743, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.96457870166355, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol)
        assert_near_equal(self.prob['loc_MAC_vtail'], 0.44959578484694906, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 15758, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # fuel values:
        # modified from GASP value to account for updated crew mass. GASP value is
        # 78843.6
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 79500.16001078, tol)
        # modified from GASP value to account for updated crew mass. GASP value is
        # 102408.05695930264
        assert_near_equal(self.prob['fus_mass_full'], 101735.01012115, tol)
        # modified from GASP value to account for updated crew mass. GASP value is
        # 1757
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1783.50656044, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1757
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50266.438, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18624.42144949, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18814

        # modified from GASP value to account for updated crew mass. GASP value is
        # 42843.6
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 43500.16001078, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42843.6
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16161.21, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16127
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 43500.16001078, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42844.0
        assert_near_equal(
            self.prob['fuel_mass_min'], 33460.16001078, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 32803.6
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 869.61632311, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 856.4910800459031
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1589.29615013, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1576.1710061411081
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 95899.83998922, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96556.0
        # extra_fuel_mass calculated differently in this version, so test for payload_mass_max_fuel not included
        assert_near_equal(self.prob['volume_wingfuel_mass'], 57066.3, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 57066.3, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(self.prob['extra_fuel_volume'], 0, tol)  # always zero when no body tank
        assert_near_equal(self.prob['max_extra_fuel_mass'], 0, tol)  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


class MassSummationTestCase2(unittest.TestCase):
    """
    This is the large single aisle 1 V3.5 test case.
    All values are from V3.5 output (or hand calculated from the output, and these cases
    are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact GASP value from the output file, likely due to rounding error

        # note: this is not the value in the GASP output, because the output calculates
        # them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.578314120156815, tol)
        # note: this is not the value in the GASP output, because the output calculates
        # them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.828924591320984, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol
        )  # calculated by hand

        # note: tail.loc_MAC_vtail not included in v3.5

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 15653, tol)

        # fuel values:
        # modified from GASP value to account for updated crew mass. GASP value is
        # 79147.2
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 79656.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 79147.2

        # calculated by hand,  #modified from GASP value to account for updated crew
        # mass. GASP value is 102321.45695930265
        assert_near_equal(self.prob['fus_mass_full'], 101684.81858046, tol)
        # modified from GASP value to account for updated crew mass. GASP value is
        # 1769
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1789.92707671, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1769

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50132.58, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18621.15351242, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18787

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 43656.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147.2
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16167.631, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16140
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 43656.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147
        assert_near_equal(
            self.prob['fuel_mass_min'], 33616.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 33107.2
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 872.74688953, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 862.5603807559726
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1592.42671655, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 1582.2403068511774
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 95743.24203151, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96253.0
        # extra_fuel_mass calculated differently in this version, so payload_mass_max_fuel test not included
        assert_near_equal(self.prob['volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(self.prob['extra_fuel_volume'], 0, tol)  # always zero when no body tank
        assert_near_equal(self.prob['max_extra_fuel_mass'], 0, tol)  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-10, rtol=1e-12)


class MassSummationTestCase3(unittest.TestCase):
    """
    This is thelarge single aisle 1V3.6 test case with a fuel margin of 0%, a wing loading of 128 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact value, likely due to rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.578314120156815, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.828924591320984, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 15653, tol)

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 79656.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 79147.2

        assert_near_equal(
            self.prob['fus_mass_full'], 101684.818580466, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 102321.45695930265
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1789.92707671, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 102321.45695930265 (is it correct?)

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50132.58, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18621.15351242, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18787

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 43656.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147.2
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16167.631, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16140
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 43656.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147
        assert_near_equal(
            self.prob['fuel_mass_min'], 33616.75796849, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 33107.2
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 872.74688953, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 862.6
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1592.42671655, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1582.2
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 95743.24203151, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96253.0
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 36000, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(self.prob['volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(self.prob['extra_fuel_volume'], 0, tol)  # always zero when no body tank
        assert_near_equal(self.prob['max_extra_fuel_mass'], 0, tol)  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-10, rtol=1e-12)


class MassSummationTestCase4(unittest.TestCase):
    """
    This is the large single aisle 1V3.6 test case with a fuel margin of 10%, a wing loading of 128 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # slightly different from GASP value, likely numerical error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.578314120156815, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.828924591320984, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 15653, tol)

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 79474.11569854, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 78966.7

        assert_near_equal(
            self.prob['fus_mass_full'], 101867.46, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 102501.95695930265
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1960.68, tol)
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 79474.11569854, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 78966.7

        assert_near_equal(
            self.prob['fus_mass_full'], 101867.46085041, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 102501.95695930265
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1960.682618, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1938

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50144.527, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18633.04024108, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18799

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 43474.11569854, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42966.7
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16339.047, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16309
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 43474.11569854, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42967
        assert_near_equal(
            self.prob['fuel_mass_min'], 33434.11569854, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 32926.7
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 956.00523534, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 944.8
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1588.77549551, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1578.6
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 95925.88430146, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96433.0
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 36000, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(self.prob['volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(self.prob['extra_fuel_volume'], 0, tol)  # always zero when no body tank
        assert_near_equal(self.prob['max_extra_fuel_mass'], 0, tol)  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-10, rtol=1e-12)


class MassSummationTestCase5(unittest.TestCase):
    """
    This is thelarge single aisle 1V3.6 test case with a fuel margin of 0%, a wing loading of 150 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=150, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0.0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.16, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 15.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1394, tol
        )  # slightly different from GASP value, likely rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 8.848695928254141, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 15.550266681026597, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            6384.349999999999,
            tol,
            # self.prob['main_gear_mass'], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 14631, tol)

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 81085.9308234, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 81424.8

        assert_near_equal(self.prob['fus_mass_full'], 102510.642, tol)  # calculated by hand
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1848.52316376, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1862

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 48940.74, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18674.791, tol)

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 45085.9308234, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 45424.8
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16225.793, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16233
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 45085.9308234, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 45425
        assert_near_equal(
            self.prob['fuel_mass_min'], 35045.9308234, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 35384.8
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 901.317636, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 908.1
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1620.99746302, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1627.8
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 94314.0691766, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 93975
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 34766.20684105, tol
        )  # note: value came from running the GASP code on my own and printing it out,  #modified from GASP value to account for updated crew mass. GASP value is 34427.4
        assert_near_equal(self.prob['volume_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 1233.79315895, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1572.6
        assert_near_equal(
            self.prob['extra_fuel_volume'], 24.6648902, tol
        )  # slightly different from GASP value, likely a rounding error,  #modified from GASP value to account for updated crew mass. GASP value is 31.43
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 1233.79315895, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1572.6

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


class MassSummationTestCase6(unittest.TestCase):
    """
    This is thelarge single aisle 1V3.6 test case with a fuel margin of 10%, a wing loading of 150 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=150, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.16, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 15.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1394, tol
        )  # note: not exact GASP value, likely rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 8.848695928254141, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 15.550266681026597, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            6384.349999999999,
            tol,
            # self.prob['main_gear_mass'], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 14631, tol)

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 80636.1673241, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 80982.7

        assert_near_equal(self.prob['fus_mass_full'], 106989.952, tol)  # calculated by hand
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 2013.09114511, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 2029

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 49209.648, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18960.975, tol)

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 44636.1673241, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 44982.7
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16390.94, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16399
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 44636.16732294, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 44982.7
        assert_near_equal(
            self.prob['fuel_mass_min'], 34596.1673241, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 34942.7
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 981.55900268, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 989.2
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1612.00619309, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1618.9
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 94763.8326759, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 94417
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 35215.97039402, tol
        )  # note: value came from running the GASP code on my own and printing it out,  #modified from GASP value to account for updated crew mass. GASP value is 34879.2
        assert_near_equal(self.prob['volume_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 784.02965965, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1120.9
        assert_near_equal(
            self.prob['extra_fuel_volume'], 104.90625688, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 112.3
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 5247.64639206, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 5618.2

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


class MassSummationTestCase7(unittest.TestCase):
    """
    This is the Advanced Tube and Wing V3.6 test case.
    All values are from V3.6 output, hand calculated from the output, or were printed out after running the code manually.
    Values not directly from the output are labeled as such.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37100, units='ft')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.165, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=11, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=145388.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.WING_LOADING, val=104.50, units='lbf/ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=17000.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.475, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.09986, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.1621, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 7.36, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28620.0, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.594, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2095, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.715, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118, units='ft')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=15970.0, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.2355, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=1014.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.085, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.105, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1504.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=126.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=9114.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=0.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=10.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, val=85, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 61.6, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.91, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.01, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1132, tol
        )  # slightly different from GASP value, likely a rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD],
            9.6498,
            tol,
            # note: value came from running the GASP code on my own and printing it out (GASP output calculates this differently)
        )
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD],
            13.4662,
            tol,
            # note: value came from running the GASP code on my own and printing it out (GASP output calculates this differently)
        )
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 11.77, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            5219.3076,
            tol,
            # self.prob['main_gear_mass'], 5219.3076, tol
        )  # note: value came from running the GASP code on my own and printing it out

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 8007, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1321 / 2, tol)

        # wing values:
        assert_near_equal(
            self.prob['isolated_wing_mass'], 13993, tol
        )  # calculated as difference between wing mass and fold mass, not an actual GASP variable

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 63122.20489199, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 62427.2

        assert_near_equal(
            self.prob['fus_mass_full'], 99380.387, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1457.73144048, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1426

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 45370.902, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18858.356, tol)

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 32322.20489199, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 31627.2
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 10785.88644048, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 10755.0
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 32322.20489185, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 31627.0
        assert_near_equal(
            self.prob['fuel_mass_min'], 16352.20489199, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 15657.2
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 710.77229745, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 695.5
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1261.88270827, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1248.0
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 82265.79510801, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 82961.0
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 30800.0039, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(self.prob['volume_wingfuel_mass'], 33892.8, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 33892.8, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        assert_near_equal(
            self.prob['extra_fuel_volume'], 33.21832644, tol
        )  # note: higher tol because slightly different from GASP value, likely numerical issues,  #modified from GASP value to account for updated crew mass. GASP value is 17.9
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 1661.65523441, tol
        )  # note: higher tol because slightly different from GASP value, likely numerical issues,  #modified from GASP value to account for updated crew mass. GASP value is 897.2
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX], 677.554, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 677.554, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


class MassSummationTestCase8(unittest.TestCase):
    """
    This is the Trans-sonic Truss-Braced Wing V3.6 test case
    All values are from V3.6 output, hand calculated from the output, or were printed out after running the code manually.
    Values not directly from the output are labeled as such.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=43000, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 44.2, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.163, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.13067, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.025, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 3.0496, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 7.642, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28620.0, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.35255, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2095, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.660, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 6.85, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.18, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.WING_LOADING, val=87.5, units='lbf/ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=1014.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.085, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.105, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1504.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=126.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=9114.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=0.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=10.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=21160.0, units='lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.73934, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.5625, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=22.47, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=19.565, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Strut.ATTACHMENT_LOCATION, val=118, units='ft')
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=15970.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.2470, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=2.5, units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.2143, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=0.825, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0, units='deg'
        )  # not in file
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.2076, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.2587, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.11, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.5936, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=30, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # not in file
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # not in file
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.03390, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.060, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=89.66, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.060, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=89.66, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')  # not in file
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=143100.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=78.94, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.346, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.43, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.066, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )
        # self.prob.model.set_input_defaults(
        #     Aircraft.Strut.AREA, 523.337, units='ft**2'
        # )  # had to calculate by hand
        self.prob.model.set_input_defaults(Aircraft.Strut.MASS_COEFFICIENT, 0.238, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 93.9, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 13.59, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 13.15, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1068, tol
        )  # note:precision came from running code on my own and printing it out

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.381, tol
        )  # note, printed out manually because calculated differently in output subroutine
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 20.056, tol
        )  # note, printed out manually because calculated differently in output subroutine
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 13.19, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            4123.4,
            tol,
            # self.prob['main_gear_mass'], 4123.4, tol
        )  # note:printed out from GASP code

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 10453.0, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1704.0 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 14040, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 18031, tol)

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 59780.52528506, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 59372.3

        assert_near_equal(
            self.prob['fus_mass_full'], 97651.376, tol
        )  # note:printed out from GASP code
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1912.71466876, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1886.0

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 43655.977, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 14654.517, tol)

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 28980.52528506, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 28572.3
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 14069.60018876, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 14043.0
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 28980.52528501, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 28572.0
        assert_near_equal(
            self.prob['fuel_mass_min'], 13010.52528506, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 12602.3
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 637.28803796, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 628.3
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1195.07883601, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1186.9
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 83319.47471494, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 83728.0
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 30800.0, tol
        )  # note:printed out from GASP code
        assert_near_equal(self.prob['volume_wingfuel_mass'], 31051.6, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 31051.6, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        # TODO: extra_fuel_volume < 0. need investigate
        assert_near_equal(
            self.prob['extra_fuel_volume'], 16.53287329, tol
        )  # note: printed out from the GASP code,  #modified from GASP value to account for updated crew mass. GASP value is 7.5568
        # TODO: extra_fuel_volume < 0. need investigate
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 827.01142371, tol
        )  # note: printed out from the GASP code,  #modified from GASP value to account for updated crew mass. GASP value is 378.0062

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


class MassSummationTestCase9(unittest.TestCase):
    """
    This is the electrified Trans-sonic Truss-Braced Wing V3.6 test case
    All values are from V3.6 output, hand calculated from the output, or were printed out after running the code manually.
    Values not directly from the output are labeled as such.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=43000, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 44.2, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.163, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.13067, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.025, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 3.0496, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 8.425, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28620, units='lbf'
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.73934, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2095, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.569, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 6.85, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.18, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.WING_LOADING, val=96.10, units='lbf/ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=1014.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.085, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.105, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1504.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=126.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=9114.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=0.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=10.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=23750.0, units='lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.82984, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.5936, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=22.47, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=19.565, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=118.0, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=15970.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.2744, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=2.5, units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.2143, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=0.825, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.2076, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.2587, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.11, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.5936, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=30.0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=1, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.03390, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.060, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=96.94, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=96.94, units='unitless'
        )
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=166100.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=78.94, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.346, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.43, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.066, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )
        # self.prob.model.set_input_defaults(
        #     Aircraft.Strut.AREA, 553.1, units='ft**2'
        # )
        self.prob.model.set_input_defaults(Aircraft.Strut.MASS_COEFFICIENT, 0.238, units='unitless')
        self.prob.model.set_input_defaults('motor_power', 830, units='kW')
        self.prob.model.set_input_defaults('motor_voltage', 850, units='V')
        self.prob.model.set_input_defaults('max_amp_per_wire', 260, units='A')
        self.prob.model.set_input_defaults(
            'safety_factor', 1, units='unitless'
        )  # (not in this GASP code)
        self.prob.model.set_input_defaults('wire_area', 0.0015, units='ft**2')
        self.prob.model.set_input_defaults('rho_wire', 565, units='lbm/ft**3')
        self.prob.model.set_input_defaults('battery_energy', 6077, units='MJ')
        self.prob.model.set_input_defaults('motor_eff', 0.98, units='unitless')
        self.prob.model.set_input_defaults('inverter_eff', 0.99, units='unitless')
        self.prob.model.set_input_defaults('transmission_eff', 0.975, units='unitless')
        self.prob.model.set_input_defaults('battery_eff', 0.975, units='unitless')
        self.prob.model.set_input_defaults('rho_battery', 0.5, units='kW*h/kg')
        self.prob.model.set_input_defaults('motor_spec_mass', 4, units='hp/lbm')
        self.prob.model.set_input_defaults('inverter_spec_mass', 12, units='kW/kg')
        self.prob.model.set_input_defaults('TMS_spec_mass', 0.125, units='lbm/kW')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.cabin_len'], 93.9, tol)
        assert_near_equal(self.prob['size.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 13.97, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 13.53, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1068, tol
        )  # (printed out from GASP code to get better precision)

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.644, tol
        )  # (printed out from GASP code)
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 20.618, tol
        )  # (printed out from GASP code)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.56, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 4786.2, tol
        )  # (printed out from GASP code)

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 13034.0, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 2124.5 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['isolated_wing_mass'], 15895, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 20461.7, tol)

        # fuel values:
        assert_near_equal(
            self.prob['OEM_wingfuel_mass'], 63921.50874092, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 63707.6

        assert_near_equal(
            self.prob['fus_mass_full'], 109537.46058162, tol
        )  # (printed out from GASP code),  #modified from GASP value to account for updated crew mass. GASP value is 108754.4
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1987.29052446, 0.00055
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 1974.5

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 49609.317, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 16350.28996446, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16436.0

        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 33121.50874092, 0.00058
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 32907.6
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 26534.716, 0.00054
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 26527.0
        assert_near_equal(
            self.prob[Mission.Summary.FUEL_MASS], 33121.50874092, 0.00056
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 32908
        assert_near_equal(
            self.prob['fuel_mass_min'], 17151.50874092, 0.0012
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 16937.6
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 662.13560226, 0.00051
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 657.9
        assert_near_equal(
            self.prob['OEM_fuel_vol'], 1277.86167649, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1273.6
        assert_near_equal(
            self.prob[Mission.Summary.OPERATING_MASS], 102178.49125908, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 102392.0
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 30800.0, tol
        )  # (printed out from GASP code)
        assert_near_equal(self.prob['volume_wingfuel_mass'], 35042.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 35042.1, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        assert_near_equal(self.prob['extra_fuel_volume'], 0.69314718, tol)
        assert_near_equal(self.prob['max_extra_fuel_mass'], 34.67277748, tol)

        assert_near_equal(self.prob[Aircraft.Electrical.HYBRID_CABLE_LENGTH], 65.6, tol)
        assert_near_equal(
            self.prob['aug_mass'], 9394.3, 0.0017
        )  # slightly above tol, due to non-integer number of wires

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


@use_tempdirs
class BWBMassSummationTestCase(unittest.TestCase):
    """
    GASP BWB model
    """

    def setUp(self):
        options = get_option_defaults()
        # options from SizeGroup
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, val=False, units='unitless')
        options.set_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=False, units='unitless')
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=True, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 22, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 3)
        options.set_val(Aircraft.CrewPayload.Design.NUM_SEATS_ABREAST_TOURIST, 18)
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_FIRST, 36, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.SEAT_PITCH_TOURIST, 32, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 21, units='inch')
        options.set_val(Aircraft.CrewPayload.Design.NUM_FIRST_CLASS, 11)
        # options from MassPremission
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=41000, units='ft')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=225, units='lbm')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.04373, units='unitless')
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 2, units='unitless')
        options.set_val(Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER, 0.0667, units='lbm')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        # inputs from SizeGroup
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, 70.0, units='lbf/ft**2')
        # prob.model.set_input_defaults(Aircraft.VerticalTail.ASPECT_RATIO, 1.705, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.TAPER_RATIO, 0.366, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.TAPER_RATIO, 0.366, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.45, units='unitless')

        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, 0.000001, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.015, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 5.0, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 7.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 0.6, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.75, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, 0.5463, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.MOMENT_RATIO, 5.2615, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.ASPECT_RATIO, 1.705, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2205, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.3588, units='unitless')
        # prob.model.set_input_defaults(
        #    Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0.0, units='unitless'
        # )
        prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.HEIGHT_TO_WIDTH_RATIO, 0.2597, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, 65.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURIZED_WIDTH_ADDITIONAL, 0.0, units='ft'
        )
        prob.model.set_input_defaults(
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE, 0.0, units='unitless'
        )
        # inputs from MassPremission
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ROOT_CHORD, val=0.03836448, units='ft'
        )
        # prob.model.set_input_defaults(
        #    Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, 0.107, units='unitless'
        # )
        # prob.model.set_input_defaults(
        #    Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, 0.135, units='unitless'
        # )
        # prob.model.set_input_defaults(Aircraft.Avionics.MASS, 1504.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, 1.155, units='unitless'
        )
        # prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, 236.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Furnishings.MASS, 9114.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, 6.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, 100.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, 5.0, units='lbm'
        )
        # prob.model.set_input_defaults(
        #    Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, 12.0, units='unitless'
        # )
        prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, 0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.CrewPayload.Design.CARGO_MASS, 0.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, 15000.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, 2.5, units='lbm/ft**2')
        prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, 1.25, units='unitless'
        )  # PYL default and the GASP manual don't agree.  Think(?) the default should be 0.7,and not 0.
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.ASPECT_RATIO, 1.705, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, 0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, 0.124, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, 0.119, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, 2.13421583, units='unitless'
        )  # 2.13421583 is for landing. In GASP, CLMAX is computed for different phases. For takeoff, 1.94302452
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, 0.5, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, 16.5, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, 0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, 0.0520, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, 0.85, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')

        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, 10.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_COEFFICIENT, 0.889, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.STRUCTURAL_MASS_INCREMENT, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, 75.78, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FOLD_MASS_COEFFICIENT, 0.15, units='unitless')
        prob.model.set_input_defaults(Mission.Design.MACH, 0.8, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, 0.0001, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, 0.2, units='unitless')
        prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, 0.827296853, units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.MAX_STRUCTURAL_SPEED, 402.5, units='mi/h')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_SPAN_RATIO, 0.61, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.CLEARANCE_RATIO, 0.2, units='unitless')

        prob.model.set_input_defaults(Aircraft.APU.MASS, 710.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, 0.116, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, 0.107, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, 0.135, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Avionics.MASS, 3225.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, 236.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, 3.0, units='lbm'
        )
        # prob.model.set_input_defaults(
        #    Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, 12.0, units='unitless'
        # )
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        # prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.001706279, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.119629, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, 10.0, units='psi')
        prob.model.set_input_defaults(Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.3650932, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, 11.45, units='lbm'
        )

        # inputs to UsefulLoadMass
        prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, 12.0, units='unitless'
        )
        # prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, [19580.1602], units='lbf')
        prob.model.set_input_defaults(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.7734, units='unitless')

        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_COEFFICIENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL, 0.2, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA, 5.0, units='lbm/ft**2'
        )

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Testing GASP data case:
        Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS -- WPES = 2055
        Aircraft.LandingGear.TOTAL_MASS -- WLG = 7800
        Aircraft.LandingGear.MAIN_GEAR_MASS -- WMG = 6630
        Aircraft.Wing.MATERIAL_FACTOR -- SKNO = 1.19461238
        half_sweep -- SWC2 = 0.479839474
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS -- WPL = 33750
        payload_mass_des -- WPLDES = 33750
        payload_mass_max -- WPLMAX = 48750
        Aircraft.HorizontalTail.MASS -- WHT = 1
        Aircraft.VerticalTail.MASS -- WVT = 864
        Aircraft.Wing.HIGH_LIFT_MASS -- WHLDEV = 974.0
        Aircraft.Controls.TOTAL_MASS -- WFC = 2115
        Aircraft.Propulsion.TOTAL_ENGINE_MASS -- WEP = 7005.
        Aircraft.Nacelle.MASS -- WNAC = 514.9
        Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS -- WPES = 2153
        Aircraft.Engine.POSITION_FACTOR -- SKEPOS = 1.05
        Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS -- WPL = 33750
        Aircraft.Design.FIXED_EQUIPMENT_MASS - WFE = 20876.
        Aircraft.Design.FIXED_USEFUL_LOAD -- WFUL = 5775.
        Aircraft.Wing.MASS -- WW = 7645.
        Aircraft.Strut.MASS -- WSTRUT = 0
        Aircraft.Wing.FOLD_MASS -- WWFOLD = 107.9
        Aircraft.Fuel.FUEL_SYSTEM_MASS -- WFSS = 1280.8
        fus_mass_full -- WX = 142354.9
        Aircraft.Fuselage.MASS -- WB = 27160
        Aircraft.Fuel.WING_VOLUME_DESIGN -- FVOLREQ = 731.6
        Mission.Summary.OPERATING_MASS -- OWE = 82982.
        fus_mass_full -- WX = 121864
        OEM_wingfuel_mass -- WFWOWE(WFW_MAX) = 67018.2
        OEM_fuel_vol -- FVOLW_MAX = 1339.8
        payload_mass_max_fuel -- WPLMXF = 30423.2
        max_wingfuel_mass -- WFWMX = 30309.0
        Aircraft.Design.STRUCTURE_MASS -- WST = 45623.
        Mission.Summary.FUEL_MASS -- WFADES = 33268.2
        Aircraft.Propulsion.MASS -- WP = 8592.
        Mission.Summary.FUEL_MASS_REQUIRED -- WFAREQ = 36595.0
        fuel_mass_min -- WFAMIN = 18268.2
        fuel_mass.wingfuel_mass_min -- WFWMIN = 11982.2
        Aircraft.Fuel.TOTAL_CAPACITY -- WFAMAX = 33268.2
        """
        prob = self.prob
        prob.run_model()

        tol = 1e-5

        # outputs from SizeGroup
        assert_near_equal(prob[Aircraft.Fuselage.AVG_DIAMETER], 38, tol)

        assert_near_equal(prob[Aircraft.Fuselage.LENGTH], 71.5245514, tol)
        assert_near_equal(prob[Aircraft.Fuselage.WETTED_AREA], 4573.42578, tol)
        assert_near_equal(prob[Aircraft.TailBoom.LENGTH], 71.5245514, tol)

        assert_near_equal(prob[Aircraft.Wing.AREA], 2142.85714286, tol)
        assert_near_equal(prob[Aircraft.Wing.SPAN], 146.38501094, tol)

        assert_near_equal(prob[Aircraft.Wing.CENTER_CHORD], 22.97244452, tol)
        assert_near_equal(prob[Aircraft.Wing.AVERAGE_CHORD], 16.2200522, tol)
        assert_near_equal(prob[Aircraft.Wing.ROOT_CHORD], 20.33371617, tol)
        assert_near_equal(prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.13596576, tol)

        assert_near_equal(prob[Aircraft.HorizontalTail.AREA], 0.00117064, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.SPAN], 0.04467601, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.ROOT_CHORD], 0.03836448, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 0.02808445, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MOMENT_ARM], 29.69074172, tol)

        assert_near_equal(prob[Aircraft.VerticalTail.AREA], 169.11964286, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.SPAN], 16.98084188, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.ROOT_CHORD], 14.58190052, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.AVERAGE_CHORD], 10.67457744, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MOMENT_ARM], 27.82191598, tol)

        assert_near_equal(prob[Aircraft.Nacelle.AVG_DIAMETER], 5.33382144, tol)
        assert_near_equal(prob[Aircraft.Nacelle.AVG_LENGTH], 7.24759657, tol)
        assert_near_equal(prob[Aircraft.Nacelle.SURFACE_AREA], 121.44575974, tol)

        # outputs from MassPremission
        assert_near_equal(prob[Aircraft.LandingGear.TOTAL_MASS], 7800.0, tol)
        assert_near_equal(prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6630.0, tol)
        assert_near_equal(prob[Aircraft.Wing.MATERIAL_FACTOR], 1.19461189, tol)
        assert_near_equal(prob['c_strut_braced'], 1, tol)
        assert_near_equal(prob['c_gear_loc'], 0.95, tol)
        assert_near_equal(prob[Aircraft.Engine.POSITION_FACTOR], 1.05, tol)
        assert_near_equal(prob['half_sweep'], 0.47984874, tol)
        assert_near_equal(prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(prob['payload_mass_des'], 33750.0, tol)
        assert_near_equal(prob['payload_mass_max'], 48750, tol)
        assert_near_equal(prob[Aircraft.HorizontalTail.MASS], 1.02401953, tol)
        assert_near_equal(prob[Aircraft.VerticalTail.MASS], 864.17404177, tol)
        assert_near_equal(prob[Aircraft.Wing.HIGH_LIFT_MASS], 971.82476285, tol)
        assert_near_equal(prob[Aircraft.Controls.TOTAL_MASS], 2114.98159054, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 7005.15475443, tol)
        assert_near_equal(prob[Aircraft.Nacelle.MASS], 303.61439936, tol)
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 1686.626, tol)
        assert_near_equal(prob[Aircraft.Engine.ADDITIONAL_MASS], 153.16770871, tol)
        assert_near_equal(prob[Aircraft.Engine.POSITION_FACTOR], 1.05, tol)
        assert_near_equal(prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 20876.477, tol)
        assert_near_equal(prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5971.79463002, tol)
        assert_near_equal(prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 1986.25111783, tol)

        # BWBWingMassGroup
        assert_near_equal(prob[Aircraft.Wing.MASS], 6959.72619224, tol)
        assert_near_equal(prob[Aircraft.Strut.MASS], 0, tol)
        assert_near_equal(prob[Aircraft.Wing.FOLD_MASS], 107.83351322, tol)

        # FuelMassGroup
        # FuelSysAndFullFuselageMass
        assert_near_equal(prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 1686.62563123, tol)
        assert_near_equal(prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1316.24565033, tol)
        assert_near_equal(prob['fus_mass_full'], 131150.22491506, tol)
        # BWBFuselageMass
        assert_near_equal(prob[Aircraft.Fuselage.MASS], 27159.28655493, tol)
        # StructMass
        assert_near_equal(prob[Aircraft.Design.STRUCTURE_MASS], 44470.83642382, tol)
        # FuelMass
        assert_near_equal(prob[Mission.Summary.FUEL_MASS], 34188.19870988, tol)
        assert_near_equal(prob[Aircraft.Propulsion.MASS], 8627.73582218, tol)
        assert_near_equal(prob[Mission.Summary.FUEL_MASS_REQUIRED], 34188.19870988, tol)
        assert_near_equal(prob['fuel_mass_min'], 19188.19870988, tol)
        # FuelAndOEMOutputs
        assert_near_equal(prob['OEM_wingfuel_mass'], 67938.19870988, tol)
        assert_near_equal(prob['OEM_fuel_vol'], 1358.15975265, tol)
        assert_near_equal(prob['payload_mass_max_fuel'], 29870.67005382, tol)
        assert_near_equal(prob['volume_wingfuel_mass'], 30308.8688, tol)
        assert_near_equal(prob['max_wingfuel_mass'], 30308.86876369, tol)
        assert_near_equal(prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 751.80590631, tol)
        assert_near_equal(prob[Mission.Summary.OPERATING_MASS], 82061.80129012, tol)
        # BodyTankCalculations
        assert_near_equal(prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 3879.32994618, tol)
        assert_near_equal(prob[Aircraft.Fuel.TOTAL_CAPACITY], 38067.52865606, tol)
        assert_near_equal(prob['extra_fuel_volume'], 145.89808883, tol)
        assert_near_equal(prob['max_extra_fuel_mass'], 7298.14981717, tol)
        assert_near_equal(prob['wingfuel_mass_min'], 11890.0488927, tol)


if __name__ == '__main__':
    unittest.main()
