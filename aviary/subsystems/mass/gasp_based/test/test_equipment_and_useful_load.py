import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.equipment_and_useful_load import (
    ACMass,
    EquipMassPartial,
    FurnishingMass,
    EquipMassSum,
    EquipMassGroup,
    UsefulLoadMass,
    EquipAndUsefulLoadMassGroup,
)
from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class FixedEquipMassTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['equip_mass_part'], 8573.1915, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=5, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        # not actual GASP value
        assert_near_equal(self.prob['equip_mass_part'], 10992.963, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.TYPE, val=[GASPEngineType.RECIP_CARB], units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        # not actual GASP value
        assert_near_equal(self.prob['equip_mass_part'], 10992.963, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase4(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        prob = om.Problem()
        prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase5smooth(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7

        assert_near_equal(self.prob['equip_mass_part'], 8573.192, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase6smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=5, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        # not actual GASP value
        assert_near_equal(self.prob['equip_mass_part'], 10992.963, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase7smooth(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=5, units='unitless')
        options.set_val(Aircraft.Engine.TYPE, val=[GASPEngineType.RECIP_CARB], units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        # not actual GASP value
        assert_near_equal(self.prob['equip_mass_part'], 10992.963, tol)  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase8(unittest.TestCase):
    """
    this is the same case as EquipMassTestCase1, except:
    Aircraft.APU.MASS = 0.0,
    Aircraft.Avionics.MASS = 0.0,
    Aircraft.AntiIcing.MASS = 0.0,
    Aircraft.Furnishings.MASS = 0.0
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=0.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=0.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=0.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['equip_mass_part'], 8410.6295, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase9smooth(unittest.TestCase):
    """
    this is the same case as EquipMassTestCase5smooth, except:
    Aircraft.APU.MASS = 0.0,
    Aircraft.Avionics.MASS = 0.0,
    Aircraft.AntiIcing.MASS = 0.0,
    Aircraft.Furnishings.MASS = 0.0
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassPartial(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=0.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=0.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=0.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['equip_mass_part'], 8410.6295, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class ACMassTestCase1(unittest.TestCase):
    """
    Created based on EquipMassTestCase1
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'ac',
            ACMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1324.0561, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class ACMassTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'ac',
            ACMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class ACMassTestCase3(unittest.TestCase):
    """
    Created based on GASP BWB model where SWF is DHYDRAL
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'ac',
            ACMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.155, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=71.52455, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=10, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=19.365, units='ft')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1301.5666, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase1(unittest.TestCase):
    """Created based on EquipMassTestCase1"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 13266.56, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase2(unittest.TestCase):
    """Test mass-weight conversion"""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1069.0, units='ft**2')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase3(unittest.TestCase):
    """
    Created based on GASP BWB model where SWF is DHYDRAL
    """

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless'
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=19.365, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, val=1283.5249, units='ft**2'
        )

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        USE_EMPIRICAL_EQUATION = True
        SMOOTH_MASS_DISCONTINUITIES = False
        """
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 11269.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_case2(self):
        # case 2A
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=False, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 18839.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

        # case 2B
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=True, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        # assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 11269.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

        # case 2C
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        self.options.set_val(
            Aircraft.Furnishings.USE_EMPIRICAL_EQUATION, val=False, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 18839.863, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase4(unittest.TestCase):
    """
    Created based on GASP BWB model where SWF is DHYDRAL
    GROSS_MASS < 10000
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=9999.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=19.365, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, val=1283.5249, units='ft**2'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 590.935, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FurnishingMassTestCase5(unittest.TestCase):
    """
    Created based on GASP BWB model where SWF is DHYDRAL
    NUM_PASSENGERS < 50
    """

    def setUp(self):
        self.options = get_option_defaults()
        self.options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=49, units='unitless')
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=False, units='unitless'
        )

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'furnishing',
            FurnishingMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=19.365, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.CABIN_AREA, val=1283.5249, units='ft**2'
        )

        setup_model_options(self.prob, self.options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        SMOOTH_MASS_DISCONTINUITIES = False
        """
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 3348.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_case2(self):
        """
        SMOOTH_MASS_DISCONTINUITIES = True
        """
        self.options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless'
        )
        setup_model_options(self.prob, self.options)
        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.run_model()
        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 3348.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassSumTestCase1(unittest.TestCase):
    """
    Created based on EquipMassTestCase1
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'sum',
            EquipMassSum(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults('equip_mass_part', val=8573.19157631, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS, val=1324.0561, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 21089.248, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            EquipMassGroup(),
            promotes=['*'],
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
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1068.96, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, val=40.0, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23163.787, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

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
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0], units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5176.429, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 4932

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=5, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

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
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 1235.429, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.TYPE, val=[GASPEngineType.RECIP_CARB], units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

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
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 892.829, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassTestCase4(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.equipment_and_useful_load as equip

        equip.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')

        prob = om.Problem()
        prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0], units='lbf')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassTestCase5(unittest.TestCase):
    """
    this is the same case as UsefulMassTestCase1, except:
    Aircraft.Design.EMERGENCY_EQUIPMENT_MASS = 0.0
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

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
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0], units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5241.429, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 4932

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipAndUsefulMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            EquipAndUsefulLoadMassGroup(),
            promotes=['*'],
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

        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, val=469.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')

        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1068.96, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, val=40.0, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5176.429, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23163.787, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    # unittest.main()
    test = FurnishingMassTestCase1()
    test.setUp()
    test.test_case1()
