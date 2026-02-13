import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.equipment_and_useful_load import (
    BWBEquipMassGroup,
    EquipMassSum,
    EquipMassGroup,
    UsefulLoadMass,
    UsefulLoadMassGroup,
    EquipAndUsefulLoadMassGroup,
)

from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission, Settings


class FixedEquipMassTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        # options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        # options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'equip',
            EquipMassSum(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS, val=1324.0561, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=683.46852785, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=13266.56, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Instruments.MASS, val=862.45194435, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Hydraulics.MASS, val=1487.78, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Electrical.MASS, val=2231.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.OxygenSystem.MASS, val=50, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, val=0.0, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 22792.3165722, tol)

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

        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
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
        self.prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=1068.96, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Furnishings.MASS_SCALER, val=40.0, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1324.05614369, tol)
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 13266.53964608, tol)
        assert_near_equal(self.prob[Aircraft.AntiIcing.MASS], 683.4685279, tol)
        assert_near_equal(self.prob[Aircraft.APU.MASS], 1077.969377, tol)
        assert_near_equal(self.prob[Aircraft.Avionics.MASS], 1514.0, tol)
        assert_near_equal(self.prob[Aircraft.Electrical.MASS], 170.0, tol)
        assert_near_equal(self.prob[Aircraft.Hydraulics.MASS], 1487.78, tol)
        assert_near_equal(self.prob[Aircraft.Instruments.MASS], 547.41157631, tol)
        assert_near_equal(self.prob[Aircraft.OxygenSystem.MASS], 50.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        # options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        # options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.FLIGHT_CREW_MASS, val=492.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, val=800.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, val=342.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, val=2872.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS, val=619.76, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS, val=165, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5341.36, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class UsefulMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Settings.VERBOSITY, val=0, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'useful_group',
            UsefulLoadMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1370.3, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 165.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS], 800.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.FLIGHT_CREW_MASS], 492.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS], 2872.0, tol)
        assert_near_equal(self.prob[Aircraft.Design.EMERGENCY_EQUIPMENT_MASS], 115.0, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS], 342.6, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.UNUSABLE_FUEL_MASS], 619.82896954, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipAndUsefulMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            EquipAndUsefulLoadMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS, val=1324.0561, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS, val=165, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.FLIGHT_CREW_MASS, val=492.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, val=800.0, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Electrical.MASS, val=3050.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, val=342.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS, val=619.76, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=13266.56, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Hydraulics.MASS, val=1487.78, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Instruments.MASS, val=547.508601, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.OxygenSystem.MASS, val=50.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, val=2872.0, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5341.429, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23163.787, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


@use_tempdirs
class BWBFixedEquipMassGroupTest(unittest.TestCase):
    """Created based on GASP BWB model"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            BWBEquipMassGroup(),
            promotes=['*'],
        )
        prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, 0.116, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, 0.107, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, 0.135, units='unitless'
        )
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, 7800.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, 2115.19946, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.00170628, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.119629, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, 1.155, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, 10.0, units='psi')
        prob.model.set_input_defaults(Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.3650932, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        setup_model_options(self.prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1301.5729093, tol)
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 11269.87654648, tol)
        assert_near_equal(self.prob[Aircraft.AntiIcing.MASS], 706.48495598, tol)
        assert_near_equal(self.prob[Aircraft.APU.MASS], 928.46163145, tol)
        assert_near_equal(self.prob[Aircraft.Avionics.MASS], 1430.0, tol)
        assert_near_equal(self.prob[Aircraft.Electrical.MASS], 170.0, tol)
        assert_near_equal(self.prob[Aircraft.Hydraulics.MASS], 1328.90233952, tol)
        assert_near_equal(self.prob[Aircraft.Instruments.MASS], 581.94821634, tol)
        assert_near_equal(self.prob[Aircraft.OxygenSystem.MASS], 50.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class BWBUsefulMassTestCase1(unittest.TestCase):
    """
    Created based on GASP BWB modele
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Settings.VERBOSITY, 0)

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'useful',
            UsefulLoadMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, 6.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, 3, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, 100.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, 5.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, 12.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, [19580.1602], units='lbf')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.45, units='unitless')

        setup_model_options(self.prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 4321.79463506, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


@use_tempdirs
class BWBFixedEquipAndUsefulMassGroupTest(unittest.TestCase):
    """this is the large single aisle 1 V3 test case"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            EquipAndUsefulLoadMassGroup(),
            promotes=['*'],
        )

        # inputs to BWBEquipMassGroup
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
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, 6.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, 3.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, 100.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, 5.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, 12.0, units='unitless'
        )
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, 7800.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, 2115.19946, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.001706279, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.119629, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, 1.155, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, 10.0, units='psi')
        prob.model.set_input_defaults(Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.3650932, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Furnishings.MASS_SCALER, 40.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER, 11.45, units='lbm'
        )

        # inputs to UsefulLoadMass
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, 6.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, 3.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, 100.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, 5.0, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, 12.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, [19580.1602], units='lbf')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.45, units='unitless')

        setup_model_options(self.prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.AirConditioning.MASS], 1301.573, tol)
        assert_near_equal(self.prob[Aircraft.Furnishings.MASS], 11269.877, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 20876.477, tol)
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 4321.79463506, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


if __name__ == '__main__':
    # unittest.main()
    test = BWBFixedEquipMassGroupTest()  # FixedEquipMassGroupTest()
    test.setUp()
    test.test_case1()
