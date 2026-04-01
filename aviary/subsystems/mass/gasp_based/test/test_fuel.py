import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.fuel import (
    BodyTankCalculations,
    BWBFuselageMass,
    FuelComponents,
    FuelMass,
    FuelSysAndFullFuselageMass,
    FuselageMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission, Settings


class BodyCalculationTestCase1(unittest.TestCase):
    """this is the large single aisle 1 V3 test case."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=857.480639944284, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=1114.006551379108, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=32853, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=42892.0, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=55725.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=1114.0, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')

        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=42893.1, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=96508, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(
            self.prob['extra_fuel_volume'], 0.69314718, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 34.67277748, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(self.prob['wingfuel_mass_min'], 32818.32722252, tol)
        # note: Aircraft.Fuel.TOTAL_CAPACITY is calculated differently in V3, so it is not included here

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BodyCalculationTestCase2(
    unittest.TestCase
):  # this is v 3.6 large single aisle 1 test case with wing loading of 150 psf and fuel margin of 10%
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=989.2, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=876.7, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=34942.7, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=44982.7, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=43852.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=876.7, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=44973.0, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=94417, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 3e-4
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 1130.6, tol)
        assert_near_equal(self.prob['extra_fuel_volume'], 112.5, tol)
        assert_near_equal(self.prob['max_extra_fuel_mass'], 5628.9, tol)
        assert_near_equal(self.prob['wingfuel_mass_min'], 29313.9, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.TOTAL_CAPACITY], 46093.7, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BodyCalculationTestCase3(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=989.2, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=876.7, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=34942.7, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=44982.7, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=43852.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=876.7, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=44973.0, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=94417, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BodyCalculationTestCase4smooth(unittest.TestCase):
    """
    this is the large single aisle 1 V3 test case.
    It tests the case Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES = True.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])
        self.prob.model.wing_calcs.options[Settings.VERBOSITY] = Verbosity.QUIET
        self.prob.model.wing_calcs.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES] = True

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=857.480639944284, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=1114.006551379108, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=32853, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=42892.0, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=55725.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=1114.0, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=42893.1, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=96508, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(
            self.prob['extra_fuel_volume'], 0.69314718, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 34.67277748, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(self.prob['wingfuel_mass_min'], 32818.32722252, tol)
        # note: Aircraft.Fuel.TOTAL_CAPACITY is calculated differently in V3, so it is not included here

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BodyCalculationTestCase5(unittest.TestCase):
    """
    This is the Advanced Tube and Wing V3.6 test case.
    It tests the case Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES = True.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])
        self.prob.model.wing_calcs.options[Settings.VERBOSITY] = Verbosity.QUIET
        self.prob.model.wing_calcs.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES] = False

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=661.583, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=677.554, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=14115.342, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=30085.342, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=33892.8, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=677.554, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=145388, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=30085.342, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=84502.658, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(
            self.prob['extra_fuel_volume'], 0.0, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 0.0, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(self.prob['wingfuel_mass_min'], 14115.342, tol)
        # note: Aircraft.Fuel.TOTAL_CAPACITY is calculated differently in V3, so it is not included here

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BodyCalculationTestCase6smooth(unittest.TestCase):
    """
    This is the Advanced Tube and Wing V3.6 test case.
    It tests the case Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES = True.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])
        self.prob.model.wing_calcs.options[Settings.VERBOSITY] = Verbosity.QUIET
        self.prob.model.wing_calcs.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES] = True

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=661.583, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=677.554, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=14115.342, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=30085.342, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=33892.8, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=677.554, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=145388, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=30085.342, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=84502.658, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(
            self.prob['extra_fuel_volume'], 0.69314626, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 34.672731, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(self.prob['wingfuel_mass_min'], 14080.669, tol)
        # note: Aircraft.Fuel.TOTAL_CAPACITY is calculated differently in V3, so it is not included here

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BodyCalculationTestCase7smooth(unittest.TestCase):
    """It tests the case Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES = True."""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])
        self.prob.model.wing_calcs.options[Settings.VERBOSITY] = Verbosity.QUIET
        self.prob.model.wing_calcs.options[Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES] = True

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_DESIGN, val=615.03323815, units='ft**3'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, val=620.75516364, units='ft**3'
        )
        self.prob.model.set_input_defaults('fuel_mass_min', val=11998.49344063, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=27968.49344063, units='lbm')
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=31051.56633854, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=620.75516364, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=143100.0, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass', val=26236.86063849, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=84331.50655937, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # note: not in version 3 output, calulated by hand
        assert_near_equal(self.prob['extra_fuel_volume'], 0.68385622, tol)
        assert_near_equal(self.prob['max_extra_fuel_mass'], 34.20802285, tol)
        assert_near_equal(self.prob['wingfuel_mass_min'], 11964.285, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


# this is the large single aisle 1 V3 test case
class FuelComponentsTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', FuelComponents(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=96506, units='lbm')
        self.prob.model.set_input_defaults('fuel_mass_required', val=42892.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 1114, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuel.TOTAL_CAPACITY, val=55725.1, units='lbm')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 78894, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 857.480639944284, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1577.160566039489, tol)
        assert_near_equal(self.prob[Mission.OPERATING_MASS], 96508.0, tol)

        assert_near_equal(self.prob['payload_mass_max_fuel'], 23166.9, tol)
        assert_near_equal(self.prob['volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 55725.1, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class FuelAndOEMTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('wing_calcs', FuelComponents(), promotes=['*'])
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Mission.OPERATING_MASS, val=96506, units='lbm')
        prob.model.set_input_defaults('fuel_mass_required', val=42892.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 1114, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuel.TOTAL_CAPACITY, val=55725.1, units='lbm')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class FuelSysAndFullFusMassTestCase(
    unittest.TestCase
):  # this is the large single aisle 1 V3 test case
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('sys_and_fus', FuelSysAndFullFuselageMass(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830.0, units='lbm')
        self.prob.model.set_input_defaults('wing_mounted_mass', val=24446.343040697346, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults('fuel_mass', val=42893, units='lbm')
        self.prob.model.set_input_defaults('wingfuel_mass_min', val=32853, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob['fus_mass_full'], 102270, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1759, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class FuelSysAndFullFusMassTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('sys_and_fus', FuelSysAndFullFuselageMass(), promotes=['*'])
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830.0, units='lbm')
        self.prob.model.set_input_defaults('wing_mounted_mass', val=24446.343040697346, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults('fuel_mass', val=42893, units='lbm')
        self.prob.model.set_input_defaults('wingfuel_mass_min', val=32853, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


# this is the large single aisle 1 V3 test case
class FuselageMassTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('fuselage', FuselageMass(), promotes=['*'])

        self.prob.model.set_input_defaults('fus_mass_full', val=102270, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, val=4000, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.TailBoom.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18763, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-12, rtol=1e-12)


class FuselageMassTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('fuselage', FuselageMass(), promotes=['*'])

        self.prob.model.set_input_defaults('fus_mass_full', val=102270, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, val=4000, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.TailBoom.LENGTH, val=129.4, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')

        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class FuelMassTestCase(unittest.TestCase):  # this is the large single aisle 1 V3 test case
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('fuel', FuelMass(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, val=1759, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Mission.OPERATING_MASS, val=96506.8, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_des', val=36000, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults('payload_mass_max', val=46040, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob['fuel_mass'], 42893, tol)
        assert_near_equal(self.prob['fuel_mass_required'], 42892.0, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 32853, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class FuelMassTestCase2(unittest.TestCase):
    """Test mass-weight conversion."""

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        prob = om.Problem()
        prob.model.add_subsystem('fuel', FuelMass(), promotes=['*'])
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, val=1759, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Mission.OPERATING_MASS, val=94505.8, units='lbm')
        prob.model.set_input_defaults('payload_mass_des', val=36000, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        prob.model.set_input_defaults('payload_mass_max', val=46040, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, val=0, units='unitless')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class BWBFuelSysAndFullFusMassTestCase(unittest.TestCase):
    """Using BWB data."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('sys_and_fus', FuelSysAndFullFuselageMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.MASS, 7055.90333649, units='lbm')
        prob.model.set_input_defaults('wing_mounted_mass', 0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults('fuel_mass', 24229.3, units='lbm')
        prob.model.set_input_defaults('wingfuel_mass_min', 9221.6, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, 10.0, units='unitless')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['fus_mass_full'], 133722.49666351, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 932.82805, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBFuselageMassTestCase(unittest.TestCase):
    """GASP data."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('fuselage', BWBFuselageMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_COEFFICIENT, 0.889, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, 4573.8833, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL, 0.2, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA, 5.0, units='lbm/ft**2'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 27159.69841266, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-11, rtol=1e-12)


class BWBFuelMassTestCase(unittest.TestCase):
    """Using BWB data."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('fuel', FuelMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, 932.82805, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Mission.OPERATING_MASS, 80126.93140329, units='lbm')
        prob.model.set_input_defaults('payload_mass_des', 33750.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults('payload_mass_max', 48750.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, 10.0, units='unitless')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['fuel_mass'], 35682.13446963, tol)
        assert_near_equal(self.prob['fuel_mass_required'], 36123.06859671, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 21123.06859671, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBFuelAndOEMTestCase(unittest.TestCase):
    """Using BWB data."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wing_calcs', FuelComponents(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        # prob.model.set_input_defaults(Aircraft.Propulsion.MASS, 16161.21, units='lbm')
        # prob.model.set_input_defaults(Aircraft.Controls.MASS, 1942.3, units='lbm')
        # prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, 43566.079, units='lbm')
        # prob.model.set_input_defaults(
        #     Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, 20876.477, units='lbm'
        # )
        # prob.model.set_input_defaults(Mission.USEFUL_LOAD, 5736.3, units='lbm')
        prob.model.set_input_defaults(Mission.OPERATING_MASS, 88282.366, units='lbm')
        prob.model.set_input_defaults('fuel_mass_required', 26652.3, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 605.90781747, units='ft**3'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.VOLUME_MARGIN, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuel.TOTAL_CAPACITY, 26652.3, units='lbm')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 61717.634, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 586.08986, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1233.80378225, tol)
        assert_near_equal(self.prob[Mission.OPERATING_MASS], 88282.366, tol)

        assert_near_equal(self.prob['payload_mass_max_fuel'], 35065.334, tol)
        assert_near_equal(self.prob['volume_wingfuel_mass'], 30308.86876357, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 30308.86876357, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class BWBBodyCalculationTest(unittest.TestCase):
    """Using BWB data."""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_DESIGN, 532.8, units='ft**3')
        prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, 1159.1, units='ft**3'
        )
        prob.model.set_input_defaults('fuel_mass_min', 9229.6045, units='lbm')
        prob.model.set_input_defaults('fuel_mass_required', 26652.3, units='lbm')
        prob.model.set_input_defaults('max_wingfuel_mass', 26646.849, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 532.7, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults('fuel_mass', 24229.0, units='lbm')
        prob.model.set_input_defaults(Mission.OPERATING_MASS, 79825.3, units='lbm')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Fuel.TOTAL_CAPACITY], 24234.451, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 5.451, tol)
        assert_near_equal(self.prob['extra_fuel_volume'], 0.69314718, tol)
        assert_near_equal(self.prob['max_extra_fuel_mass'], 34.67277748, tol)
        assert_near_equal(self.prob['wingfuel_mass_min'], 9194.93172252, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
