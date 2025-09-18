import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.fuel import (
    BodyTankCalculations,
    FuelAndOEMOutputs,
    FuelMass,
    FuelMassGroup,
    FuelSysAndFullFuselageMass,
    FuselageMass,
    StructMass,
    BWBFuselageMass,
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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=42892.0, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=55725.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=1114.0, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=42893.1, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.OPERATING_MASS, val=96508, units='lbm')

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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=44982.7, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=43852.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=876.7, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=44973.0, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.OPERATING_MASS, val=94417, units='lbm')

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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=44982.7, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=43852.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=876.7, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=44973.0, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.OPERATING_MASS, val=94417, units='lbm')

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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=42892.0, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=55725.1, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=1114.0, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=42893.1, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.OPERATING_MASS, val=96508, units='lbm')

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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=30085.342, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=33892.8, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=677.554, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=145388, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=30085.342, units='lbm')
        self.prob.model.set_input_defaults(
            Mission.Summary.OPERATING_MASS, val=84502.658, units='lbm'
        )

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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=30085.342, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=33892.8, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=677.554, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=145388, units='lbm')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=30085.342, units='lbm')
        self.prob.model.set_input_defaults(
            Mission.Summary.OPERATING_MASS, val=84502.658, units='lbm'
        )

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
    """
    It tests the case Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES = True.
    """

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
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=27968.49344063, units='lbm'
        )
        self.prob.model.set_input_defaults('max_wingfuel_mass', val=31051.56633854, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, val=620.75516364, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=143100.0, units='lbm')
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS, val=26236.86063849, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Mission.Summary.OPERATING_MASS, val=84331.50655937, units='lbm'
        )

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
class FuelAndOEMTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('wing_calcs', FuelAndOEMOutputs(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Propulsion.MASS, val=16129, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, val=50461.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=21089.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_USEFUL_LOAD, val=4932.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Mission.Summary.FUEL_MASS_REQUIRED, val=42892.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 1114, units='ft**3'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')
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
        assert_near_equal(self.prob[Mission.Summary.OPERATING_MASS], 96508.0, tol)

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
        prob.model.add_subsystem('wing_calcs', FuelAndOEMOutputs(), promotes=['*'])
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults(Aircraft.Propulsion.MASS, val=16129, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, val=50461.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=21089.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, val=4932.0, units='lbm')
        prob.model.set_input_defaults(Mission.Summary.FUEL_MASS_REQUIRED, val=42892.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 1114, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')
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

        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830.0, units='lbm')
        self.prob.model.set_input_defaults('wing_mounted_mass', val=24446.343040697346, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=42893, units='lbm')
        self.prob.model.set_input_defaults('wingfuel_mass_min', val=32853, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

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
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830.0, units='lbm')
        self.prob.model.set_input_defaults('wing_mounted_mass', val=24446.343040697346, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, val=42893, units='lbm')
        self.prob.model.set_input_defaults('wingfuel_mass_min', val=32853, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

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
    """
    Test mass-weight conversion
    """

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


# this is the large single aisle 1 V3 test case
class StructMassTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('struct', StructMass(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS, val=18763, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2275, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=2297, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=3785, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50461.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-12, rtol=1e-12)


class StructMassTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fuel as fuel

        fuel.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('struct', StructMass(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS, val=18763, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2275, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=2297, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=3785, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class FuelMassTestCase(unittest.TestCase):  # this is the large single aisle 1 V3 test case
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('fuel', FuelMass(), promotes=['*'])

        self.prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, val=50461.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, val=1759, units='lbm')
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults('eng_comb_mass', val=14370.8, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=21089.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_USEFUL_LOAD, val=4932.0, units='lbm'
        )
        self.prob.model.set_input_defaults('payload_mass_des', val=36000, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults('payload_mass_max', val=46040, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS], 42893, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.MASS], 16129, tol)
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 42892.0, tol)
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
        prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, val=50461.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, val=1759, units='lbm')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        prob.model.set_input_defaults('eng_comb_mass', val=14370.8, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=21089.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, val=4932.0, units='lbm')
        prob.model.set_input_defaults('payload_mass_des', val=36000, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        prob.model.set_input_defaults('payload_mass_max', val=46040, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


# this is the large single aisle 1 V3 test case
class FuelMassGroupTestCase1(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', FuelMassGroup(), promotes=['*'])

        # top level
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15830, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3895, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=21089.0, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, val=4932, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 1114.0, units='ft**3'
        )

        # sys and fus
        self.prob.model.set_input_defaults('wing_mounted_mass', val=24446.343040697346, units='lbm')

        # fus and struct
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
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2275, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=2297, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=3785, units='lbm'
        )
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
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2275, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=2297, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=3785, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )

        # fuel
        self.prob.model.set_input_defaults('eng_comb_mass', val=14370.8, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_des', val=36000, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_max', val=46040, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        # wingfuel
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 78894, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1577.160566039489, tol)
        assert_near_equal(self.prob[Mission.Summary.OPERATING_MASS], 96508.0, tol)
        assert_near_equal(
            self.prob['payload_mass_max_fuel'], 36000, tol
        )  # note: this is calculated differently in V3, so this is the V3.6 value
        assert_near_equal(self.prob['volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 55725.1, tol)

        # sys and fus
        assert_near_equal(self.prob['fus_mass_full'], 102270, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1759, tol)

        # fus and struct
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50461.0, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18763, tol)

        # fuel
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS], 42893, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.MASS], 16129, tol)
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 42892.0, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 32853, tol)

        # body tank
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(
            self.prob['extra_fuel_volume'], 0.69314718, tol
        )  # note: not in version 3 output, calculated by hand
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 34.67277748, tol
        )  # note: not in version 3 output, calculated by hand
        # note: Aircraft.Fuel.TOTAL_CAPACITY is calculated differently in V3, so it is not included here

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-12)


class FuelMassGroupTestCase2(
    unittest.TestCase
):  # this is v 3.6 large single aisle 1 test case with wing loading of 150 psf and fuel margin of 10%
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', FuelMassGroup(), promotes=['*'])

        # top level
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=13833, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3632, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=21031.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_USEFUL_LOAD, val=4885.18, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 876.7, units='ft**3'
        )

        # sys and fus
        self.prob.model.set_input_defaults(
            'wing_mounted_mass', val=24446.343040697346, units='lbm'
        )  # note: calculated by hand

        # fus and struct
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, val=4000, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.TailBoom.LENGTH, val=129.5, units='ft'
        )  # note: calculated by hand
        self.prob.model.set_input_defaults(
            'pylon_len', val=0, units='ft'
        )  # note: calculated by hand
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.75, units='unitless'
        )
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2181, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=2158, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=7511, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=3785, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )

        # fuel
        self.prob.model.set_input_defaults('eng_comb_mass', val=14370.8, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_des', val=36000, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_max', val=46040, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        # wingfuel
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 80982.7, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1618.9, tol)
        assert_near_equal(self.prob[Mission.Summary.OPERATING_MASS], 94417.0, tol)
        assert_near_equal(self.prob['payload_mass_max_fuel'], 34879.2, tol)
        assert_near_equal(self.prob['volume_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 43852.1, tol)

        # sys and fus
        assert_near_equal(
            self.prob['fus_mass_full'], 107806.75695930266, tol
        )  # note: calculated by hand
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 2029, tol)

        # fus and struct
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 48470.0, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 19002.0, 0.00054
        )  # tol is slightly higher because GASP iteration is less rigorous.

        # fuel
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS], 44982.7, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.MASS], 16399.0, tol)
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 44982.7, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 34942.7, tol)

        # body tank
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 1130.6, 0.0112
        )  # tol is slightly higher because GASP iteration is less rigorous, and also because of numerical issues with inputs.
        assert_near_equal(
            self.prob['extra_fuel_volume'], 112.5, 0.0022
        )  # tol is slightly higher because GASP iteration is less rigorous.
        assert_near_equal(
            self.prob['max_extra_fuel_mass'], 5628.9, 0.0025
        )  # tol is slightly higher because GASP iteration is less rigorous.

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-12)


class FuelMassGroupTestCase3(unittest.TestCase):  # this is v 3.6 advanced tube and wing case
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('group', FuelMassGroup(), promotes=['*'])

        # top level
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=145388.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS, val=15098.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=3765, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Design.FIXED_EQUIPMENT_MASS, val=17201.0, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, val=4701, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 677.6, units='ft**3'
        )

        # sys and fus
        self.prob.model.set_input_defaults(
            'wing_mounted_mass', val=17272.898624554255, units='lbm'
        )  # note: calculated by hand

        # fus and struct
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, val=4209, units='ft**2')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.TailBoom.LENGTH, val=119.03, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=4.484, units='unitless'
        )
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2018, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=1500, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=6140, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=2795, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.TailBoom.LENGTH, val=119.03, units='ft')
        self.prob.model.set_input_defaults('pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('min_dive_vel', val=420, units='kn')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=4.484, units='unitless'
        )
        self.prob.model.set_input_defaults('MAT', val=0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, val=2018, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, val=1500, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, val=6140, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, val=2795, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )

        # fuel
        self.prob.model.set_input_defaults('eng_comb_mass', val=9328.2, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_des', val=30800, units='lbm')
        self.prob.model.set_input_defaults('payload_mass_max', val=46770.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10, units='unitless')

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4

        # wingfuel
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 62427.2, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1248.0, tol)
        assert_near_equal(self.prob[Mission.Summary.OPERATING_MASS], 82961.0, tol)
        assert_near_equal(self.prob['volume_wingfuel_mass'], 33892.8, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 33892.8, tol)

        # sys and fus
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1426, tol)

        # fus and struct
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 46539.0, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18988, tol)

        # fuel
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS], 31627.2, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.MASS], 10755.0, tol)
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 31627, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 15657.2, tol)

        # body tank
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        assert_near_equal(self.prob['extra_fuel_volume'], 17.9, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


class BWBFuelSysAndFullFusMassTestCase(unittest.TestCase):
    """Using BWB data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('sys_and_fus', FuelSysAndFullFuselageMass(), promotes=['*'])

        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.MASS, 7055.90333649, units='lbm')
        prob.model.set_input_defaults('wing_mounted_mass', 0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, 24229.3, units='lbm')
        prob.model.set_input_defaults('wingfuel_mass_min', 9221.6, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, 10.0, units='unitless')

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
    """GASP data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('fuselage', BWBFuselageMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_COEFFICIENT, 0.889, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, 4573.8833, units='ft**2')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
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


class BWBStructMassTestCase(unittest.TestCase):
    """Using BWB data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('struct', StructMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuselage.MASS, 27159.69841266, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MASS, 7055.90333649, units='lbm')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, 1.02401953, units='lbm')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, 864.17404177, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, 7800.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, 2230.13208284, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.STRUCTURAL_MASS_INCREMENT, 0.0, units='lbm')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 45110.93189329, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-12, rtol=1e-12)


class BWBFuelMassTestCase(unittest.TestCase):
    """Using BWB data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('fuel', FuelMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, 45110.93189329, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, 932.82805, units='lbm')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults('eng_comb_mass', 6934.7, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, 2115.19946, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.FIXED_EQUIPMENT_MASS, 20876.477, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, 4156.795, units='lbm')
        prob.model.set_input_defaults('payload_mass_des', 33750.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults('payload_mass_max', 48750.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, 10.0, units='unitless')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS], 35682.13446963, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.MASS], 7867.52805, tol)
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 36123.06859671, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 21123.06859671, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBFuelAndOEMTestCase(unittest.TestCase):
    """Using BWB data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wing_calcs', FuelAndOEMOutputs(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Propulsion.MASS, 16161.21, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, 1942.3, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.STRUCTURE_MASS, 43566.079, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.FIXED_EQUIPMENT_MASS, 20876.477, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, 5736.3, units='lbm')
        prob.model.set_input_defaults(Mission.Summary.FUEL_MASS_REQUIRED, 26652.3, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 605.90781747, units='ft**3'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuel.TOTAL_CAPACITY, 26652.3, units='lbm')

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 61717.634, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 586.08986, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1233.80378225, tol)
        assert_near_equal(self.prob[Mission.Summary.OPERATING_MASS], 88282.366, tol)

        assert_near_equal(self.prob['payload_mass_max_fuel'], 35065.334, tol)
        assert_near_equal(self.prob['volume_wingfuel_mass'], 30308.86876357, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 30308.86876357, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


@use_tempdirs
class BWBBodyCalculationTest(unittest.TestCase):
    """Using BWB data"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('wing_calcs', BodyTankCalculations(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_DESIGN, 532.8, units='ft**3')
        prob.model.set_input_defaults(
            Aircraft.Fuel.WING_VOLUME_STRUCTURAL_MAX, 1159.1, units='ft**3'
        )
        prob.model.set_input_defaults('fuel_mass_min', 9229.6045, units='lbm')
        prob.model.set_input_defaults(Mission.Summary.FUEL_MASS_REQUIRED, 26652.3, units='lbm')
        prob.model.set_input_defaults('max_wingfuel_mass', 26646.849, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 532.7, units='ft**3')
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, 24229.0, units='lbm')
        prob.model.set_input_defaults(Mission.Summary.OPERATING_MASS, 79825.3, units='lbm')

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


@use_tempdirs
class BWBFuelMassGroupTest(unittest.TestCase):
    """Using BWB data"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.NUM_ENGINES, val=[2], units='unitless')
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')

        prob = self.prob = om.Problem()
        prob.model.add_subsystem('group', FuelMassGroup(), promotes=['*'])

        # FuelSysAndFullFuselageMass
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.MASS, 6962.4, units='lbm')
        prob.model.set_input_defaults('wing_mounted_mass', 0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, 6.687, units='lbm/galUS')
        prob.model.set_input_defaults(Mission.Summary.FUEL_MASS, 33268.2, units='lbm')
        # prob.model.set_input_defaults('wingfuel_mass_min', 0.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, 10.0, units='unitless')

        # BWBFuselageMass
        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_COEFFICIENT, 0.889, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, 4573.8833, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_RATIO_AFTBODY_TO_TOTAL, 0.2, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.AFTBODY_MASS_PER_UNIT_AREA, 5.0, units='lbm/ft**2'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, 1283.5249, units='ft**2')

        # StructMass
        # prob.model.set_input_defaults(Aircraft.Fuselage.MASS, 27159.7, units='lbm')
        prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MASS, 1.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS, 790.1, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.TOTAL_MASS, 7800.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS, 2153.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Design.STRUCTURAL_MASS_INCREMENT, 0.0, units='lbm')

        # FuelMass
        prob.model.set_input_defaults('eng_comb_mass', 7311.5, units='lbm')
        prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, 2115, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.FIXED_EQUIPMENT_MASS, 20876, units='lbm')
        prob.model.set_input_defaults(Aircraft.Design.FIXED_USEFUL_LOAD, 5775, units='lbm')
        prob.model.set_input_defaults('payload_mass_des', 33750.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, 0.035, units='unitless'
        )
        prob.model.set_input_defaults('payload_mass_max', 48750.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, 10.0, units='unitless')

        # FuelAndOEMOutputs
        prob.model.set_input_defaults(Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX, 783.6, units='ft**3')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        """
        Testing GASP data case:
        Aircraft.Fuel.FUEL_SYSTEM_MASS -- WFSS = 1281
        fus_mass_full -- 124040.19837521 -- WX = 121864
        Mission.Summary.OPERATING_MASS -- OWE = 82982.
        fuel_and_oem.OEM_wingfuel_mass -- WFWOWE(WFW_MAX) = 67018.2
        fuel_and_oem.OEM_fuel_vol -- FVOLW_MAX = 1339.8
        fuel_and_oem.payload_mass_max_fuel -- WPLMXF = 30423.2
        max_wingfuel_mass -- WFWMX = 30309.0
        Aircraft.Design.STRUCTURE_MASS -- WST = 45623.
        Aircraft.Fuselage.MASS -- WB = 27160.
        Mission.Summary.FUEL_MASS -- WFADES = 33268.2
        Aircraft.Propulsion.MASS -- WP = 8592.
        Mission.Summary.FUEL_MASS_REQUIRED -- WFAREQ = 36595.0
        fuel_mass_min -- WFAMIN = 18268.2
        Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY -- 0.0
        Aircraft.Fuel.TOTAL_CAPACITY -- WFAMAX = 33268.2
        body_tank.extra_fuel_volume -- FVOLAUX = 0.2
        body_tank.max_extra_fuel_mass -- 0.0
        wingfuel_mass_min -- WFWMIN = 11982.2
        Note: can not match Aircraft.Fuel.FUEL_SYSTEM_MASS to GASP. Guess it is WFSS
        """
        self.prob.run_model()

        tol = 1e-7

        # sys and fus
        assert_near_equal(self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1308.89996255, tol)
        assert_near_equal(self.prob['fus_mass_full'], 124040.19837521, tol)

        # wingfuel
        assert_near_equal(self.prob[Mission.Summary.OPERATING_MASS], 82252.59837521, tol)
        assert_near_equal(self.prob['OEM_wingfuel_mass'], 67747.40162479, tol)
        assert_near_equal(self.prob['OEM_fuel_vol'], 1354.34550784, tol)
        assert_near_equal(self.prob['payload_mass_max_fuel'], 33750.0, tol)
        # assert_near_equal(self.prob['volume_wingfuel_mass'], 26646.849, tol)
        assert_near_equal(self.prob['max_wingfuel_mass'], 39197.43049744, tol)

        # fus and struct
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 44866.19841266, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 27159.69841266, tol)

        # fuel
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS], 33997.40162479, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.MASS], 8620.39996255, tol)
        assert_near_equal(self.prob[Mission.Summary.FUEL_MASS_REQUIRED], 33997.40162479, tol)
        assert_near_equal(self.prob['fuel_mass_min'], 18997.40162479, tol)

        # body tank
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0.0, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.TOTAL_CAPACITY], 33997.40162479, tol)
        assert_near_equal(self.prob['extra_fuel_volume'], 0.0, tol)
        assert_near_equal(self.prob['max_extra_fuel_mass'], 0.0, tol)
        assert_near_equal(self.prob['wingfuel_mass_min'], 18997.40162479, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
