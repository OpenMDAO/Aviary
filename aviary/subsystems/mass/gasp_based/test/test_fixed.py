import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary import constants
from aviary.constants import RHO_SEA_LEVEL_ENGLISH
from aviary.subsystems.mass.gasp_based.fixed import (
    ControlMass,
    ElectricAugmentationMass,
    EngineMass,
    FixedMassGroup,
    GearMass,
    HighLiftMass,
    MassParameters,
    PayloadMass,
    TailMass,
)
from aviary.utils.aviary_values import AviaryValues, get_keys
from aviary.variable_info.functions import extract_options, setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
class MassParametersTestCase1(unittest.TestCase):
    """this is large single aisle 1 v3 bug fixed test case."""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, val=0)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SWEEP, val=25, units='deg'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=118.8, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults('max_mach', val=0.9, units='unitless')  # bug fixed value
        self.prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2203729275531838, tol
        )  # bug fixed value
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)  # bug fixed value
        assert_near_equal(self.prob['c_gear_loc'], 1, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Engine.POSITION_FACTOR], 0.95, tol)
        assert_near_equal(self.prob['half_sweep'], 0.3947081519145335, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class MassParametersTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, val=2, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=117.8, units='ft'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            'max_mach', val=0.72, units='unitless'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2213063198183813, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)
        assert_near_equal(self.prob['c_gear_loc'], 0.95, tol)  # not actual bug fixed value
        # not actual bug fixed value
        assert_near_equal(self.prob[Aircraft.Engine.POSITION_FACTOR], 1, tol)
        assert_near_equal(self.prob['half_sweep'], 0.3947081519145335, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class MassParametersTestCase3(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=3, units='unitless')
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, val=0, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=117.8, units='ft'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            'max_mach', val=0.72, units='unitless'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2213063198183813, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)
        assert_near_equal(self.prob['c_gear_loc'], 0.95, tol)  # not actual bug fixed value
        assert_near_equal(
            self.prob[Aircraft.Engine.POSITION_FACTOR], 0.98, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['half_sweep'], 0.3947081519145335, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class MassParametersTestCase4(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=4, units='unitless')
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, val=0, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=117.8, units='ft'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            'max_mach', val=0.72, units='unitless'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2213063198183813, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)
        assert_near_equal(self.prob['c_gear_loc'], 0.95, tol)  # not actual bug fixed value
        assert_near_equal(
            self.prob[Aircraft.Engine.POSITION_FACTOR], 0.95, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['half_sweep'], 0.3947081519145335, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class MassParametersTestCase5(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, val=4, units='unitless')
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, val=0, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=117.8, units='ft'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(
            'max_mach', val=0.9, units='unitless'
        )  # not actual bug fixed value
        self.prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2213063198183813, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)
        assert_near_equal(self.prob['c_gear_loc'], 0.95, tol)  # not actual bug fixed value
        assert_near_equal(
            self.prob[Aircraft.Engine.POSITION_FACTOR], 0.9, tol
        )  # not actual bug fixed value
        assert_near_equal(self.prob['half_sweep'], 0.3947081519145335, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


# this is the large single aisle 1 V3 test case
@use_tempdirs
class PayloadMassTestCase(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(
            Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm'
        )  # bug fixed value and original value

        self.prob = om.Problem()
        self.prob.model.add_subsystem('payload', PayloadMass(), promotes=['*'])
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 36000, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['payload_mass_des'], 36000, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['payload_mass_max'], 46040, tol
        )  # bug fixed value and original value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class ElectricAugmentationTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        options = {
            Aircraft.Propulsion.TOTAL_NUM_ENGINES: 2,
        }
        self.prob.model.add_subsystem('aug', ElectricAugmentationMass(**options), promotes=['*'])

        self.prob.model.set_input_defaults(
            'motor_power', val=830, units='kW'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'motor_voltage', val=850, units='V'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'max_amp_per_wire', val=260, units='A'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'safety_factor', val=1.0, units='unitless'
        )  # not included in eTTVW v3.6
        self.prob.model.set_input_defaults(
            Aircraft.Electrical.HYBRID_CABLE_LENGTH, val=65.6, units='ft'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'wire_area', val=0.0015, units='ft**2'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'rho_wire', val=565, units='lbm/ft**3'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'battery_energy', val=6077, units='MJ'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'motor_eff', val=0.98, units='unitless'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'inverter_eff', val=0.99, units='unitless'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'transmission_eff', val=0.975, units='unitless'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'battery_eff', val=0.975, units='unitless'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'rho_battery', val=0.5, units='kW*h/kg'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'motor_spec_mass', val=4, units='hp/lbm'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'inverter_spec_mass', val=12, units='kW/kg'
        )  # electrified diff configuration value v3.6
        self.prob.model.set_input_defaults(
            'TMS_spec_mass', val=0.125, units='lbm/kW'
        )  # electrified diff configuration value v3.6

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        assert_near_equal(
            self.prob['aug_mass'], 9394.3, 0.0017
        )  # electrified diff configuration value v3.6. Higher tol because num_wires is discrete in GASP and is not in Aviary

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=4e-12, rtol=1e-12)


@use_tempdirs
class EngineTestCase1(unittest.TestCase):  # this is the large single aisle 1 V3 test case
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606.0, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 3785.0, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765.0 / 2, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['eng_comb_mass'], 14370.8, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['wing_mounted_mass'], 24446.343040697346, tol
        )  # bug fixed value and original value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-12)


@use_tempdirs
class EngineTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.HAS_PROPELLERS, val=[True], units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'prop_mass', val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'aug_mass', val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606.0, tol
        )  # note: these are only the right values because this was given a prop mass of zero. This is not a large single aisle test case
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 3785.0, tol
        )  # note: these are only the right values because this was given a prop mass of zero. This is not a large single aisle test case
        assert_near_equal(
            self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765.0 / 2, tol
        )  # note: these are only the right values because this was given a prop mass of zero. This is not a large single aisle test case
        assert_near_equal(
            self.prob['eng_comb_mass'], 14370.8, tol
        )  # note: these are only the right values because this was given a prop mass of zero. This is not a large single aisle test case
        assert_near_equal(
            self.prob['prop_mass_all'], 0, tol
        )  # note: these are only the right values because this was given a prop mass of zero. This is not a large single aisle test case
        assert_near_equal(
            self.prob['wing_mounted_mass'], 24446.343040697346, tol
        )  # note: these are only the right values because this was given a prop mass of zero. This is not a large single aisle test case

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-11, rtol=1e-12)


# arbitrary test case with multiple engine types
@use_tempdirs
class EngineTestCaseMultiEngine(unittest.TestCase):
    def test_case_1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')

        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 6)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, np.array([0.14, 0.19]))

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=[0.21366, 0.15], units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0, 18000], units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=[3, 2.45], units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=[339.58, 235.66], units='ft**2'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=[1.25, 1.28], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=[1, 0.9], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=[0.35, 0.0, 0.1], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS, val=6384.35, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model_options['*'] = extract_options(options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 23405.94, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 8074.09809932, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Engine.ADDITIONAL_MASS], [882.4158, 513.0], tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['eng_comb_mass'], 26142.7716, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['wing_mounted_mass'], 41417.49593562, tol
        )  # bug fixed value and original value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


@use_tempdirs
class TailTestCase(unittest.TestCase):  # this is the large single aisle 1 V3 test case
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('tail', TailMass(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0, units='rad'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SPAN, val=28.22, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.SPAN, val=42.59, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=118.8, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=381.8, units='ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_ARM, val=55.1, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ROOT_CHORD, val=13.261162230765065, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=476.8, units='ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_ARM, val=50.3, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ROOT_CHORD, val=18.762708015981357, units='ft'
        )  # bug fixed value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob['loc_MAC_vtail'], 0.44959578484694906, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 2285, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 2312, tol)  # bug fixed value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


# this is a different configuration with turbofan_23k_1 test case
@use_tempdirs
class HighLiftTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, val=2)

        self.prob.model.add_subsystem('HL', HighLiftMass(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, val=1764.6, units='ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SLAT_CHORD_RATIO, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FLAP_CHORD_RATIO, val=0.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.346, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FLAP_SPAN_RATIO, val=0.88, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=93.1, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=185.8, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, val=13.979, units='ft')
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.3648, units='unitless'
        )

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 4829.6, tol)

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', show_only_incorrect=True
        )
        assert_check_partials(partial_data, atol=5e-10, rtol=1e-12)


# this is the large single aisle 1 V3 test case
@use_tempdirs
class ControlMassTestCase(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('control_mass', ControlMass(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1392.1, units='ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.951, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 3945, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


@use_tempdirs
class GearTestCase1(unittest.TestCase):  # this is the large single aisle 1 V3 test case
    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('gear_mass', GearMass(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.AVG_DIAMETER, val=7.35, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0, units='unitless'
        )

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.LandingGear.TOTAL_MASS], 7511, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol
        )  # bug fixed value and original value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-11, rtol=1e-12)


@use_tempdirs
class GearTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        self.prob = om.Problem()
        self.prob.model.add_subsystem('gear_mass', GearMass(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.LandingGear.TOTAL_MASS], 7016, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 5963.6, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


@use_tempdirs
class GearTestCaseMultiengine(unittest.TestCase):
    def test_case1(self):
        options = get_option_defaults()

        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2, 4]))

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'gear_mass',
            GearMass(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=[0.0, 0.15], units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.AVG_DIAMETER, val=[7.5, 8.22], units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04)
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=152000)

        self.prob.model_options['*'] = extract_options(options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 5614.3311546, tol
        )  # bug fixed value and original value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)


# this is the large single aisle 1 V3 test case
@use_tempdirs
class FixedMassGroupTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(
            Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm'
        )  # bug fixed value and original value
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FixedMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=118.8, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1392.1, units='ft**2'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SWEEP, val=25, units='deg'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'max_mach', val=0.9, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0, units='rad'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SPAN, val=28.22, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.SPAN, val=42.59, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=381.8, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_ARM, val=55.1, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ROOT_CHORD, val=13.261162230765065, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=476.8, units='ft**2'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_ARM, val=50.3, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ROOT_CHORD, val=18.762708015981357, units='ft'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.966, units='unitless'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.951, units='unitless'
        )  # bug fixed value
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.AVG_DIAMETER, val=7.35, units='ft'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=128)
        self.prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, val=17.48974)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol
        )  # bug fixed value and original value

        assert_near_equal(
            self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2203729275531838, tol
        )  # bug fixed value
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)  # bug fixed value and original value
        assert_near_equal(self.prob['c_gear_loc'], 1, tol)  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Engine.POSITION_FACTOR], 0.95, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['half_sweep'], 0.3947081519145335, tol
        )  # bug fixed value and original value

        assert_near_equal(
            self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 36000, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['payload_mass_des'], 36000, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['payload_mass_max'], 46040, tol
        )  # bug fixed value and original value

        assert_near_equal(
            self.prob['tail.loc_MAC_vtail'], 0.44959578484694906, tol
        )  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 2285, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 2312, tol)  # bug fixed value
        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 4082.1, tol)

        # bug fixed value
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 3945, tol)

        assert_near_equal(
            self.prob[Aircraft.LandingGear.TOTAL_MASS], 7511, tol
        )  # bug fixed value and original value

        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 3785, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['eng_comb_mass'], 14370.8, tol
        )  # bug fixed value and original value
        assert_near_equal(
            self.prob['wing_mounted_mass'], 24446.343040697346, tol
        )  # bug fixed value and original value

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-11, rtol=1e-12)


@use_tempdirs
class FixedMassGroupTestCase2(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, val=2, units='unitless')
        options.set_val(Aircraft.Engine.HAS_PROPELLERS, val=[True], units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=False, units='unitless')
        options.set_val(
            Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm'
        )  # bug fixed value and original value
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'group',
            FixedMassGroup(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=117.8, units='ft'
        )  # original GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SWEEP, val=25, units='deg'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'max_mach', val=0.72, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=10 / 117.8, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0.1, units='rad'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SPAN, val=28, units='ft'
        )  # original GASP value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.SPAN, val=42.25, units='ft'
        )  # original GASP value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_ARM, val=54.7, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ROOT_CHORD, val=13.16130387591471, units='ft'
        )  # original GASP value
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units='ft**2'
        )  # original GASP value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_ARM, val=49.9, units='ft'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ROOT_CHORD, val=18.61267549773935, units='ft'
        )  # original GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )  # bug fixed value and original value

        self.prob.model.set_input_defaults(
            'augmentation.motor_power', val=200, units='kW'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.motor_voltage', val=50, units='V'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.max_amp_per_wire', val=50, units='A'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.safety_factor', val=1.33, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Electrical.HYBRID_CABLE_LENGTH, val=200, units='ft'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.wire_area', val=0.0015, units='ft**2'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.rho_wire', val=1, units='lbm/ft**3'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.battery_energy', val=1, units='MJ'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.motor_eff', val=1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.inverter_eff', val=1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.transmission_eff', val=1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.battery_eff', val=1, units='unitless'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.rho_battery', val=200, units='kW*h/kg'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.motor_spec_mass', val=10, units='hp/lbm'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.inverter_spec_mass', val=10, units='kW/kg'
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            'augmentation.TMS_spec_mass', val=0.125, units='lbm/kW'
        )  # electrified diff configuration value v3.6

        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        # self.prob.model.set_input_defaults(
        #     Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        # )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(
            'engine.prop_mass', val=0, units='lbm'
        )  # bug fixed value and original value
        self.prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=128)
        self.prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, val=17.48974)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 5963.6, tol
        )  # not actual GASP value
        assert_near_equal(self.prob['aug_mass'], 228.51036478, tol)  # not actual GASP value

        assert_near_equal(self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.2213063198183813, tol)
        assert_near_equal(self.prob['c_strut_braced'], 0.9928, tol)  # not actual GASP value
        assert_near_equal(self.prob['c_gear_loc'], 1, tol)  # not actual GASP value
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Engine.POSITION_FACTOR], 1, tol)
        assert_near_equal(
            self.prob['half_sweep'], 0.3947081519145335, tol
        )  # bug fixed and original value

        assert_near_equal(
            self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 36000, tol
        )  # bug fixed and original value
        assert_near_equal(self.prob['payload_mass_des'], 36000, tol)  # bug fixed and original value
        assert_near_equal(self.prob['payload_mass_max'], 46040, tol)  # bug fixed and original value

        assert_near_equal(self.prob['tail.loc_MAC_vtail'], 1.799, tol)  # not actual GASP value
        # original GASP value
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 2275, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 2297, tol)  # original GASP value
        # original GASP value
        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 4162.1, tol)

        # original GASP value
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 3895, tol)

        assert_near_equal(
            self.prob[Aircraft.LandingGear.TOTAL_MASS], 7016, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol
        )  # bug fixed and original value
        assert_near_equal(
            self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 3785, tol
        )  # bug fixed and original value
        assert_near_equal(
            self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol
        )  # bug fixed and original value
        assert_near_equal(self.prob['eng_comb_mass'], 14599.28196478, tol)  # not actual GASP value
        assert_near_equal(self.prob['wing_mounted_mass'], 24027.6, tol)  # not actual GASP value
        assert_near_equal(self.prob['engine.prop_mass_all'], 0, tol)  # bug fixed and original value

        partial_data = self.prob.check_partials(
            out_stream=None,
            method='cs',
            form='central',
        )
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


@use_tempdirs
class FixedMassGroupTestCase3(unittest.TestCase):
    # Tests partials calculations in FixedMassGroup using as complete code coverage as possible.

    def _run_case(self, data):
        prob = om.Problem()
        prob.model.add_subsystem(
            'fixed_mass',
            FixedMassGroup(),
            promotes=['*'],
        )

        setup_model_options(prob, data)

        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        prob.setup(force_alloc_complex=True)

        for key in get_keys(data):
            val, units = data.get_item(key)
            try:
                prob.set_val(key, val, units)
            except:
                pass

        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-7, rtol=1e-12)

    def setUp(self):
        import aviary.subsystems.mass.gasp_based.fixed as fixed

        # Set GRAV_ENGLISH_LBM = 1.1 to find errors that aren't
        # caught when GRAV_ENGLISH_LBM = 1 and is misplaced.
        constants.GRAV_ENGLISH_LBM = 1.1
        fixed.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.mass.gasp_based.fixed as fixed

        constants.GRAV_ENGLISH_LBM = 1.0
        fixed.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        data = AviaryValues(
            {
                Aircraft.Engine.NUM_ENGINES: (np.array([2]), 'unitless'),
                Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES: (2, 'unitless'),
                Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES: (False, 'unitless'),
                Aircraft.Engine.NUM_FUSELAGE_ENGINES: (np.array([0]), 'unitless'),
                Aircraft.CrewPayload.NUM_PASSENGERS: (150, 'unitless'),
                Aircraft.CrewPayload.Design.NUM_PASSENGERS: (150, 'unitless'),
                Aircraft.Electrical.HAS_HYBRID_SYSTEM: (False, 'unitless'),
                Aircraft.Engine.HAS_PROPELLERS: ([False], 'unitless'),
                Aircraft.Wing.FLAP_TYPE: ('plain', 'unitless'),
                Aircraft.Wing.SWEEP: (30.0, 'deg'),
                Aircraft.Wing.VERTICAL_MOUNT_LOCATION: (0, 'unitless'),
                Aircraft.Wing.TAPER_RATIO: (0.25, 'unitless'),
                Aircraft.Wing.ASPECT_RATIO: (11.0, 'unitless'),
                Aircraft.Wing.SPAN: (100.0, 'ft'),
                'max_mach': (0.9, 'unitless'),
                Aircraft.Strut.ATTACHMENT_LOCATION: (10.0, 'ft'),
                Aircraft.LandingGear.MAIN_GEAR_LOCATION: (0.2, 'unitless'),
                Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS: (200.0, 'lbm'),
                Aircraft.CrewPayload.CARGO_MASS: (0.0, 'lbm'),
                Aircraft.CrewPayload.Design.MAX_CARGO_MASS: (10040.0, 'lbm'),
                'wire_area': (0.0015, 'ft**2'),
                'rho_wire': (1.1, 'lbm/ft**3'),
                'battery_energy': (0.0, 'MJ'),
                'motor_eff': (1.5, 'unitless'),
                'inverter_eff': (0.95, 'unitless'),
                'transmission_eff': (0.96, 'unitless'),
                'battery_eff': (0.97, 'unitless'),
                'rho_battery': (210.0, 'MJ/lb'),
                'motor_spec_mass': (10.0, 'hp/lbm'),
                'inverter_spec_mass': (10.5, 'kW/lbm'),
                'TMS_spec_mass': (10.6, 'lbm/kW'),
                Aircraft.Engine.MASS_SPECIFIC: (np.array([0.21366]), 'lbm/lbf'),
                Aircraft.Engine.SCALED_SLS_THRUST: (np.array([4000.0]), 'lbf'),
                Aircraft.Nacelle.MASS_SPECIFIC: (3.0, 'lbm/ft**2'),
                Aircraft.Nacelle.SURFACE_AREA: (5.0, 'ft**2'),
                Aircraft.Engine.PYLON_FACTOR: (np.array([1.25]), 'unitless'),
                Aircraft.Engine.ADDITIONAL_MASS_FRACTION: (
                    np.array([0.14]),
                    'unitless',
                ),
                Aircraft.Engine.MASS_SCALER: (np.array([1.05]), 'unitless'),
                Aircraft.Propulsion.MISC_MASS_SCALER: (1.06, 'unitless'),
                Aircraft.Engine.WING_LOCATIONS: (np.array([0.35]), 'unitless'),
                'prop_mass': (0.5, 'lbm'),
                Aircraft.VerticalTail.TAPER_RATIO: (0.26, 'unitless'),
                Aircraft.VerticalTail.ASPECT_RATIO: (5.0, 'unitless'),
                Aircraft.VerticalTail.SWEEP: (25.0, 'deg'),
                Aircraft.VerticalTail.SPAN: (20.0, 'ft'),
                Mission.Design.GROSS_MASS: (152000.0, 'lbm'),
                Aircraft.HorizontalTail.MASS_COEFFICIENT: (1.07, 'unitless'),
                Aircraft.Fuselage.LENGTH: (120.0, 'ft'),
                Aircraft.HorizontalTail.SPAN: (14.0, 'ft'),
                Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (1.08, 'unitless'),
                Aircraft.HorizontalTail.TAPER_RATIO: (0.35, 'unitless'),
                Aircraft.VerticalTail.MASS_COEFFICIENT: (1.09, 'unitless'),
                Aircraft.HorizontalTail.AREA: (500.0, 'ft**2'),
                'min_dive_vel': (200.0, 'kn'),
                Aircraft.HorizontalTail.MOMENT_ARM: (20.0, 'ft'),
                Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (0.15, 'unitless'),
                Aircraft.HorizontalTail.ROOT_CHORD: (8.0, 'ft'),
                Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION: (0.3, 'unitless'),
                Aircraft.VerticalTail.AREA: (250.0, 'ft**2'),
                Aircraft.VerticalTail.MOMENT_ARM: (6.0, 'ft'),
                Aircraft.VerticalTail.THICKNESS_TO_CHORD: (0.12, 'unitless'),
                Aircraft.VerticalTail.ROOT_CHORD: (7.0, 'ft'),
                Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT: (1.10, 'unitless'),
                Aircraft.Wing.AREA: (3500.0, 'ft**2'),
                Aircraft.Wing.NUM_FLAP_SEGMENTS: (2, 'unitless'),
                Aircraft.Wing.SLAT_CHORD_RATIO: (0.15, 'unitless'),
                Aircraft.Wing.FLAP_CHORD_RATIO: (0.3, 'unitless'),
                Aircraft.Wing.SLAT_SPAN_RATIO: (0.9, 'unitless'),
                Aircraft.Wing.FLAP_SPAN_RATIO: (0.65, 'unitless'),
                Aircraft.Wing.LOADING: (128.0, 'lbf/ft**2'),
                Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.15, 'unitless'),
                Aircraft.Fuselage.AVG_DIAMETER: (11.0, 'ft'),
                Aircraft.Wing.CENTER_CHORD: (17.0, 'ft'),
                Mission.Landing.LIFT_COEFFICIENT_MAX: (1.8, 'unitless'),
                'density': (RHO_SEA_LEVEL_ENGLISH, 'slug/ft**3'),
                Aircraft.Wing.ULTIMATE_LOAD_FACTOR: (7.0, 'unitless'),
                Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT: (1.11, 'unitless'),
                Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: (200.0, 'lbm'),
                Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER: (1.12, 'unitless'),
                Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER: (1.13, 'unitless'),
                Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER: (
                    1.14,
                    'unitless',
                ),
                Aircraft.Controls.CONTROL_MASS_INCREMENT: (25.0, 'lbm'),
                Aircraft.LandingGear.MASS_COEFFICIENT: (1.15, 'unitless'),
                Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT: (1.16, 'unitless'),
                Aircraft.Nacelle.CLEARANCE_RATIO: (0.2, 'unitless'),
                Aircraft.Nacelle.AVG_DIAMETER: (7.5, 'ft'),
            }
        )

        # Try to cover all if-then branches in fixed.py.
        for flap_type in ['split', 'single_slotted', 'fowler']:
            for has_hybrid in [False, True]:
                for has_prop in [False, True]:
                    for gear_loc in [0.0, 0.05, 0.01]:
                        for fuse_mounted in [False, True]:
                            for num_engines in [2, 4]:
                                num_fuse_eng = num_engines if fuse_mounted else 0
                                data.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, num_fuse_eng)
                                data.set_val(Aircraft.Engine.NUM_ENGINES, [num_engines])
                                data.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, num_engines)
                                data.set_val(Aircraft.LandingGear.MAIN_GEAR_LOCATION, gear_loc)
                                data.set_val(Aircraft.Wing.FLAP_TYPE, flap_type)
                                data.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, has_hybrid)
                                data.set_val(Aircraft.Engine.HAS_PROPELLERS, [has_prop])

                                self._run_case(data)


class BWBMassParametersTestCase(unittest.TestCase):
    """GASP BWB model"""

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 2, units='unitless')

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501094, units='ft')
        prob.model.set_input_defaults('max_mach', 0.9, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.19461189, tol)
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)
        assert_near_equal(self.prob['c_gear_loc'], 0.95, tol)
        assert_near_equal(self.prob[Aircraft.Engine.POSITION_FACTOR], 1.05, tol)
        assert_near_equal(self.prob['half_sweep'], 0.47984874, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBPayloadMassTestCase(unittest.TestCase):
    "GASP BWB model"

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=225, units='lbm')

        self.prob = om.Problem()
        self.prob.model.add_subsystem('payload', PayloadMass(), promotes=['*'])
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, 0.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, 15000.0, units='lbm'
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(self.prob['payload_mass_des'], 33750.0, tol)
        assert_near_equal(self.prob['payload_mass_max'], 48750.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class BWBEngineTestCase(unittest.TestCase):
    "GASP BWB model"

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.04373)

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'engine',
            EngineMass(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, 2.5, units='lbm/ft**2')
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, 194.957186763, units='ft**2'
        )  # 6.76*3.14159265*9.18
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, 1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Propulsion.MISC_MASS_SCALER, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_MASS, 6630.0, units='lbm')
        prob.model.set_input_defaults(Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 7005.15475443, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.MASS], 487.39296691, tol)
        assert_near_equal(self.prob['pylon_mass'], 558.757916785, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 2092.30176475, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 153.16770871, tol)
        assert_near_equal(self.prob['eng_comb_mass'], 7311.49017184, tol)
        assert_near_equal(self.prob['wing_mounted_mass'], 0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-9, rtol=1e-12)


class BWBTailTestCase(unittest.TestCase):
    """GASP BWB model"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('tail', TailMass(), promotes=['*'])

        prob.model.set_input_defaults(Aircraft.VerticalTail.TAPER_RATIO, 0.366, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.ASPECT_RATIO, 1.705, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, 0.0, units='rad')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SPAN, 16.98084188, units='ft')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000.0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, 0.124, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.SPAN, 0.04467601, units='ft')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.TAPER_RATIO, 0.366, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, 0.119, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501094, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.00117064, units='ft**2')
        prob.model.set_input_defaults('min_dive_vel', 420, units='kn')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MOMENT_ARM, 29.6907417, units='ft')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.ROOT_CHORD, 0.03836448, units='ft')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.11964286, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MOMENT_ARM, 27.82191598, units='ft')
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.1, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.ROOT_CHORD, 14.58190052, units='ft')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob['loc_MAC_vtail'], 0.97683077, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 1.02401953, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 864.17404177, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class BWBHighLiftTestCase(unittest.TestCase):
    def setUp(self):
        prob = self.prob = om.Problem()

        aviary_options = get_option_defaults()
        aviary_options.set_val(Aircraft.Wing.FLAP_TYPE, val=4)
        aviary_options.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, val=2)

        prob.model.add_subsystem('HL', HighLiftMass(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85714286, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, 0.0001, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, 0.2, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, 0.831687927, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_SPAN_RATIO, 0.61, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.LOADING, 70.0, units='lbf/ft**2')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501094, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, 22.97244452, units='ft')
        prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, 1.94034, units='unitless'
        )  # 1.94034 is taken from .out file. In GASP, CLMAX is computed for different phases

        setup_model_options(self.prob, aviary_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7

        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 1068.88854499, tol)
        assert_near_equal(self.prob['flap_mass'], 1068.46572125, tol)  # WFLAP = 997.949249689
        assert_near_equal(self.prob['slat_mass'], 0.42282374, tol)  # WLED

        partial_data = self.prob.check_partials(
            out_stream=None, method='cs', show_only_incorrect=True
        )
        assert_check_partials(partial_data, atol=5e-10, rtol=1e-12)


@use_tempdirs
class BWBControlMassTestCase(unittest.TestCase):
    """GAST BWB model"""

    def setUp(self):
        prob = self.prob = om.Problem()
        prob.model.add_subsystem('control_mass', ControlMass(), promotes=['*'])

        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, 0.5, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85714286, units='ft**2')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.97744787, units='unitless'
        )
        prob.model.set_input_defaults('min_dive_vel', 420, units='kn')
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
        prob.model.set_input_defaults(Aircraft.Controls.CONTROL_MASS_INCREMENT, 0, units='lbm')

        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 2045.5556421, tol)
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 2174.28611375, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-12)


class BWBGearTestCase(unittest.TestCase):
    """GASP BWB model"""

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem('gear_mass', GearMass(), promotes=['*'])

        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, 0.0520, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, 0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.CLEARANCE_RATIO, 0.2, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, 7.35163168, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless'
        )

        setup_model_options(
            self.prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')})
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.LandingGear.TOTAL_MASS], 7800.0, tol)
        assert_near_equal(self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6630.0, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-11, rtol=1e-12)


class BWBFixedMassGroupTestCase1(unittest.TestCase):
    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=225, units='lbm')
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.04373)

        prob = self.prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FixedMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501094, units='ft')
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults('min_dive_vel', 420, units='kn')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85714286, units='ft**2')

        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults('max_mach', 0.9, units='unitless')
        prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, 0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, 15000.0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.TAPER_RATIO, 0.366, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.ASPECT_RATIO, 1.705, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, 0.0, units='rad')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SPAN, 16.98084188, units='ft')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, 0.124, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.SPAN, 0.04467601, units='ft')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, 1, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.TAPER_RATIO, 0.366, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, 0.119, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.00117064, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.MOMENT_ARM, 29.6907417, units='ft')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.1, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.ROOT_CHORD, 0.03836448, units='ft')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.11964286, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.MOMENT_ARM, 27.82191598, units='ft')
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.1, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.ROOT_CHORD, 14.58190052, units='ft')
        prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, 1.94034, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, 0.5, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.77335889, units='unitless'
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
        prob.model.set_input_defaults(Aircraft.Controls.CONTROL_MASS_INCREMENT, 0, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, 0.0520, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, 0.85, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Nacelle.CLEARANCE_RATIO, 0.2, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_DIAMETER, 7.35163168, units='ft')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SPECIFIC, 0.178884, units='lbm/lbf')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, 19580.1602, units='lbf')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, 2.5, units='lbm/ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.SURFACE_AREA, 219.95229788, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, 1.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, 1, units='unitless')

        prob.model.set_input_defaults(Aircraft.Engine.WING_LOCATIONS, 0.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38)
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, 0.0001)
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, 0.2)
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, 0.831687927)
        prob.model.set_input_defaults(Aircraft.Wing.LOADING, 70.0)
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165)
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, 22.97244452)
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_SPAN_RATIO, 0.61, units='unitless')

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 1e-7
        assert_near_equal(self.prob[Aircraft.LandingGear.TOTAL_MASS], 7800.0, tol)
        assert_near_equal(self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6630.0, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MATERIAL_FACTOR], 1.19461189, tol)
        assert_near_equal(self.prob['c_strut_braced'], 1, tol)
        assert_near_equal(self.prob['c_gear_loc'], 0.95, tol)
        assert_near_equal(self.prob[Aircraft.Engine.POSITION_FACTOR], 0.95, tol)
        assert_near_equal(self.prob['half_sweep'], 0.47984874, tol)
        assert_near_equal(self.prob[Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS], 33750.0, tol)
        assert_near_equal(self.prob['payload_mass_des'], 33750, tol)
        assert_near_equal(self.prob['payload_mass_max'], 48750, tol)
        assert_near_equal(self.prob['tail.loc_MAC_vtail'], 0.97683077, tol)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.MASS], 1.02401953, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.MASS], 864.17404177, tol)
        assert_near_equal(self.prob[Aircraft.Wing.HIGH_LIFT_MASS], 1068.88854499, tol)
        assert_near_equal(self.prob[Aircraft.Controls.TOTAL_MASS], 2114.98158947, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 7005.15475443, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.MASS], 549.8807447, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS], 2230.13208284, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 153.16770871, tol)
        assert_near_equal(self.prob['pylon_mass'], 565.18529673, tol)
        assert_near_equal(self.prob['eng_comb_mass'], 7311.49017184, tol)
        assert_near_equal(self.prob['wing_mounted_mass'], 0.0, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SURFACE_CONTROL_MASS], 1986.25111783, tol)

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-9, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
