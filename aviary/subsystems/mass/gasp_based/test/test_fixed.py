import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.fixed import (
    ElectricAugmentationMass,
    FixedMassGroup,
    HighLiftMass,
    HorizontalTailMass,
    MassParameters,
    PayloadGroup,
    VerticalTailMass,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import FlapType
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings


@use_tempdirs
class MassParametersTestCase(unittest.TestCase):
    """Tests for the MassParameters component against GASP output, including a BWB model."""

    def _make_prob(
        self,
        num_wing_engines,
        smooth_mass_discontinuities,
        sweep,
        taper_ratio,
        aspect_ratio,
        span,
        max_mach,
        main_gear_location,
        verbosity=0,
    ):
        options = AviaryValues()
        options.set_val(Settings.VERBOSITY, verbosity)
        options.set_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, num_wing_engines, units='unitless'
        )
        options.set_val(
            Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
            smooth_mass_discontinuities,
            units='unitless',
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            'parameters',
            MassParameters(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=sweep, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=taper_ratio, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=aspect_ratio, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=span, units='ft')
        prob.model.set_input_defaults(Aircraft.Design.MAX_MACH, val=max_mach, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=main_gear_location
        )

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def _check(self, prob, expected_values, tol, check_partials=False):
        prob.run_model()

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        if check_partials:
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_case1(self):
        """This is large single aisle 1 v3 bug fixed test case."""
        prob = self._make_prob(
            num_wing_engines=2,
            smooth_mass_discontinuities=False,
            sweep=25,  # bug fixed value
            taper_ratio=0.33,  # bug fixed value
            aspect_ratio=10.13,  # bug fixed value
            span=118.8,  # bug fixed value
            max_mach=0.9,  # bug fixed value
            main_gear_location=0.15,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.2203729275531838,  # bug fixed value
            'c_strut_braced': 1,  # bug fixed value
            'c_gear_loc': 1,  # bug fixed value
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 0.95,  # bug fixed value
            'half_sweep': 0.3947081519145335,  # bug fixed value
        }
        # this is the "normal" (not smoothed) code path; sibling test_case2-5 exercise
        # the same equations/Jacobian with different numbers, so only this one checks partials
        self._check(prob, expected_values, tol=1e-4, check_partials=True)

    def test_case2(self):
        prob = self._make_prob(
            num_wing_engines=0,
            smooth_mass_discontinuities=False,
            sweep=25,
            taper_ratio=0.33,
            aspect_ratio=10.13,
            span=117.8,  # not actual bug fixed value
            max_mach=0.72,  # not actual bug fixed value
            main_gear_location=0,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.2213063198183813,  # not actual bug fixed value
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,  # not actual bug fixed value
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 1,  # not actual bug fixed value
            'half_sweep': 0.3947081519145335,
        }
        self._check(prob, expected_values, tol=1e-4)

    def test_case3(self):
        prob = self._make_prob(
            num_wing_engines=3,
            smooth_mass_discontinuities=False,
            sweep=25,
            taper_ratio=0.33,
            aspect_ratio=10.13,
            span=117.8,  # not actual bug fixed value
            max_mach=0.72,  # not actual bug fixed value
            main_gear_location=0,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.2213063198183813,  # not actual bug fixed value
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,  # not actual bug fixed value
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 0.98,  # not actual bug fixed value
            'half_sweep': 0.3947081519145335,
        }
        self._check(prob, expected_values, tol=1e-4)

    def test_case4(self):
        prob = self._make_prob(
            num_wing_engines=4,
            smooth_mass_discontinuities=False,
            sweep=25,
            taper_ratio=0.33,
            aspect_ratio=10.13,
            span=117.8,  # not actual bug fixed value
            max_mach=0.72,  # not actual bug fixed value
            main_gear_location=0,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.2213063198183813,  # not actual bug fixed value
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,  # not actual bug fixed value
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 0.95,  # not actual bug fixed value
            'half_sweep': 0.3947081519145335,
        }
        self._check(prob, expected_values, tol=1e-4)

    def test_case5(self):
        prob = self._make_prob(
            num_wing_engines=4,
            smooth_mass_discontinuities=False,
            sweep=25,
            taper_ratio=0.33,
            aspect_ratio=10.13,
            span=117.8,  # not actual bug fixed value
            max_mach=0.9,  # not actual bug fixed value
            main_gear_location=0,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.2213063198183813,  # not actual bug fixed value
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,  # not actual bug fixed value
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 0.9,  # not actual bug fixed value
            'half_sweep': 0.3947081519145335,
        }
        self._check(prob, expected_values, tol=1e-4)

    def test_bwb_case1(self):
        """GASP BWB model, not smoothed."""
        prob = self._make_prob(
            num_wing_engines=0,
            smooth_mass_discontinuities=False,
            sweep=30.0,
            taper_ratio=0.27444,
            aspect_ratio=10.0,
            span=146.38501094,
            max_mach=0.9,
            main_gear_location=0,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.19461189,
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 1.05,
            'half_sweep': 0.47984874,
        }
        self._check(prob, expected_values, tol=1e-7)

    def test_bwb_case2(self):
        """GASP BWB model, smoothed."""
        prob = self._make_prob(
            num_wing_engines=0,
            smooth_mass_discontinuities=True,
            sweep=30.0,
            taper_ratio=0.27444,
            aspect_ratio=10.0,
            span=146.38501094,
            max_mach=0.9,
            main_gear_location=0,
        )
        expected_values = {
            Aircraft.Wing.MATERIAL_FACTOR: 1.19461189,
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 1.05,
            'half_sweep': 0.47984874,
        }
        # smooth_mass_discontinuities=True takes the smoothed Jacobian branch, which
        # test_case1's check never exercises, so this scenario needs its own partials check
        self._check(prob, expected_values, tol=1e-7, check_partials=True)


@use_tempdirs
class PayloadGroupTestCase(unittest.TestCase):
    """Tests for the PayloadGroup component against GASP output, including a BWB model."""

    def _make_prob(
        self, num_passengers, num_passengers_design, mass_per_pax, cargo_mass, max_cargo_mass
    ):
        options = AviaryValues()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=num_passengers, units='unitless')
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS,
            val=num_passengers_design,
            units='unitless',
        )

        prob = om.Problem()
        prob.model.add_subsystem('payload', PayloadGroup(), promotes=['*'])
        prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=cargo_mass, units='lbm')
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, val=mass_per_pax, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=max_cargo_mass, units='lbm'
        )

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        prob = self._make_prob(
            num_passengers=180,
            num_passengers_design=180,
            mass_per_pax=200,
            cargo_mass=0,
            max_cargo_mass=10040,
        )
        prob.run_model()

        expected_values = {
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 36000,  # bug fixed value and original value
            'payload_mass_des': 36000,  # bug fixed value and original value
            'payload_mass_max': 46040,  # bug fixed value and original value
        }
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, 1e-4)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    def test_bwb(self):
        """GASP BWB model."""
        prob = self._make_prob(
            num_passengers=150,
            num_passengers_design=150,
            mass_per_pax=225,
            cargo_mass=0.0,
            max_cargo_mass=15000.0,
        )
        prob.run_model()

        expected_values = {
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 33750.0,
            Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS: 33750.0,
            'payload_mass_des': 33750.0,
            'payload_mass_max': 48750.0,
        }
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, 1e-7)
        # PayloadGroup has no branching; test_case1 already covers this component's
        # single code path with a partials check, so this scenario doesn't need its own


@use_tempdirs
class ElectricAugmentationTestCase(unittest.TestCase):
    """Tests for the ElectricAugmentationMass component against GASP output."""

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

        expected_values = {
            'aug_mass': 9394.3,  # electrified diff configuration value v3.6. Higher tol because num_wires is discrete in GASP and is not in Aviary
        }
        tol = 0.0017

        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(self.prob[var_name], expected_val, tol)

        data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=4e-12, rtol=1e-12)


@use_tempdirs
class HorizontalTailMassTestCase(unittest.TestCase):
    """Tests for the HorizontalTailMass component against GASP output, including BWB."""

    def _make_prob(self, values, gravity=9.80665):
        options = AviaryValues()
        options.set_val(Mission.GRAVITY, val=gravity, units='m/s**2')

        prob = om.Problem()
        prob.model.add_subsystem('h_tail', HorizontalTailMass(), promotes=['*'])

        for var_name, (val, units) in values.items():
            prob.model.set_input_defaults(var_name, val=val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        values = {
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MASS_COEFFICIENT: (
                0.232,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.Fuselage.LENGTH: (129.4, 'ft'),  # bug fixed value and original value
            Aircraft.HorizontalTail.SPAN: (42.59, 'ft'),  # bug fixed value
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (
                1,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.TAPER_RATIO: (
                0.352,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.AREA: (381.8, 'ft**2'),  # bug fixed value
            'min_dive_vel': (420, 'kn'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MOMENT_ARM: (55.1, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (
                0.12,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.ROOT_CHORD: (13.261162230765065, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.MASS_SCALER: (1.0, 'unitless'),
        }
        prob = self._make_prob(values)
        prob.run_model()

        expected_values = {
            Aircraft.HorizontalTail.MASS: 2285,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-11, rtol=1e-12)

    def test_bwb(self):
        """GASP BWB model."""
        values = {
            Aircraft.Design.GROSS_MASS: (150000.0, 'lbm'),
            Aircraft.HorizontalTail.MASS_COEFFICIENT: (0.124, 'unitless'),
            Aircraft.Fuselage.LENGTH: (71.5245514, 'ft'),
            Aircraft.HorizontalTail.SPAN: (0.04467601, 'ft'),
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (1.0, 'unitless'),
            Aircraft.HorizontalTail.TAPER_RATIO: (0.366, 'unitless'),
            Aircraft.HorizontalTail.AREA: (0.00117064, 'ft**2'),
            'min_dive_vel': (420, 'kn'),
            Aircraft.HorizontalTail.MOMENT_ARM: (29.6907417, 'ft'),
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (0.1, 'unitless'),
            Aircraft.HorizontalTail.ROOT_CHORD: (0.03836448, 'ft'),
            Aircraft.HorizontalTail.MASS_SCALER: (1.0, 'unitless'),
        }
        prob = self._make_prob(values)
        prob.run_model()

        expected_values = {
            Aircraft.HorizontalTail.MASS: 1.02401953,
        }
        tol = 1e-7
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

    def test_alt_gravity(self):
        values = {
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MASS_COEFFICIENT: (
                0.232,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.Fuselage.LENGTH: (129.4, 'ft'),  # bug fixed value and original value
            Aircraft.HorizontalTail.SPAN: (42.59, 'ft'),  # bug fixed value
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (
                1,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.TAPER_RATIO: (
                0.352,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.AREA: (381.8, 'ft**2'),  # bug fixed value
            'min_dive_vel': (420, 'kn'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MOMENT_ARM: (55.1, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (
                0.12,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.ROOT_CHORD: (13.261162230765065, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.MASS_SCALER: (1.0, 'unitless'),
        }
        prob = self._make_prob(values, gravity=8)
        prob.run_model()

        expected_values = {
            Aircraft.HorizontalTail.MASS: 2047.30587383,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)
        # gravity only scales the result linearly (mass_to_force_english); it isn't a
        # branch, so this doesn't need its own partials check beyond test_case1's


@use_tempdirs
class VerticalTailMassTestCase(unittest.TestCase):
    """Tests for the VerticalTailMass component against GASP output, including BWB."""

    def _make_prob(self, values, gravity=9.80665):
        options = AviaryValues()
        options.set_val(Mission.GRAVITY, val=gravity, units='m/s**2')

        prob = om.Problem()
        prob.model.add_subsystem('v_tail', VerticalTailMass(), promotes=['*'])

        for var_name, (val, units) in values.items():
            prob.model.set_input_defaults(var_name, val=val, units=units)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        values = {
            Aircraft.VerticalTail.TAPER_RATIO: (
                0.801,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.ASPECT_RATIO: (
                1.67,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.SWEEP: (0, 'rad'),  # bug fixed value
            Aircraft.VerticalTail.SPAN: (28.22, 'ft'),  # bug fixed value
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MASS_COEFFICIENT: (
                0.232,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.Fuselage.LENGTH: (129.4, 'ft'),  # bug fixed value and original value
            Aircraft.HorizontalTail.SPAN: (42.59, 'ft'),  # bug fixed value
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (
                1,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.TAPER_RATIO: (
                0.352,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.MASS_COEFFICIENT: (
                0.289,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.Wing.SPAN: (118.8, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.AREA: (381.8, 'ft**2'),  # bug fixed value
            'min_dive_vel': (420, 'kn'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MOMENT_ARM: (55.1, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (
                0.12,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.ROOT_CHORD: (13.261162230765065, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION: (
                0,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.AREA: (476.8, 'ft**2'),  # bug fixed value
            Aircraft.VerticalTail.MOMENT_ARM: (50.3, 'ft'),  # bug fixed value
            Aircraft.VerticalTail.THICKNESS_TO_CHORD: (
                0.12,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.ROOT_CHORD: (18.762708015981357, 'ft'),  # bug fixed value
            Aircraft.VerticalTail.MASS_SCALER: (1.0, 'unitless'),
        }
        prob = self._make_prob(values)
        prob.run_model()

        expected_values = {
            'loc_MAC_vtail': 0.44959578484694906,
            Aircraft.VerticalTail.MASS: 2312,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1e-11, rtol=1e-12)

    def test_bwb(self):
        """GASP BWB model."""
        values = {
            Aircraft.VerticalTail.TAPER_RATIO: (0.366, 'unitless'),
            Aircraft.VerticalTail.ASPECT_RATIO: (1.705, 'unitless'),
            Aircraft.VerticalTail.SWEEP: (0.0, 'rad'),
            Aircraft.VerticalTail.SPAN: (16.98084188, 'ft'),
            Aircraft.Design.GROSS_MASS: (150000.0, 'lbm'),
            Aircraft.HorizontalTail.MASS_COEFFICIENT: (0.124, 'unitless'),
            Aircraft.Fuselage.LENGTH: (71.5245514, 'ft'),
            Aircraft.HorizontalTail.SPAN: (0.04467601, 'ft'),
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (1.0, 'unitless'),
            Aircraft.HorizontalTail.TAPER_RATIO: (0.366, 'unitless'),
            Aircraft.VerticalTail.MASS_COEFFICIENT: (0.119, 'unitless'),
            Aircraft.Wing.SPAN: (146.38501094, 'ft'),
            Aircraft.HorizontalTail.AREA: (0.00117064, 'ft**2'),
            'min_dive_vel': (420, 'kn'),
            Aircraft.HorizontalTail.MOMENT_ARM: (29.6907417, 'ft'),
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (0.1, 'unitless'),
            Aircraft.HorizontalTail.ROOT_CHORD: (0.03836448, 'ft'),
            Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION: (0, 'unitless'),
            Aircraft.VerticalTail.AREA: (169.11964286, 'ft**2'),
            Aircraft.VerticalTail.MOMENT_ARM: (27.82191598, 'ft'),
            Aircraft.VerticalTail.THICKNESS_TO_CHORD: (0.1, 'unitless'),
            Aircraft.VerticalTail.ROOT_CHORD: (14.58190052, 'ft'),
            Aircraft.VerticalTail.MASS_SCALER: (1.0, 'unitless'),
        }
        prob = self._make_prob(values)
        prob.run_model()

        expected_values = {
            'loc_MAC_vtail': 0.97683077,
            Aircraft.VerticalTail.MASS: 864.17404177,
        }
        tol = 1e-7
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

    def test_alt_gravity(self):
        values = {
            Aircraft.VerticalTail.TAPER_RATIO: (
                0.801,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.ASPECT_RATIO: (
                1.67,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.SWEEP: (0, 'rad'),  # bug fixed value
            Aircraft.VerticalTail.SPAN: (28.22, 'ft'),  # bug fixed value
            Aircraft.Design.GROSS_MASS: (175400, 'lbm'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MASS_COEFFICIENT: (
                0.232,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.Fuselage.LENGTH: (129.4, 'ft'),  # bug fixed value and original value
            Aircraft.HorizontalTail.SPAN: (42.59, 'ft'),  # bug fixed value
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER: (
                1,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.TAPER_RATIO: (
                0.352,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.MASS_COEFFICIENT: (
                0.289,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.Wing.SPAN: (118.8, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.AREA: (381.8, 'ft**2'),  # bug fixed value
            'min_dive_vel': (420, 'kn'),  # bug fixed value and original value
            Aircraft.HorizontalTail.MOMENT_ARM: (55.1, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD: (
                0.12,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.HorizontalTail.ROOT_CHORD: (13.261162230765065, 'ft'),  # bug fixed value
            Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION: (
                0,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.AREA: (476.8, 'ft**2'),  # bug fixed value
            Aircraft.VerticalTail.MOMENT_ARM: (50.3, 'ft'),  # bug fixed value
            Aircraft.VerticalTail.THICKNESS_TO_CHORD: (
                0.12,
                'unitless',
            ),  # bug fixed value and original value
            Aircraft.VerticalTail.ROOT_CHORD: (18.762708015981357, 'ft'),  # bug fixed value
            Aircraft.VerticalTail.MASS_SCALER: (1.0, 'unitless'),
        }
        prob = self._make_prob(values, gravity=10)
        prob.run_model()

        expected_values = {
            'loc_MAC_vtail': 0.44959578484694906,
            Aircraft.VerticalTail.MASS: 2336.36180144,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)
        # gravity only scales the result linearly (mass_to_force_english); it isn't a
        # branch, so this doesn't need its own partials check beyond test_case1's


@use_tempdirs
class HighLiftTestCase(unittest.TestCase):
    """Tests for the HighLiftMass component against GASP output, including BWB."""

    def _make_prob(self, flap_type, num_flap_segments, sea_level_density, values):
        prob = om.Problem()

        aviary_options = AviaryValues()
        aviary_options.set_val(Aircraft.Wing.FLAP_TYPE, val=flap_type)
        aviary_options.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, val=num_flap_segments)
        aviary_options.set_val(Mission.SEA_LEVEL_DENSITY, sea_level_density, units='slug/ft**3')

        prob.model.add_subsystem('HL', HighLiftMass(), promotes=['*'])

        for var_name, (val, units) in values.items():
            prob.model.set_input_defaults(var_name, val=val, units=units)

        setup_model_options(prob, aviary_options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_case1(self):
        """This is a different configuration with turbofan_23k_1 test case."""
        values = {
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT: (1.9, 'unitless'),
            Aircraft.Wing.AREA: (1764.6, 'ft**2'),
            Aircraft.Wing.SLAT_CHORD_RATIO: (0.1, 'unitless'),
            Aircraft.Wing.FLAP_CHORD_RATIO: (0.25, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.346, 'unitless'),
            Aircraft.Wing.SLAT_SPAN_RATIO: (0.9, 'unitless'),
            Aircraft.Wing.FLAP_SPAN_RATIO: (0.88, 'unitless'),
            Aircraft.Design.WING_LOADING: (93.1, 'lbf/ft**2'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.11, 'unitless'),
            Aircraft.Wing.SPAN: (185.8, 'ft'),
            Aircraft.Fuselage.AVG_DIAMETER: (13.1, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (13.979, 'ft'),
            Mission.Landing.LIFT_COEFFICIENT_MAX: (2.3648, 'unitless'),
        }
        prob = self._make_prob(
            flap_type=FlapType.DOUBLE_SLOTTED,
            num_flap_segments=2,
            sea_level_density=0.0023769,
            values=values,
        )
        prob.run_model()

        expected_values = {
            Aircraft.Wing.HIGH_LIFT_MASS: 4829.6,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        data = prob.check_partials(out_stream=None, method='cs', show_only_incorrect=True)
        assert_check_partials(data, atol=5e-10, rtol=1e-12)

    def test_case2(self):
        values = {
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT: (1.9, 'unitless'),
            Aircraft.Wing.AREA: (1370.3125, 'ft**2'),
            Aircraft.Wing.SLAT_CHORD_RATIO: (0.15, 'unitless'),
            Aircraft.Wing.FLAP_CHORD_RATIO: (0.15, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.33, 'unitless'),
            Aircraft.Wing.SLAT_SPAN_RATIO: (0.9, 'unitless'),
            Aircraft.Wing.FLAP_SPAN_RATIO: (0.65, 'unitless'),
            Aircraft.Design.WING_LOADING: (128.0, 'lbf/ft**2'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.15, 'unitless'),
            Aircraft.Wing.SPAN: (117.81878299, 'ft'),
            Aircraft.Fuselage.AVG_DIAMETER: (13.1, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (17.48974356, 'ft'),
            Mission.Landing.LIFT_COEFFICIENT_MAX: (2.3648, 'unitless'),
        }
        prob = self._make_prob(
            flap_type=FlapType.DOUBLE_SLOTTED,
            num_flap_segments=2,
            sea_level_density=0.0023769,
            values=values,
        )
        prob.run_model()

        expected_values = {
            Aircraft.Wing.HIGH_LIFT_MASS: 2940.12660159,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)
        # same DOUBLE_SLOTTED branch as test_case1, just different numbers; no need
        # for its own partials check

    def test_bwb(self):
        values = {
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT: (1.0, 'unitless'),
            Aircraft.Wing.AREA: (2142.85714286, 'ft**2'),
            Aircraft.Wing.SLAT_CHORD_RATIO: (0.0001, 'unitless'),
            Aircraft.Wing.FLAP_CHORD_RATIO: (0.2, 'unitless'),
            Aircraft.Wing.TAPER_RATIO: (0.27444, 'unitless'),
            Aircraft.Wing.SLAT_SPAN_RATIO: (0.831687927, 'unitless'),
            Aircraft.Wing.FLAP_SPAN_RATIO: (0.61, 'unitless'),
            Aircraft.Design.WING_LOADING: (70.0, 'lbf/ft**2'),
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: (0.165, 'unitless'),
            Aircraft.Wing.SPAN: (146.38501094, 'ft'),
            Aircraft.Fuselage.AVG_DIAMETER: (38, 'ft'),
            Aircraft.Wing.CENTER_CHORD: (22.97244452, 'ft'),
            # 1.94034 is taken from .out file. In GASP, CLMAX is computed for different phases
            Mission.Landing.LIFT_COEFFICIENT_MAX: (1.94034, 'unitless'),
        }
        prob = self._make_prob(
            flap_type=4,
            num_flap_segments=2,
            sea_level_density=0.0023769,
            values=values,
        )
        prob.run_model()

        expected_values = {
            Aircraft.Wing.HIGH_LIFT_MASS: 1068.88854499,
            'flap_mass': 1068.46572125,
            'slat_mass': 0.42282374,
        }
        tol = 1e-7
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)
        # flap_type=4 is still DOUBLE_SLOTTED, the same branch test_case1 already
        # covers with a partials check


@use_tempdirs
class FixedMassGroupTestCase(unittest.TestCase):
    """Tests for the FixedMassGroup group against GASP output, including BWB."""

    def test_case1(self):
        """This is the large single aisle 1 V3 test case."""
        options = AviaryValues()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)

        options.set_val(Aircraft.Engine.NUM_ENGINES, [2], units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 2, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, False, units='unitless')
        options.set_val(Aircraft.Wing.FLAP_TYPE, FlapType.DOUBLE_SLOTTED, units='unitless')
        options.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, 2, units='unitless')
        options.set_val(Mission.SEA_LEVEL_DENSITY, 1.225, units='kg/m**3')
        options.set_val(Mission.GRAVITY, val=9.80665, units='m/s**2')

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FixedMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, val=200, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=118.8, units='ft')  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1392.1, units='ft**2'
        )  # bug fixed value and original value

        prob.model.set_input_defaults(
            Aircraft.Wing.SWEEP, val=25, units='deg'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Design.MAX_MACH, val=0.9, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0, units='rad'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.SPAN, val=28.22, units='ft'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.SPAN, val=42.59, units='ft'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=381.8, units='ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_ARM, val=55.1, units='ft'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ROOT_CHORD, val=13.261162230765065, units='ft'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION, val=0, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=476.8, units='ft**2'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_ARM, val=50.3, units='ft'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.ROOT_CHORD, val=18.762708015981357, units='ft'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.966, units='unitless'
        )  # bug fixed value and original value

        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.951, units='unitless'
        )  # bug fixed value
        prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, val=0.85, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Nacelle.AVG_DIAMETER, val=7.35, units='ft'
        )  # bug fixed value and original value

        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1)
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128)
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15)
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, val=17.48974)
        prob.model.set_input_defaults(Aircraft.Design.LANDING_TO_TAKEOFF_MASS_RATIO, val=1.0)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6384.35,
            Aircraft.Wing.MATERIAL_FACTOR: 1.2203729275531838,
            'c_strut_braced': 1,
            'c_gear_loc': 1,
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 0.95,
            'half_sweep': 0.3947081519145335,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 36000,
            'payload_mass_des': 36000,
            'payload_mass_max': 46040,
            'loc_MAC_vtail': 0.44959578484694906,
            Aircraft.HorizontalTail.MASS: 2285,
            Aircraft.VerticalTail.MASS: 2312,
            Aircraft.Wing.HIGH_LIFT_MASS: 4082.1,
            Aircraft.Controls.COCKPIT_CONTROL_MASS: 137.25749725,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS: 0.0,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 3807.92115815,
            Aircraft.Controls.MASS: 3945,
            Aircraft.LandingGear.TOTAL_MASS: 7511,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606,
            Aircraft.Engine.ADDITIONAL_MASS: 1765 / 2,
            'eng_comb_mass': 14370.8,
            'wing_mounted_mass': 24446.343040697346,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=3e-11, rtol=1e-12)

    def test_case2(self):
        options = AviaryValues()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, val=0, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14)
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=True, units='unitless')

        options.set_val(Aircraft.Engine.NUM_ENGINES, [2], units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, False, units='unitless')
        options.set_val(Aircraft.Wing.FLAP_TYPE, FlapType.DOUBLE_SLOTTED, units='unitless')
        options.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, 2, units='unitless')
        options.set_val(Mission.SEA_LEVEL_DENSITY, 1.225, units='kg/m**3')
        options.set_val(Mission.GRAVITY, val=9.80665, units='m/s**2')

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FixedMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, val=200, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.SPAN, val=117.8, units='ft'
        )  # original GASP value
        prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.GROSS_MASS, val=175400, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            'min_dive_vel', val=420, units='kn'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units='ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.SWEEP, val=25, units='deg'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Design.MAX_MACH, val=0.72, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=10 / 117.8, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0.1, units='rad'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.SPAN, val=28, units='ft'
        )  # original GASP value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units='ft'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.SPAN, val=42.25, units='ft'
        )  # original GASP value
        prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units='ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_ARM, val=54.7, units='ft'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ROOT_CHORD, val=13.16130387591471, units='ft'
        )  # original GASP value
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION, val=0, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units='ft**2'
        )  # original GASP value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_ARM, val=49.9, units='ft'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.VerticalTail.ROOT_CHORD, val=18.61267549773935, units='ft'
        )  # original GASP value
        prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # bug fixed value and original value

        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.ULTIMATE_LOAD_FACTOR, val=3.893, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Controls.CONTROL_MASS_INCREMENT, val=0, units='lbm'
        )  # bug fixed value and original value

        prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, val=0.85, units='unitless'
        )  # bug fixed value and original value

        prob.model.set_input_defaults('motor_power', val=200, units='kW')  # not actual GASP value
        prob.model.set_input_defaults('motor_voltage', val=50, units='V')  # not actual GASP value
        prob.model.set_input_defaults(
            'max_amp_per_wire', val=50, units='A'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'safety_factor', val=1.33, units='unitless'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            Aircraft.Electrical.HYBRID_CABLE_LENGTH, val=200, units='ft'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'wire_area', val=0.0015, units='ft**2'
        )  # not actual GASP value
        prob.model.set_input_defaults('rho_wire', val=1, units='lbm/ft**3')  # not actual GASP value
        prob.model.set_input_defaults('battery_energy', val=1, units='MJ')  # not actual GASP value
        prob.model.set_input_defaults('motor_eff', val=1, units='unitless')  # not actual GASP value
        prob.model.set_input_defaults(
            'inverter_eff', val=1, units='unitless'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'transmission_eff', val=1, units='unitless'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'battery_eff', val=1, units='unitless'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'rho_battery', val=200, units='kW*h/kg'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'motor_spec_mass', val=10, units='hp/lbm'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'inverter_spec_mass', val=10, units='kW/kg'
        )  # not actual GASP value
        prob.model.set_input_defaults(
            'TMS_spec_mass', val=0.125, units='lbm/kW'
        )  # electrified diff configuration value v3.6

        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500, units='lbf'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Nacelle.SURFACE_AREA, val=339.58, units='ft**2'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SCALER, val=1, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(
            Aircraft.Engine.Propeller.MASS, val=0, units='lbm'
        )  # bug fixed value and original value
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=13.1)
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, val=128)
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15)
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, val=17.48974)

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.LandingGear.MAIN_GEAR_MASS: 5963.6,
            'aug_mass': 228.51036478,
            Aircraft.Wing.MATERIAL_FACTOR: 1.2213063198183813,
            'c_strut_braced': 0.9928,
            'c_gear_loc': 1,
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 1,
            'half_sweep': 0.3947081519145335,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 36000,
            'payload_mass_des': 36000,
            'payload_mass_max': 46040,
            'loc_MAC_vtail': 1.799,
            Aircraft.HorizontalTail.MASS: 2275,
            Aircraft.VerticalTail.MASS: 2297,
            Aircraft.Wing.HIGH_LIFT_MASS: 4162.1,
            Aircraft.Controls.MASS: 3895,
            Aircraft.LandingGear.TOTAL_MASS: 7016,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 12606,
            Aircraft.Engine.ADDITIONAL_MASS: 1765 / 2,
            'eng_comb_mass': 14599.28196478,
            'wing_mounted_mass': 24027.6,
            'prop_mass_sum': 0,
        }
        tol = 5e-4
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)

        data = prob.check_partials(
            out_stream=None,
            method='cs',
            form='central',
        )
        assert_check_partials(data, atol=3e-10, rtol=1e-12)

    def test_bwb(self):
        """GASP BWB model."""
        options = AviaryValues()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=150, units='unitless')
        options.set_val(Settings.VERBOSITY, 0)
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.04373)
        options.set_val(Mission.SEA_LEVEL_DENSITY, 0.0023769, units='slug/ft**3')

        options.set_val(Aircraft.Engine.NUM_ENGINES, [2], units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 2, units='unitless')
        options.set_val(Aircraft.Propulsion.TOTAL_NUM_ENGINES, 2, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, False, units='unitless')
        options.set_val(Aircraft.Wing.FLAP_TYPE, FlapType.DOUBLE_SLOTTED, units='unitless')
        options.set_val(Aircraft.Wing.NUM_FLAP_SEGMENTS, 2, units='unitless')
        options.set_val(Mission.GRAVITY, val=9.80665, units='m/s**2')

        prob = om.Problem()
        prob.model.add_subsystem(
            'group',
            FixedMassGroup(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            Aircraft.CrewPayload.MASS_PER_PASSENGER_WITH_BAGS, val=225, units='lbm'
        )
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.VerticalTail.MASS_SCALER, val=1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501094, units='ft')
        prob.model.set_input_defaults(Aircraft.Design.GROSS_MASS, 150000, units='lbm')
        prob.model.set_input_defaults('min_dive_vel', 420, units='kn')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85714286, units='ft**2')

        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.MAX_MACH, 0.9, units='unitless')
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
            Aircraft.HorizontalTail.VERTICAL_TAIL_MOUNT_LOCATION, 0, units='unitless'
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
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, 0, units='lbm'
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
            Aircraft.LandingGear.MAIN_GEAR_MASS_FRACTION, 0.85, units='unitless'
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
        prob.model.set_input_defaults(Aircraft.Design.WING_LOADING, 70.0)
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165)
        prob.model.set_input_defaults(Aircraft.Wing.CENTER_CHORD, 22.97244452)
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_SPAN_RATIO, 0.61, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        expected_values = {
            Aircraft.LandingGear.TOTAL_MASS: 7800.0,
            Aircraft.LandingGear.MAIN_GEAR_MASS: 6630.0,
            Aircraft.Wing.MATERIAL_FACTOR: 1.19461189,
            'c_strut_braced': 1,
            'c_gear_loc': 0.95,
            Aircraft.Wing.ENGINE_POSITION_MASS_SCALER: 0.95,
            'half_sweep': 0.47984874,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS: 33750.0,
            'payload_mass_des': 33750,
            'payload_mass_max': 48750,
            'loc_MAC_vtail': 0.97683077,
            Aircraft.HorizontalTail.MASS: 1.02401953,
            Aircraft.VerticalTail.MASS: 864.17404177,
            Aircraft.Wing.HIGH_LIFT_MASS: 1068.88854499,
            Aircraft.Controls.MASS: 2114.98158947,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS: 7005.15475443,
            Aircraft.Nacelle.MASS: 549.8807447,
            Aircraft.Propulsion.TOTAL_ENGINE_POD_MASS: 2230.13208284,
            Aircraft.Engine.ADDITIONAL_MASS: 153.16770871,
            'pylon_mass': 565.18529673,
            'eng_comb_mass': 7311.49017184,
            'wing_mounted_mass': 0.0,
            Aircraft.Wing.SURFACE_CONTROL_MASS: 1986.25111783,
        }
        tol = 1e-7
        for var_name, expected_val in expected_values.items():
            with self.subTest(var=var_name):
                assert_near_equal(prob[var_name], expected_val, tol)
        # same has_hybrid_system=False group structure, smooth_mass_discontinuities,
        # and flap_type as test_case1; test_case2 alone covers the hybrid-system branch


if __name__ == '__main__':
    unittest.main()
