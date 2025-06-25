import json
import os
import unittest

import numpy as np
import openmdao.api as om
import pandas as pd
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.gaspaero import (
    CruiseAero,
    LowSpeedAero,
    FormFactorAndSIWB,
    GroundEffect,
    LiftCoeff,
    LiftCoeffClean,
    BWBBodyLiftCurveSlope,
    BWBFormFactorAndSIWB,
    BWBLiftCoeff,
    BWBLiftCoeffClean,
    BWBAeroSetup,
    UFac,
    Xlifts,
    AeroGeom,
    WingTailRatios,
    DragCoefClean,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

here = os.path.abspath(os.path.dirname(__file__))
cruise_data = pd.read_csv(os.path.join(here, 'data', 'aero_data_cruise.csv'))
ground_data = pd.read_csv(os.path.join(here, 'data', 'aero_data_ground.csv'))
setup_data = json.load(open(os.path.join(here, 'data', 'aero_data_setup.json')))


class GASPAeroTest(unittest.TestCase):
    """
    Test overall pre-mission and mission aero systems in cruise and near-ground flight.
    Note: The case output_alpha=True is not tested.
    """

    cruise_tol = 1.5e-3
    ground_tol = 0.5e-3
    aviary_options = AviaryValues()
    aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

    def test_cruise(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'aero',
            CruiseAero(num_nodes=2, input_atmos=True),
            promotes=['*'],
        )

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        _init_geom(prob)

        # extra params needed for cruise aero
        prob.set_val(Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, setup_data['clmwfu'])
        prob.set_val(Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT, setup_data['scfac'])

        for i, row in cruise_data.iterrows():
            alt = row['alt']
            mach = row['mach']
            alpha = row['alpha']

            with self.subTest(alt=alt, mach=mach, alpha=alpha):
                # prob.set_val(Dynamic.Mission.ALTITUDE, alt)
                prob.set_val(Dynamic.Atmosphere.MACH, mach)
                prob.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, alpha)
                prob.set_val(Dynamic.Atmosphere.SPEED_OF_SOUND, row['sos'])
                prob.set_val(Dynamic.Atmosphere.KINEMATIC_VISCOSITY, row['nu'])

                prob.run_model()

                assert_near_equal(prob['CL'][0], row['cl'], tolerance=self.cruise_tol)
                assert_near_equal(prob['CD'][0], row['cd'], tolerance=self.cruise_tol)

                # some partials are computed using "cs" method. So, use "fd" here. computation is not as good as "cs".
                partial_data = prob.check_partials(method='fd', out_stream=None)
                assert_check_partials(partial_data, atol=0.8, rtol=0.002)

    def test_ground(self):
        prob = om.Problem()
        prob.model.add_subsystem(
            'aero',
            LowSpeedAero(
                num_nodes=2,
                input_atmos=True,
            ),
            promotes=['*'],
        )

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        _init_geom(prob)

        # extra params needed for ground aero
        # leave time ramp inputs as defaults, gear/flaps extended and stay
        prob.set_val(Aircraft.Wing.HEIGHT, 8.0)  # not defined in standalone aero
        prob.set_val('airport_alt', 0.0)  # not defined in standalone aero
        prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, setup_data['cfoc'])
        prob.set_val(Mission.Design.GROSS_MASS, setup_data['wgto'])

        for i, row in ground_data.iterrows():
            ilift = row['ilift']  # 2: takeoff, 3: landing
            alt = row['alt']
            mach = row['mach']
            alpha = row['alpha']

            with self.subTest(ilift=ilift, alt=alt, mach=mach, alpha=alpha):
                prob.set_val(Dynamic.Atmosphere.MACH, mach)
                prob.set_val(Dynamic.Mission.ALTITUDE, alt)
                prob.set_val(Dynamic.Vehicle.ANGLE_OF_ATTACK, alpha)
                prob.set_val(Dynamic.Atmosphere.SPEED_OF_SOUND, row['sos'])
                prob.set_val(Dynamic.Atmosphere.KINEMATIC_VISCOSITY, row['nu'])

                # note we're just letting the time ramps for flaps/gear default to the
                # takeoff config such that the default times correspond to full flap and
                # gear increments
                if row['ilift'] == 2:
                    # takeoff flaps config
                    prob.set_val('flap_defl', setup_data['delfto'])
                    prob.set_val('CL_max_flaps', setup_data['clmwto'])
                    prob.set_val('dCL_flaps_model', setup_data['dclto'])
                    prob.set_val('dCD_flaps_model', setup_data['dcdto'])
                    prob.set_val('aero_ramps.flap_factor:final_val', 1.0)
                    prob.set_val('aero_ramps.gear_factor:final_val', 1.0)
                else:
                    # landing flaps config
                    prob.set_val('flap_defl', setup_data['delfld'])
                    prob.set_val('CL_max_flaps', setup_data['clmwld'])
                    prob.set_val('dCL_flaps_model', setup_data['dclld'])
                    prob.set_val('dCD_flaps_model', setup_data['dcdld'])

                prob.run_model()

                assert_near_equal(prob['CL'][0], row['cl'], tolerance=self.ground_tol)
                assert_near_equal(prob['CD'][0], row['cd'], tolerance=self.ground_tol)

                partial_data = prob.check_partials(method='fd', out_stream=None)
                assert_check_partials(partial_data, atol=4.5, rtol=5e-3)

    def ttest_ground_alpha_out(self):
        # Test that drag output matches between both CL computation methods
        prob = om.Problem()
        prob.model.add_subsystem(
            'lift_from_aoa',
            LowSpeedAero(),
            promotes_inputs=['*', (Dynamic.Vehicle.ANGLE_OF_ATTACK, 'alpha_in')],
            promotes_outputs=[(Dynamic.Vehicle.LIFT, 'lift_req')],
        )

        prob.model.add_subsystem(
            'lift_required',
            LowSpeedAero(lift_required=True),
            promotes_inputs=['*', 'lift_req'],
        )

        setup_model_options(prob, AviaryValues({Aircraft.Engine.NUM_ENGINES: ([2], 'unitless')}))

        prob.setup(check=False, force_alloc_complex=True)

        _init_geom(prob)

        # extra params needed for ground aero
        # leave time ramp inputs as defaults, gear/flaps extended and stay
        prob.set_val(Aircraft.Wing.HEIGHT, 8.0)  # not defined in standalone aero
        prob.set_val('airport_alt', 0.0)  # not defined in standalone aero
        prob.set_val(Aircraft.Wing.FLAP_CHORD_RATIO, setup_data['cfoc'])
        prob.set_val(Mission.Design.GROSS_MASS, setup_data['wgto'])

        prob.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, 1)
        prob.set_val(Dynamic.Atmosphere.MACH, 0.1)
        prob.set_val(Dynamic.Mission.ALTITUDE, 10)
        prob.set_val('alpha_in', 5)
        prob.run_model()

        assert_near_equal(prob['lift_from_aoa.drag'], prob['lift_required.drag'], tolerance=1e-6)

        partial_data = prob.check_partials(method='fd', out_stream=None)
        assert_check_partials(partial_data, atol=0.02, rtol=1e-4)


def _init_geom(prob):
    """Initialize user inputs and geometry/sizing data."""
    prob.set_val('interference_independent_of_shielded_area', 1.89927266)
    prob.set_val('drag_loss_due_to_shielded_wing_area', 68.02065834)
    # i.e. common auto IVC vars for the setup + cruise and ground aero models
    prob.set_val(Aircraft.Wing.AREA, setup_data['sw'])
    prob.set_val(Aircraft.Wing.SPAN, setup_data['b'])
    prob.set_val(Aircraft.Wing.AVERAGE_CHORD, setup_data['cbarw'])
    prob.set_val(Aircraft.Wing.TAPER_RATIO, setup_data['slm'])
    prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, setup_data['tcr'])
    prob.set_val(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, setup_data['hwing'])
    prob.set_val(Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, setup_data['sah'])
    prob.set_val(Aircraft.HorizontalTail.SPAN, setup_data['bht'])
    prob.set_val(Aircraft.VerticalTail.SPAN, setup_data['bvt'])
    prob.set_val(Aircraft.HorizontalTail.AREA, setup_data['sht'])
    prob.set_val(Aircraft.HorizontalTail.AVERAGE_CHORD, setup_data['cbarht'])
    prob.set_val(Aircraft.Fuselage.AVG_DIAMETER, setup_data['swf'])
    # ground & cruise, mission: mach
    prob.set_val(Aircraft.Design.STATIC_MARGIN, setup_data['stmarg'])
    prob.set_val(Aircraft.Design.CG_DELTA, setup_data['delcg'])
    prob.set_val(Aircraft.Wing.ASPECT_RATIO, setup_data['ar'])
    prob.set_val(Aircraft.Wing.SWEEP, setup_data['dlmc4'])
    prob.set_val(Aircraft.HorizontalTail.SWEEP, setup_data['dwpqch'])
    prob.set_val(Aircraft.HorizontalTail.MOMENT_RATIO, setup_data['coelth'])
    # ground & cruise, mission: alt
    prob.set_val(Aircraft.Wing.FORM_FACTOR, setup_data['ckw'])
    prob.set_val(Aircraft.Fuselage.FORM_FACTOR, setup_data['ckf'])
    prob.set_val(Aircraft.Nacelle.FORM_FACTOR, setup_data['ckn'])
    prob.set_val(Aircraft.VerticalTail.FORM_FACTOR, setup_data['ckvt'])
    prob.set_val(Aircraft.HorizontalTail.FORM_FACTOR, setup_data['ckht'])
    prob.set_val(Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, setup_data['cki'])
    prob.set_val(Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR, setup_data['ckstrt'])
    prob.set_val(Aircraft.Design.DRAG_COEFFICIENT_INCREMENT, setup_data['delcd'])
    prob.set_val(Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT, setup_data['delfe'])
    prob.set_val(Aircraft.Wing.MIN_PRESSURE_LOCATION, setup_data['xcps'])
    prob.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.4)  # overridden in standalone code
    prob.set_val(Aircraft.Strut.AREA_RATIO, setup_data['sstqsw'])
    prob.set_val(Aircraft.VerticalTail.AVERAGE_CHORD, setup_data['cbarvt'])
    prob.set_val(Aircraft.Fuselage.LENGTH, setup_data['elf'])
    prob.set_val(Aircraft.Nacelle.AVG_LENGTH, setup_data['eln'])
    prob.set_val(Aircraft.Fuselage.WETTED_AREA, setup_data['sf'])
    prob.set_val(Aircraft.Nacelle.SURFACE_AREA, setup_data['sn'] / setup_data['enp'])
    prob.set_val(Aircraft.VerticalTail.AREA, setup_data['svt'])
    prob.set_val(Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, setup_data['tc'])
    prob.set_val(Aircraft.Strut.CHORD, 0)  # not defined in standalone aero
    # ground & cruise, mission: alpha
    prob.set_val(Aircraft.Wing.ZERO_LIFT_ANGLE, setup_data['alphl0'])
    # ground & cruise, config-specific: CL_max_flaps
    # ground: wing_height
    # ground: airport_alt
    # ground: flap_defl
    # ground: flap_chord_ratio
    # ground: CL_max_flaps
    # ground: dCL_flaps_model
    # ground: t_init_flaps
    # ground: t_curr
    # ground: dt_flaps
    # ground: gross_mass_initial
    # ground: dCD_flaps_model
    # ground: t_init_gear
    # ground: dt_gear
    # ground & cruise, mission: q


class XLiftsTest(unittest.TestCase):
    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

        prob = om.Problem()
        prob.model.add_subsystem(
            'xlifts',
            Xlifts(num_nodes=2),
            promotes=['*'],
        )

        # Xlifts
        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, [0.8, 0.8], units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.STATIC_MARGIN, 0.05, units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.CG_DELTA, 0.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.SWEEP, 45.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, 0.5463, units='unitless'
        )
        prob.model.set_input_defaults('sbar', 0.001, units='unitless')
        prob.model.set_input_defaults('cbar', 0.00173, units='unitless')
        prob.model.set_input_defaults('hbar', 0.001, units='unitless')
        prob.model.set_input_defaults('bbar', 0.000305, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-06
        assert_near_equal(prob['lift_curve_slope'], [5.94800206, 5.94800206], tol)
        assert_near_equal(prob['lift_ratio'], [-0.140812203, -0.140812203], tol)

        partial_data = prob.check_partials(method='fd', out_stream=None)
        assert_check_partials(partial_data, atol=1e-4, rtol=1e-3)


class LiftCoeffTest(unittest.TestCase):
    """Test partials of LiftCoeff"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'lift_coeff',
            LiftCoeff(num_nodes=2),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, [-2.0, -2.0], units='deg')
        prob.model.set_input_defaults('lift_curve_slope', [4.876, 4.876], units='unitless')
        prob.model.set_input_defaults('lift_ratio', [0.5, 0.5], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, -1.2, units='deg')
        prob.model.set_input_defaults('CL_max_flaps', 2.188, units='unitless')
        prob.model.set_input_defaults('dCL_flaps_model', 0.418, units='unitless')
        prob.model.set_input_defaults('kclge', [1.15, 1.15], units='unitless')
        prob.model.set_input_defaults('flap_factor', [1.0, 1.0], units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class LiftCoeffCleanTest(unittest.TestCase):
    """Test partials of LiftCoeffClean"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'lift_coeff',
            LiftCoeffClean(num_nodes=2, output_alpha=False),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, [-2.0, -2.0], units='deg')
        prob.model.set_input_defaults('lift_curve_slope', [5.975, 5.975], units='unitless')
        prob.model.set_input_defaults('lift_ratio', [0.0357, 0.0357], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, -1.2, units='deg')
        prob.model.set_input_defaults(
            Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, 1.8885, units='unitless'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob['CL'], [-0.08640507, -0.08640507], tol)
        assert_near_equal(prob['alpha_stall'], [16.90930203, 16.90930203], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)

    def test_case2(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'lift_coeff',
            LiftCoeffClean(num_nodes=2, output_alpha=True),
            promotes=['*'],
        )

        prob.model.set_input_defaults('CL', [-0.08640507, -0.08640507], units='unitless')
        prob.model.set_input_defaults('lift_curve_slope', [5.975, 5.975], units='unitless')
        prob.model.set_input_defaults('lift_ratio', [0.0357, 0.0357], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, -1.2, units='deg')
        prob.model.set_input_defaults(
            Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, 1.8885, units='unitless'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob[Dynamic.Vehicle.ANGLE_OF_ATTACK], [-1.99999997, -1.99999997], tol)
        assert_near_equal(prob['alpha_stall'], [16.90930203, 16.90930203], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class UFacTest(unittest.TestCase):
    """Test UFac computation"""

    def test_case1(self):
        """
        aircraft of tube and wing type (not used in AeroSetup)
        """
        prob = om.Problem()

        prob.model.add_subsystem(
            'ufac',
            UFac(num_nodes=2),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            'lift_ratio', val=[-0.140812203, -0.140812203], units='unitless'
        )
        prob.model.set_input_defaults('bbar_alt', val=1.0, units='unitless')
        prob.model.set_input_defaults('sigma', val=1.0, units='unitless')
        prob.model.set_input_defaults('sigstr', val=1.0, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['ufac'], [1.0, 1.0], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)

    def test_case2(self):
        """
        BWB case with smooth derivative (used in BWBAeroSetup)
        """
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')

        prob = om.Problem()

        prob.model.add_subsystem(
            'ufac',
            UFac(num_nodes=2, smooth_ufac=True, mu=150.0),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            'lift_ratio', val=[-0.140812203, -0.140812203], units='unitless'
        )
        prob.model.set_input_defaults('bbar_alt', val=1.0, units='unitless')
        prob.model.set_input_defaults('sigma', val=1.0, units='unitless')
        prob.model.set_input_defaults('sigstr', val=1.0, units='unitless')

        setup_model_options(prob, options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['ufac'], [0.97484503, 0.97484503], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)

    def test_case3(self):
        """
        BWB case without smoothness, hence exactly the same as GASP (used in BWBAeroSetup)
        """
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')

        prob = om.Problem()

        prob.model.add_subsystem(
            'ufac',
            UFac(num_nodes=2, smooth_ufac=False),
            promotes=['*'],
        )

        prob.model.set_input_defaults(
            'lift_ratio', val=[-0.140812203, -0.140812203], units='unitless'
        )
        prob.model.set_input_defaults('bbar_alt', val=1.0, units='unitless')
        prob.model.set_input_defaults('sigma', val=1.0, units='unitless')
        prob.model.set_input_defaults('sigstr', val=1.0, units='unitless')

        setup_model_options(prob, options)
        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['ufac'], [0.975, 0.975], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class FormFactorAndSIWBTest(unittest.TestCase):
    """Test fuselage form factor computation and SIWB computation"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'form_factor',
            FormFactorAndSIWB(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, val=19.365, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=146.38501, units='ft')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['body_form_factor'], 1.35024726, tol)
        assert_near_equal(prob['siwb'], 0.964972794, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class BWBFormFactorAndSIWBTest(unittest.TestCase):
    """Test fuselage form factor computation and SIWB computation"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'form_factor',
            BWBFormFactorAndSIWB(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Fuselage.HYDRAULIC_DIAMETER, val=19.365, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=146.38501, units='ft')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['body_form_factor'], 1.35024726, tol)
        assert_near_equal(prob['siwb'], 0.964972794, tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class GroundEffectTest(unittest.TestCase):
    """Test fuselage form factor computation and SIWB computation"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'kclge',
            GroundEffect(num_nodes=2),
            promotes=['*'],
        )

        # mission inputs
        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, [-2.0, -2.0], units='deg')
        prob.model.set_input_defaults(Dynamic.Mission.ALTITUDE, [0.0, 0.0], units='ft')
        prob.model.set_input_defaults(
            'lift_curve_slope', [4.87625889, 4.87625889], units='unitless'
        )
        # user inputs
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, -1.2, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.13, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.HEIGHT, 8, units='ft')
        prob.model.set_input_defaults('airport_alt', 0.0, units='ft')
        prob.model.set_input_defaults('flap_defl', 10.0, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, 0.3, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.33, units='unitless')
        # from flaps
        prob.model.set_input_defaults('dCL_flaps_model', 0.4182, units='unitless')
        # from sizing
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, 12.61453152, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8187662, units='ft')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-7
        assert_near_equal(prob['kclge'], [1.15131091, 1.15131091], tol)


class BWBBodyLiftCurveSlopeTest(unittest.TestCase):
    """Body lift curve slope test for BWB"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'body_lift_curve_slope',
            BWBBodyLiftCurveSlope(num_nodes=3),
            promotes=['*'],
        )
        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, [0.2, 0.198, 0.8], units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_CURVE_SLOPE_MACH0, 1.8265, units='1/rad'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-6
        assert_near_equal(prob['body_lift_curve_slope'], [1.86416388, 1.86339139, 3.04416667], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class BWBLiftCoeffTest(unittest.TestCase):
    """Body lift curve slope test for BWB"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'lift_coeff',
            BWBLiftCoeff(num_nodes=2),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, [-2.0, -2.0], units='deg')
        prob.model.set_input_defaults('lift_curve_slope', [4.876, 4.876], units='unitless')
        prob.model.set_input_defaults(
            'body_lift_curve_slope', [3.04416704, 3.04416704], units='unitless'
        )
        prob.model.set_input_defaults('lift_ratio', [0.5, 0.5], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, -1.2, units='deg')
        prob.model.set_input_defaults('CL_max_flaps', 2.188, units='unitless')
        prob.model.set_input_defaults('dCL_flaps_model', 0.418, units='unitless')
        prob.model.set_input_defaults('kclge', [1.15, 1.15], units='unitless')
        prob.model.set_input_defaults('flap_factor', [0.9, 0.9], units='unitless')

        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.11353, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.PLANFORM_AREA, 1943.76587, units='ft**2')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-6
        assert_near_equal(prob['CL'], [0.23762299, 0.23762299], tol)
        assert_near_equal(prob['alpha_stall'], [16.88565997, 16.88565997], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)

    def test_case2(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'lift_coeff',
            BWBLiftCoeff(num_nodes=2),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, [21.4, 21.4], units='deg')
        prob.model.set_input_defaults('lift_curve_slope', [4.6386786, 4.6386786], units='unitless')
        prob.model.set_input_defaults(
            'body_lift_curve_slope', [1.85912228, 1.85912228], units='unitless'
        )
        prob.model.set_input_defaults('lift_ratio', [-0.140812159, -0.140812159], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, 0.0, units='deg')
        prob.model.set_input_defaults('CL_max_flaps', 2.12823462, units='unitless')
        prob.model.set_input_defaults('dCL_flaps_model', 0.390711457, units='unitless')
        prob.model.set_input_defaults('kclge', [1.001, 1.001], units='unitless')
        prob.model.set_input_defaults('flap_factor', [0.9, 0.9], units='unitless')

        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.11353, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.PLANFORM_AREA, 1943.76587, units='ft**2')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-6
        assert_near_equal(prob['CL'], [1.76135088, 1.761350889], tol)
        assert_near_equal(prob['alpha_stall'], [21.44000465, 21.44000465], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class BWBLiftCoeffCleanTest(unittest.TestCase):
    """Body lift curve slope test for BWB"""

    def test_case1(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'lift_coeff_clean',
            BWBLiftCoeffClean(num_nodes=2, output_alpha=True),
            promotes=['*'],
        )
        prob.model.set_input_defaults('CL', val=[0.416944444, 0.416944444], units='unitless')
        prob.model.set_input_defaults('lift_curve_slope', [5.9489522, 5.9489522], units='unitless')
        prob.model.set_input_defaults(
            'body_lift_curve_slope', [3.04416704, 3.04416704], units='unitless'
        )
        prob.model.set_input_defaults('lift_ratio', [-0.140812159, -0.140812159], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, 0, units='deg')
        prob.model.set_input_defaults(
            Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, 1.53789318, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.11353, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.PLANFORM_AREA, 1943.76587, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-6
        assert_near_equal(prob['mod_lift_curve_slope'], [6.51504303212, 6.51504303212], tol)
        assert_near_equal(prob[Dynamic.Vehicle.ANGLE_OF_ATTACK], [3.66676903, 3.66676903], tol)
        assert_near_equal(prob['alpha_stall'], [14.8118105, 14.8118105], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)

    def test_case2(self):
        prob = om.Problem()

        prob.model.add_subsystem(
            'body_lift_curve_slope',
            BWBLiftCoeffClean(num_nodes=2, output_alpha=False),
            promotes=['*'],
        )
        prob.model.set_input_defaults(Dynamic.Vehicle.ANGLE_OF_ATTACK, val=[1.5, 1.5], units='deg')
        prob.model.set_input_defaults('lift_curve_slope', [5.9489522, 5.9489522], units='unitless')
        prob.model.set_input_defaults(
            'body_lift_curve_slope', [3.04416704, 3.04416704], units='unitless'
        )
        prob.model.set_input_defaults('lift_ratio', [-0.140812159, -0.140812159], units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ZERO_LIFT_ANGLE, 0, units='deg')
        prob.model.set_input_defaults(
            Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP, 1.53789318, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.EXPOSED_AREA, 1352.11353, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.PLANFORM_AREA, 1943.76587, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Fuselage.LIFT_COEFFICENT_RATIO_BODY_TO_WING, 0.35, units='unitless'
        )

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-6
        assert_near_equal(prob['CL'], [0.17056343, 0.17056343], tol)
        assert_near_equal(prob['alpha_stall'], [14.81181653, 14.81181653], tol)

        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-11, rtol=1e-11)


class BWBCruiseAeroTest(unittest.TestCase):
    def test_cruise(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))

        prob = om.Problem()
        prob.model.add_subsystem(
            'aero',
            CruiseAero(num_nodes=2, input_atmos=True),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        # prob.run_model()


class BWBAeroSetupTest(unittest.TestCase):
    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))
        options.set_val(Aircraft.Wing.HAS_STRUT, False)

        prob = om.Problem()
        prob.model.add_subsystem(
            'aero_setup',
            BWBAeroSetup(num_nodes=2, input_atmos=True),
            promotes=['*'],
        )

        # WingTailRatios
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, 16.2200522, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.SPAN, 0.04467601, units='ft')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SPAN, 16.98084188, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.00117064, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AVERAGE_CHORD, 0.0280845, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')

        # Xlifts
        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, [0.8, 0.8], units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.STATIC_MARGIN, 0.05, units='unitless')
        prob.model.set_input_defaults(Aircraft.Design.CG_DELTA, 0.25, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.SWEEP, 45.0, units='deg')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, 0.5463, units='unitless'
        )

        # BWBFormFactorAndSIWB
        prob.model.set_input_defaults(Aircraft.Fuselage.HYDRAULIC_DIAMETER, 19.3650932, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')

        # AeroGeom
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, [993.11760441, 993.11760441], units='ft/s'
        )
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.KINEMATIC_VISCOSITY, [0.00034882, 0.00034882], units='ft**2/s'
        )
        prob.model.set_input_defaults(Aircraft.Wing.FORM_FACTOR, 2.563, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.FORM_FACTOR, 1.35, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.FORM_FACTOR, 1.2, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.FORM_FACTOR, 2.361, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.FORM_FACTOR, 2.413, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.DRAG_COEFFICIENT_INCREMENT, 0.00025, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT, 0.25, units='ft**2'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MIN_PRESSURE_LOCATION, 0.275, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.325, units='unitless')
        prob.model.set_input_defaults(Aircraft.Strut.AREA_RATIO, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AVERAGE_CHORD, 10.67, units='ft')
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_LENGTH, 18.11, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, 4573.8833, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.SURFACE_AREA, 411.93, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.1, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13596576, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Strut.CHORD, 0.0, units='ft')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob['hbar'], 0, tol)
        assert_near_equal(prob['bbar'], 0.0003052, tol)
        assert_near_equal(prob['sbar'], 0, tol)
        assert_near_equal(prob['cbar'], 0.00173147, tol)

        tol = 1e-5
        assert_near_equal(prob['lift_curve_slope'], [5.94845697, 5.94845697], tol)
        assert_near_equal(prob['lift_ratio'], [-0.140812203, -0.140812203], tol)

        assert_near_equal(prob['ufac'], [0.975, 0.975], tol)

        tol = 1e-6
        assert_near_equal(prob['cf'], [0.00283643, 0.00283643], tol)
        assert_near_equal(prob['SA1'], [0.80832432, 0.80832432], tol)
        assert_near_equal(prob['SA2'], [-0.13650645, -0.13650645], tol)
        assert_near_equal(prob['SA3'], [0.03398855, 0.03398855], tol)
        assert_near_equal(prob['SA4'], [0.10197432, 0.10197432], tol)
        assert_near_equal(prob['SA5'], [0.00962771, 0.00962771], tol)
        assert_near_equal(prob['SA6'], [2.09276756, 2.09276756], tol)
        assert_near_equal(prob['SA7'], [0.04049829, 0.04049829], tol)

        assert_near_equal(prob['body_form_factor'], 1.35024721, tol)
        assert_near_equal(prob['siwb'], 0.96497277, tol)


class WingTailRatiosTest(unittest.TestCase):
    def test_case1(self):
        """BWB data"""
        prob = om.Problem()
        prob.model.add_subsystem(
            'wing_tail_ratio',
            WingTailRatios(),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 146.38501, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, 16.2200522, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.165, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.VERTICAL_MOUNT_LOCATION, 0.5, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, 0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.HorizontalTail.SPAN, 0.04467601, units='ft')
        prob.model.set_input_defaults(Aircraft.VerticalTail.SPAN, 16.98084188, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.00117064, units='ft**2')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AVERAGE_CHORD, 0.0280845, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.AVG_DIAMETER, 38.0, units='ft')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob['hbar'], 0, tol)
        assert_near_equal(prob['bbar'], 0.0003052, tol)
        assert_near_equal(prob['sbar'], 0, tol)
        assert_near_equal(prob['cbar'], 0.00173147, tol)
        assert_near_equal(prob['bbar_alt'], 1.0, tol)


class AeroGeomTest(unittest.TestCase):
    def test_case1(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Engine.NUM_ENGINES, np.array([2]))
        options.set_val(Aircraft.Wing.HAS_STRUT, False)

        prob = om.Problem()
        prob.model.add_subsystem(
            'aero_geom',
            AeroGeom(num_nodes=2),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, [0.8, 0.8], units='unitless')
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.SPEED_OF_SOUND, [993.11760441, 993.11760441], units='ft/s'
        )
        prob.model.set_input_defaults(
            Dynamic.Atmosphere.KINEMATIC_VISCOSITY, [0.00034882, 0.00034882], units='ft**2/s'
        )
        prob.model.set_input_defaults(Aircraft.Wing.FORM_FACTOR, 2.563, units='unitless')
        prob.model.set_input_defaults(Aircraft.Fuselage.FORM_FACTOR, 1.35, units='unitless')
        prob.model.set_input_defaults(Aircraft.Nacelle.FORM_FACTOR, 1.2, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.FORM_FACTOR, 2.361, units='unitless')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.FORM_FACTOR, 2.413, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.DRAG_COEFFICIENT_INCREMENT, 0.00025, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT, 0.25, units='ft**2'
        )
        prob.model.set_input_defaults(Aircraft.Wing.MIN_PRESSURE_LOCATION, 0.275, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.325, units='unitless')
        prob.model.set_input_defaults(Aircraft.Strut.AREA_RATIO, 0.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AVERAGE_CHORD, 10.67, units='ft')
        prob.model.set_input_defaults(Aircraft.Nacelle.AVG_LENGTH, 18.11, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.WETTED_AREA, 4573.8833, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Nacelle.SURFACE_AREA, 411.93, units='ft**2')
        prob.model.set_input_defaults(Aircraft.VerticalTail.AREA, 169.1, units='ft**2')
        prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED, 0.13596576, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 10.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 30.0, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, 0.27444, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, 16.2200522, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AVERAGE_CHORD, 0.0280845, units='ft')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, 71.5245514, units='ft')
        prob.model.set_input_defaults(Aircraft.HorizontalTail.AREA, 0.00117064, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.AREA, 2142.85718, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Strut.CHORD, 0.0, units='ft')
        prob.model.set_input_defaults('ufac', [0.975, 0.975], units='unitless')
        prob.model.set_input_defaults('siwb', 0.96497277, units='unitless')
        prob.model.set_input_defaults('body_form_factor', 1.35024721, units='unitless')

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-5
        # GASP uses different skin friction parameters for different areas.
        # That explains the differences between GASP and Aviary
        assert_near_equal(prob['cf'], [0.00283643, 0.00283643], tol)
        assert_near_equal(prob['SA1'], [0.80832432, 0.80832432], tol)
        assert_near_equal(prob['SA2'], [-0.13650645, -0.13650645], tol)
        assert_near_equal(prob['SA3'], [0.03398855, 0.03398855], tol)
        assert_near_equal(prob['SA4'], [0.10197432, 0.10197432], tol)
        assert_near_equal(prob['SA5'], [0.00962771, 0.00962771], tol)
        assert_near_equal(prob['SA6'], [2.09276756, 2.09276756], tol)
        assert_near_equal(prob['SA7'], [0.04049836, 0.04049836], tol)


class DragCoefCleanTest(unittest.TestCase):
    def test_case1(self):
        """BWB data"""
        prob = om.Problem()
        prob.model.add_subsystem(
            'drag_coeff_clean',
            DragCoefClean(num_nodes=2),
            promotes=['*'],
        )

        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, [0.8, 0.8], units='unitless')
        prob.model.set_input_defaults('CL', [2.41793118, 2.41793118], units='unitless')

        # user inputs
        prob.model.set_input_defaults(
            Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT, 0.025, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(
            Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0, units='unitless'
        )

        # from aero setup
        prob.model.set_input_defaults('cf', [0.00283643, 0.00283643], units='unitless')
        prob.model.set_input_defaults('SA1', [0.80832432, 0.80832432], units='unitless')
        prob.model.set_input_defaults('SA2', [-0.13650645, -0.13650645], units='unitless')
        prob.model.set_input_defaults('SA5', [0.00962771, 0.00962771], units='unitless')
        prob.model.set_input_defaults('SA6', [2.09276756, 2.09276756], units='unitless')
        prob.model.set_input_defaults('SA7', [0.04049836, 0.04049836], units='unitless')

        prob.setup(check=False, force_alloc_complex=True)
        prob.run_model()

        tol = 1e-4
        assert_near_equal(prob['CD'], [0.5136233, 0.5136233], tol)


if __name__ == '__main__':
    # unittest.main()
    test = BWBCruiseAeroTest()
    test.test_cruise()
