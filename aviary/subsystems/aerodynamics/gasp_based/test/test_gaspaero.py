import json
import os
import unittest

import numpy as np
import openmdao.api as om
import pandas as pd
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.aerodynamics.gasp_based.gaspaero import CruiseAero, LowSpeedAero
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

here = os.path.abspath(os.path.dirname(__file__))
cruise_data = pd.read_csv(os.path.join(here, 'data', 'aero_data_cruise.csv'))
ground_data = pd.read_csv(os.path.join(here, 'data', 'aero_data_ground.csv'))
setup_data = json.load(open(os.path.join(here, 'data', 'aero_data_setup.json')))


class GASPAeroTest(unittest.TestCase):
    """Test overall pre-mission and mission aero systems in cruise and near-ground flight."""

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

    def test_ground_alpha_out(self):
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


if __name__ == '__main__':
    unittest.main()
