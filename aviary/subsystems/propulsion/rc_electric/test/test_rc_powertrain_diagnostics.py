"""Consolidated regression tests for the small_uav RC powertrain / propeller surrogate."""


import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.propulsion.rc_electric.model import rc_performance as rcp
from aviary.subsystems.propulsion.rc_electric.model.rc_performance import (
    PropCoefficients,
    Propeller,
)
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.dbf_variables import Aircraft, Dynamic

# --- small_uav cruise operating point (matches Takeoff_Cruise_Attempt.py) ---
MASS = 7.0000189            # kg gross
G = 9.80665
W = MASS * G                # N
S = 0.65039                 # m^2 wing area
RHO = 1.214                 # kg/m^3 at ~200 ft
SOS = 340.3                 # m/s speed of sound (mach -> velocity)
CD0, K = 0.04, 0.04         # SimplestDragCoeff drag polar: CD = CD0 + K*CL^2
DIAMETER = 0.4826           # m   (19 in prop in the example CSV)
PITCH = 12.0                # inch
MACH_CRUISE = 0.0538
V_CRUISE = MACH_CRUISE * SOS  # ~18.3 m/s

# Expected propeller-surrogate grid extents (xt columns: D[m], pitch[in], RPM[rev/s], V[m/s]).
EXPECTED_RANGES = {
    'diameter_m': (0.3302, 0.4826),
    'pitch_in': (4.0, 14.0),
    'rpm_revps': (16.6667, 183.3333),
}


def _drag(V):
    """Steady level-cruise required thrust = drag, from the custom_aero polar."""
    q = 0.5 * RHO * V**2
    cl = W / (q * S)
    cd = CD0 + K * cl**2
    return cd * q * S


def _thrust_vs_rpm(V, rps):
    """Isolated PropCoefficients + Propeller thrust [N] over an RPM (rev/s) sweep."""
    nn = len(rps)
    p = om.Problem()
    p.model.add_subsystem(
        'propco',
        PropCoefficients(
            method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn
        ),
        promotes=['*'],
    )
    p.model.add_subsystem('prop', Propeller(num_nodes=nn), promotes=['*'])
    p.setup()
    p.set_val('temp_diameter', np.full(nn, DIAMETER), units='m')
    p.set_val('temp_pitch', np.full(nn, PITCH), units='inch')
    p.set_val(Dynamic.Vehicle.Propulsion.RPM, rps, units='rev/s')
    p.set_val(Dynamic.Mission.VELOCITY, np.full(nn, V), units='m/s')
    p.set_val(Aircraft.Engine.Propeller.DIAMETER, DIAMETER, units='m')
    p.set_val(Dynamic.Atmosphere.DENSITY, np.full(nn, RHO), units='kg/m**3')
    p.run_model()
    return p.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='N')


@use_tempdirs
class TestPropSurrogateData(unittest.TestCase):
    """From diag_prop_gradient.py: the surrogate grid + training data must stay sane."""

    def test_training_ranges(self):
        # If the prop data file changes shape/units, these catch it.
        self.assertAlmostEqual(rcp.xt[:, 0].min(), EXPECTED_RANGES['diameter_m'][0], places=3)
        self.assertAlmostEqual(rcp.xt[:, 0].max(), EXPECTED_RANGES['diameter_m'][1], places=3)
        self.assertAlmostEqual(rcp.xt[:, 1].min(), EXPECTED_RANGES['pitch_in'][0], places=3)
        self.assertAlmostEqual(rcp.xt[:, 1].max(), EXPECTED_RANGES['pitch_in'][1], places=3)
        self.assertAlmostEqual(rcp.xt[:, 2].min(), EXPECTED_RANGES['rpm_revps'][0], places=2)
        self.assertAlmostEqual(rcp.xt[:, 2].max(), EXPECTED_RANGES['rpm_revps'][1], places=2)
        self.assertGreaterEqual(rcp.xt[:, 3].min(), 0.0)
        self.assertGreater(rcp.xt[:, 3].max(), 45.0)

    def test_no_nan_in_training_data(self):
        self.assertFalse(np.isnan(rcp.xt).any(), "NaN in prop surrogate input grid (xt)")
        self.assertFalse(np.isnan(rcp.ct).any(), "NaN in prop thrust-coefficient data (ct)")
        self.assertFalse(np.isnan(rcp.cp).any(), "NaN in prop power-coefficient data (cp)")

    def test_example_prop_within_grid(self):
        # The RPM RangeClamp in rcpropulsion_mission must match the data RPM extents,
        # otherwise the clamp lets the surrogate extrapolate.
        self.assertAlmostEqual(rcp.xt[:, 2].min(), 16.6667, places=2)
        self.assertAlmostEqual(rcp.xt[:, 2].max(), 183.3333, places=2)
        # The example's 19 in / 12 in-pitch prop must sit inside the trained grid.
        self.assertGreaterEqual(DIAMETER, rcp.xt[:, 0].min())
        self.assertLessEqual(DIAMETER, rcp.xt[:, 0].max())
        self.assertGreaterEqual(PITCH, rcp.xt[:, 1].min())
        self.assertLessEqual(PITCH, rcp.xt[:, 1].max())


@use_tempdirs
class TestCruiseOperatingPoint(unittest.TestCase):
    """From diag_cruise_speed.py: cruise thrust is reachable on a positive-slope region."""

    def test_drag_reachable_and_well_conditioned(self):
        rps = np.linspace(20.0, 183.0, 200)
        thrust = _thrust_vs_rpm(V_CRUISE, rps)
        self.assertFalse(np.isnan(thrust).any(), "prop surrogate returned NaN thrust at cruise")

        d_req = _drag(V_CRUISE)
        reachable = np.where(thrust >= d_req)[0]
        self.assertTrue(
            len(reachable) > 0,
            f"cruise drag {d_req:.2f} N unreachable (max thrust {thrust.max():.1f} N)",
        )

        i = reachable[0]
        slope = np.gradient(thrust, rps)
        # The operating point must sit on the positive-slope side of the thrust curve;
        # a flat / negative slope is what made the throttle-balance Jacobian singular.
        self.assertGreater(
            slope[i], 0.0, "cruise operating point is on a flat/negative thrust slope"
        )
        # ...and comfortably above the low-RPM windmilling region.
        self.assertGreater(rps[i], 25.0, "operating RPM is in the windmilling region")


@use_tempdirs
class TestPowertrainSolves(unittest.TestCase):
    """From diag_powertrain.py: the full RCPropMission converges, balances power, finite partials."""

    def _build(self, throttle, nn=1, power_balance_mode = 'feedforward'):
        p = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, 1)
        p.model.add_subsystem(
            'rc_prop_group',
            RCPropMission(
                num_nodes=nn, aviary_options=options, power_balance_mode=power_balance_mode
            ),
            promotes=['*'],
        )
        p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=50)
        p.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        p.model.nonlinear_solver.linesearch.options['bound_enforcement'] = 'scalar'
        p.model.nonlinear_solver.options['err_on_non_converge'] = False
        p.model.nonlinear_solver.options['iprint'] = -1
        p.model.linear_solver = om.DirectSolver(assemble_jac=True)
        p.setup(force_alloc_complex=True)

        # Powertrain parameters (rc_builder.get_parameters), 6S at full charge (25.2 V).
        p.set_val(Aircraft.Battery.VOLTAGE, 25.2, units='V')
        p.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
        p.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.05, units='ohm')
        p.set_val(Aircraft.Engine.Motor.KV, 400.0, units='rpm/V')
        p.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 2.2, units='A')
        p.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 100.0, units='A')
        p.set_val(Aircraft.Engine.Propeller.DIAMETER, DIAMETER, units='m')
        p.set_val(Aircraft.Engine.Propeller.PITCH, PITCH, units='inch')
        p.set_val(Dynamic.Atmosphere.DENSITY, RHO, units='kg/m**3')
        p.set_val(Dynamic.Mission.VELOCITY, V_CRUISE, units='m/s')
        p.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.full(nn, throttle))
        p.set_val('full_throttle', np.ones(nn))
        p.set_val(Dynamic.Vehicle.Propulsion.CURRENT, np.full(nn, 12.0), units='A')
        return p

    def test_converges_finite_at_cruise(self):
        p = self._build(throttle=0.5)
        p.run_model()
        checks = {
            Dynamic.Vehicle.Propulsion.RPM: 'rpm',
            Dynamic.Vehicle.Propulsion.CURRENT: 'A',
            Dynamic.Vehicle.Propulsion.THRUST: 'N',
            'ct': None,
            'cp': None,
        }
        for name, units in checks.items():
            val = p.get_val(name, units=units) if units else p.get_val(name)
            self.assertFalse(np.isnan(val).any(), f"{name} is NaN at the cruise solve")
        self.assertGreater(p.get_val(Dynamic.Vehicle.Propulsion.RPM, units='rpm')[0], 0.0)
        self.assertGreater(p.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='N')[0], 0.0)

    def test_power_balance(self):
        p = self._build(throttle=0.5, nn=3, power_balance_mode='solver')
        p.run_model()
        battery = p.get_val('battery.power', units='W')
        esc = p.get_val('esc.power', units='W')
        motor = p.get_val('motor.power', units='W')
        prop = p.get_val(Dynamic.Vehicle.Propulsion.PROP_POWER, units='W')
        assert_near_equal(battery + esc + motor - prop, np.zeros(3), tolerance=1e-5)

    def test_partials_finite(self):
        # The cruise throttle-balance NaN that blocked the example lived in derivatives,
        # so guard against NaN analytic partials anywhere in the powertrain.
        p = self._build(throttle=0.5, nn=3)
        p.run_model()
        data = p.check_partials(out_stream=None, method='fd', compact_print=True)
        for comp, subs in data.items():
            for key, d in subs.items():
                jac = d.get('J_fwd')
                if jac is None:
                    continue
                jac = np.asarray(jac, dtype=float)
                if jac.size:
                    self.assertFalse(
                        np.isnan(jac).any(), f"NaN analytic partial in {comp}: d{key[0]}/d{key[1]}"
                    )


if __name__ == '__main__':
    unittest.main()
