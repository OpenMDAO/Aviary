"""Diagnostic: find a cruise speed whose steady operating point sits in the
propeller's well-conditioned (positive-slope) thrust region.

Drag model (custom_aero/simple_drag.py): lift = weight, CD = CD0 + k*CL^2,
Drag = CD * q * S, q = 0.5*rho*V^2.  At steady level cruise, required thrust = drag.
For each candidate speed we sweep RPM through the surrogate, find the RPM where
thrust == drag, and report the local slope d(thrust)/d(RPM) and advance ratio J.
"""
import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.rc_performance import (
    PropCoefficients, Propeller, xt,
)
from aviary.variable_info.variables import Aircraft, Dynamic

# --- aircraft / environment constants ---
MASS = 7.0000189          # kg  (loaded gross mass)
G = 9.80665
W = MASS * G              # N
S = 0.65039               # m^2 wing area
RHO = 1.214               # kg/m^3 at ~200 ft
SOS = 340.3               # m/s speed of sound (mach -> velocity)
CD0, K = 0.04, 0.04       # drag polar from SimplestDragCoeff
DIAMETER = 0.4826         # m
PITCH = 12.0              # inch


def drag(V):
    q = 0.5 * RHO * V**2
    cl = W / (q * S)
    cd = CD0 + K * cl**2
    return cd * q * S, cl


def thrust_curve(V, rps):
    nn = len(rps)
    p = om.Problem()
    p.model.add_subsystem(
        'propco',
        PropCoefficients(method='lagrange2', extrapolate=True,
                         training_data_gradients=True, vec_size=nn),
        promotes=['*'])
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


rps = np.linspace(20.0, 183.0, 200)
machs = [0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12]

print(f"  W = {W:.2f} N,  S = {S} m^2,  D = {DIAMETER} m,  pitch = {PITCH} in\n")
print("  mach   V[m/s]   q[Pa]    CL     drag[N]   op-RPM[rpm]   J=V/nD   slope dT/dn   max-thrust[N]")
for m in machs:
    V = m * SOS
    D_req, cl = drag(V)
    T = thrust_curve(V, rps)
    dT = np.gradient(T, rps)
    # find RPM where thrust crosses required thrust (lowest such crossing)
    idx = np.where(T >= D_req)[0]
    if len(idx) == 0:
        print(f"  {m:.2f}   {V:6.2f}   {0.5*RHO*V*V:6.1f}   {cl:5.2f}   {D_req:6.2f}    "
              f"  UNREACHABLE (max thrust {T.max():.1f} N < drag)")
        continue
    i = idx[0]
    n_op = rps[i]
    J = V / (n_op * DIAMETER)
    print(f"  {m:.2f}   {V:6.2f}   {0.5*RHO*V*V:6.1f}   {cl:5.2f}   {D_req:6.2f}    "
          f"{n_op*60:8.0f}     {J:5.2f}    {dT[i]:8.3f}      {T.max():7.1f}")

print("\nGuidance: pick a mach where op-RPM is comfortably above ~2400 rpm (40 rev/s),")
print("slope dT/dn is solidly positive, and drag is reachable with margin.")
