"""Diagnostic: characterize the propeller surrogate thrust vs RPM at cruise.

We isolate PropCoefficients (the ct/cp metamodel) + Propeller so we can see where
d(thrust)/d(RPM) goes to zero -- that flat spot is what makes the cruise throttle
balance Jacobian singular at certain nodes.
"""
import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.rc_performance import (
    PropCoefficients, Propeller, xt,
)
from aviary.variable_info.variables import Aircraft, Dynamic

# Cruise operating point from Cruise_Attempt.py: mach 0.04, ~200 ft.
V_CRUISE = 0.04 * 340.3          # m/s (approx sea-level-ish speed of sound)
DIAMETER = 0.4826                # m   (default in Vectorization / prop)
PITCH = 12.0                     # inch
RHO = 1.214                      # kg/m^3 (approx at 200 ft)

print("=== Propeller training-data ranges (xt columns = D[m], pitch[in], RPM[rev/s], V[m/s]) ===")
print("diameter :", xt[:, 0].min(), "->", xt[:, 0].max())
print("pitch    :", xt[:, 1].min(), "->", xt[:, 1].max())
print("RPM rev/s:", xt[:, 2].min(), "->", xt[:, 2].max())
print("velocity :", xt[:, 3].min(), "->", xt[:, 3].max())
print(f"\nCruise point: V={V_CRUISE:.3f} m/s, D={DIAMETER} m, pitch={PITCH} in, rho={RHO}")

# Sweep RPM (rev/s) across the trained range.
rps = np.linspace(max(xt[:, 2].min(), 1.0), xt[:, 2].max(), 60)
nn = len(rps)

p = om.Problem()
p.model.add_subsystem(
    'propco',
    PropCoefficients(method='lagrange2', extrapolate=True,
                     training_data_gradients=True, vec_size=nn),
    promotes=['*'],
)
p.model.add_subsystem('prop', Propeller(num_nodes=nn), promotes=['*'])
p.setup(force_alloc_complex=True)

p.set_val('temp_diameter', np.full(nn, DIAMETER), units='m')
p.set_val('temp_pitch', np.full(nn, PITCH), units='inch')
p.set_val(Dynamic.Vehicle.Propulsion.RPM, rps, units='rev/s')
p.set_val(Dynamic.Mission.VELOCITY, np.full(nn, V_CRUISE), units='m/s')
p.set_val(Aircraft.Engine.Propeller.DIAMETER, DIAMETER, units='m')
p.set_val(Dynamic.Atmosphere.DENSITY, np.full(nn, RHO), units='kg/m**3')

p.run_model()

ct = p.get_val('ct')
cp = p.get_val('cp')
thrust = p.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='N')

# Numerical slope d(thrust)/d(RPM)
dT = np.gradient(thrust, rps)

print("\n  RPM[rev/s]   RPM[rpm]      ct          cp        thrust[N]   dThrust/dRPM")
for i in range(nn):
    flag = "  <-- near-zero slope" if abs(dT[i]) < 0.05 * np.max(np.abs(dT)) else ""
    print(f"  {rps[i]:8.2f}   {rps[i]*60:8.0f}   {ct[i]:9.5f}   {cp[i]:9.5f}   "
          f"{thrust[i]:9.3f}   {dT[i]:11.4f}{flag}")

# Where does thrust peak / slope cross zero?
sign_change = np.where(np.diff(np.sign(dT)) != 0)[0]
print("\nRPM (rev/s) where d(thrust)/d(RPM) changes sign (thrust extrema):",
      [round(float(rps[i]), 2) for i in sign_change])
print("Negative-ct region (RPM rev/s):",
      [round(float(r), 2) for r, c in zip(rps, ct) if c < 0])
