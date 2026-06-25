"""Diagnostic: solve the full RC powertrain (RCPropMission) at the 100 ft/s cruise
condition over a throttle sweep, to find the true operating throttle/RPM/current for
the required thrust and to check whether the inner NonlinearBlockGS converges there.
"""
import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic

V = 18.29        # m/s (60 ft/s)
RHO = 1.214      # kg/m^3
DRAG_REQ = 7.0   # N  (approx required thrust at 60 ft/s from diag_cruise_speed)

throttles = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

print(f"Cruise: V = {V} m/s (100 ft/s),  required thrust ~ {DRAG_REQ} N\n")
print("  throttle   conv?   RPM[rpm]   current[A]   thrust[N]")

for thr in throttles:
    p = om.Problem()
    p.model.add_subsystem(
        'rc', RCPropMission(num_nodes=1, aviary_options=AviaryValues()), promotes=['*'])
    # Outer Newton so the implicit current_flow state is actually solved (the inner
    # NonlinearBlockGS alone leaves it pinned at its initial value).
    p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=50)
    p.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
    p.model.nonlinear_solver.options['err_on_non_converge'] = False
    p.model.nonlinear_solver.options['iprint'] = -1
    p.model.linear_solver = om.DirectSolver()
    p.setup()

    # powertrain parameters (from rc_builder.get_parameters)
    p.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
    p.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
    p.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.05, units='ohm')
    p.set_val(Aircraft.Engine.Motor.KV, 400.0, units='rpm/V')
    p.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 2.2, units='A')
    p.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 100.0, units='A')
    p.set_val(Aircraft.Engine.Propeller.DIAMETER, 0.4826, units='m')
    p.set_val(Aircraft.Engine.Propeller.PITCH, 12.0, units='inch')
    p.set_val(Dynamic.Atmosphere.DENSITY, RHO, units='kg/m**3')
    p.set_val(Dynamic.Mission.VELOCITY, V, units='m/s')
    p.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, thr, units='unitless')
    p.set_val('full_throttle', 1.0, units='unitless')
    # initial guesses for the implicit current state
    p.set_val(Dynamic.Vehicle.Propulsion.CURRENT, 40.0, units='A')

    converged = True
    try:
        p.run_model()
    except Exception:
        converged = False

    rpm = p.get_val(Dynamic.Vehicle.Propulsion.RPM, units='rpm')[0]
    cur = p.get_val(Dynamic.Vehicle.Propulsion.CURRENT, units='A')[0]
    thrust = p.get_val(Dynamic.Vehicle.Propulsion.THRUST, units='N')[0]
    # residual of the implicit current balance, to judge convergence quality
    print(f"   {thr:5.2f}     {str(converged):5s}   {rpm:8.0f}   {cur:9.2f}   {thrust:8.2f}")

print("\nFind the throttle whose thrust ~= required thrust; use that throttle/RPM/current")
print("as the seed in Cruise_Attempt.py.")
