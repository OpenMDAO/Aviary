import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.rc_electric.model.UAV_mission import RCPropMission
from aviary.variable_info.dbf_variables import Aircraft, Dynamic


prob = om.Problem()
model = prob.model

model.add_subsystem('atm', Atmosphere(num_nodes=1), promotes=['*'])
model.add_subsystem('rc_engine', RCPropMission(num_nodes=1), promotes=['*'])

model.connect('rotations_per_minute', 'rpm_slack')
model.set_input_defaults(Dynamic.Mission.VELOCITY, units='mi/h')

prob.setup()

prob.set_val(Dynamic.Mission.ALTITUDE, 200.0, units='ft')
prob.set_val(Dynamic.Mission.VELOCITY, 0, units='mi/h')
prob.set_val(f'battery.{Aircraft.Battery.MASS}', 2.0, units='lbm')
prob.set_val(Aircraft.Engine.Motor.MASS, 1.0362, units='lbm')
prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 2.2, units='A')
prob.set_val(Aircraft.Engine.Motor.KV, 400, units='rpm/V')
prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.05, units='ohm')
prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 16, units='inch')
prob.set_val(Aircraft.Engine.Propeller.PITCH, 12, units='inch')
prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')

prob.run_model()

throttles = np.linspace(0.1, 1.0, num=10)
thrusts = np.zeros(throttles.shape)
esc_voltages = np.zeros(throttles.shape)
esc_powers = np.zeros(throttles.shape)
esc_effs = np.zeros(throttles.shape)
rpms = np.zeros(throttles.shape)
cps = np.zeros(throttles.shape)
cts = np.zeros(throttles.shape)
motor_powers = np.zeros(throttles.shape)
battery_voltages = np.zeros(throttles.shape)

for j, throttle in enumerate(throttles):
    prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, throttle)
    prob.run_model()
    thrust = prob.get_val('thrust_net', units='lbf')[0]
    thrusts[j] = thrust
    esc_voltage = prob.get_val('esc.voltage_out', units='V')[0]
    esc_voltages[j] = esc_voltage
    esc_power = prob.get_val('esc.power', units='W')[0]
    esc_powers[j] = esc_power
    esc_eff = prob.get_val('esc.efficiency')[0]
    esc_effs[j] = esc_eff
    rpm = prob.get_val(Dynamic.Vehicle.Propulsion.RPM)[0]
    rpms[j] = rpm
    ct = prob.get_val('ct')[0]
    cp = prob.get_val('cp')[0]
    cts[j] = ct
    cps[j] = cp
    motor_power = prob.get_val('motor.power', units='W')[0]
    motor_powers[j] = motor_power

    battery_volt = prob.get_val('battery.voltage_out', units='V')[0]
    battery_voltages[j] = battery_volt

plt.figure()
plt.plot(throttles, battery_voltages, '+-')
plt.xlabel('throttle')
plt.ylabel('Battery Voltage (V)')
plt.grid()

plt.figure()
plt.plot(throttles, thrusts, '+-')
plt.xlabel('throttle')
plt.ylabel('thrust (lbf)')
plt.grid()

plt.figure()
plt.plot(throttles, esc_voltages, '+-')
plt.xlabel('throttle')
plt.ylabel('ESC Voltage (V)')
plt.grid()

plt.figure()
plt.plot(throttles, cts, '+-', label='CT')
plt.plot(throttles, cps, '+-', label='CP')
plt.xlabel('throttle')
plt.ylabel('CT and CP')
plt.legend()
plt.grid()

plt.figure()
plt.plot(throttles, rpms, '+-')
plt.xlabel('throttle')
plt.ylabel('RPM')
plt.grid()

plt.figure()
plt.plot(throttles, esc_powers, '+-')
plt.xlabel('throttle')
plt.ylabel('ESC Power (W)')
plt.grid()

plt.figure()
plt.plot(throttles, esc_effs, '+-')
plt.xlabel('throttle')
plt.ylabel('ESC Efficiency')
plt.grid()

plt.figure()
plt.plot(throttles, motor_powers, '+-')
plt.xlabel('throttle')
plt.ylabel('Motor Power (W)')
plt.grid()

plt.show()

plt

print('done')



