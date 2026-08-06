import numpy as np
import openmdao.api as om
from functools import partial

from aviary.variable_info.functions import add_aviary_input as _add_aviary_input
from aviary.variable_info.functions import add_aviary_option as _add_aviary_option
from aviary.variable_info.functions import add_aviary_output as _add_aviary_output
from aviary.variable_info.variables import Settings
from aviary.variable_info.UAV_variables import Aircraft, Dynamic
from aviary.variable_info.UAV_variable_meta_data import ExtendedMetaData

# RC electric variables live in ExtendedMetaData; bind it onto the add_aviary_* helpers.
add_aviary_input = partial(_add_aviary_input, meta_data=ExtendedMetaData)
add_aviary_output = partial(_add_aviary_output, meta_data=ExtendedMetaData)
add_aviary_option = partial(_add_aviary_option, meta_data=ExtendedMetaData)


#This only works if we already know how many batteries we have.
class Throttle(om.ExplicitComponent):
    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.Motor.MAX_CONT_CURRENT, val=100.0, units='A')
        self.options.declare('num_nodes', default=1, types=int)
    def setup(self):
        nn = self.options['num_nodes']
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.THROTTLE, shape=(nn), units='unitless')

        self.add_output(Dynamic.Vehicle.Propulsion.CURRENT, shape=(nn), units='A')

    def setup_partials(self):
        nn = self.options['num_nodes']
        max_current = np.asarray(self.options[Aircraft.Engine.Motor.MAX_CONT_CURRENT][0]).item()
        self.declare_partials(Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.THROTTLE, val=max_current, rows=np.arange(nn), cols=np.arange(nn))


    def compute(self, inputs, outputs):
        throttle = inputs[Dynamic.Vehicle.Propulsion.THROTTLE]
        max_current = np.asarray(self.options[Aircraft.Engine.Motor.MAX_CONT_CURRENT][0]).item()
        outputs[Dynamic.Vehicle.Propulsion.CURRENT] = throttle * max_current









class Battery(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #TODO: add Battery options
        add_aviary_input(self, Aircraft.Battery.VOLTAGE, units = "V")
        add_aviary_input(self, Aircraft.Battery.RESISTANCE, units='ohm')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.CURRENT, shape=(nn,), units='A')

        self.add_output('voltage_out', val=np.zeros(nn), units='V')
        self.add_output('power', val=np.zeros(nn), units='W')
        self.add_output('dt_denergy_used', val=np.zeros(nn), units='W')
        ar = np.arange(nn)

        self.declare_partials(
            ['voltage_out', 'power'],
            [Aircraft.Battery.VOLTAGE, Aircraft.Battery.RESISTANCE],
        )

        self.declare_partials(
            ['voltage_out', 'power'],
            [Dynamic.Vehicle.Propulsion.CURRENT],
            rows=ar, cols=ar
        )

        # dt_denergy_used is the energy_used state rate; missing these partials
        # made the mission energy objective's gradient silently zero.
        self.declare_partials('dt_denergy_used', Aircraft.Battery.VOLTAGE)
        self.declare_partials(
            'dt_denergy_used', Dynamic.Vehicle.Propulsion.CURRENT, rows=ar, cols=ar
        )

    def compute(self, inputs, outputs):
        V = inputs[Aircraft.Battery.VOLTAGE]
        I = inputs[Dynamic.Vehicle.Propulsion.CURRENT]
        R = inputs[Aircraft.Battery.RESISTANCE]

        outputs['power'] = V * I - I**2 * R

        outputs['dt_denergy_used'] = V * I
        outputs['voltage_out'] = V - I * R

    def compute_partials(self, inputs, partials):
        V = inputs[Aircraft.Battery.VOLTAGE]
        I = inputs[Dynamic.Vehicle.Propulsion.CURRENT]
        R = inputs[Aircraft.Battery.RESISTANCE]

        partials['voltage_out', Aircraft.Battery.VOLTAGE] = 1
        partials['voltage_out', Dynamic.Vehicle.Propulsion.CURRENT] = -R
        partials['voltage_out', Aircraft.Battery.RESISTANCE] = -I

        partials['power', Aircraft.Battery.VOLTAGE] = I
        partials['power', Dynamic.Vehicle.Propulsion.CURRENT] = V - 2 * I * R
        partials['power', Aircraft.Battery.RESISTANCE] = -I**2

        partials['dt_denergy_used', Aircraft.Battery.VOLTAGE] = I
        partials['dt_denergy_used', Dynamic.Vehicle.Propulsion.CURRENT] = V * np.ones(len(I))



class ElectronicSpeedController(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

        self.options.declare('n', default = 1.6054, desc = 'a coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
        self.options.declare('o', default = 1.6519, desc = 'b coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
        self.options.declare('p', default = 0.6455, desc = 'c coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('voltage_in', val=np.zeros(nn), units = 'V')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.CURRENT, shape=(nn,), units='A')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.THROTTLE, shape=(nn,), units='unitless')

        self.add_output('efficiency', val=np.zeros(nn), units='unitless')
        self.add_output('voltage_out', val=np.zeros(nn), units = 'V')
        self.add_output('power', val=np.zeros(nn), units = 'W')
        self.add_output('current_out', val=np.zeros(nn), units='A')
        ar = np.arange(nn)

        self.declare_partials('efficiency', Dynamic.Vehicle.Propulsion.THROTTLE, rows=ar, cols=ar)
        self.declare_partials('voltage_out', ['voltage_in', Dynamic.Vehicle.Propulsion.THROTTLE], rows=ar, cols=ar)
        self.declare_partials('power', ['voltage_in', Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.THROTTLE], rows=ar, cols=ar)
        self.declare_partials('current_out', Dynamic.Vehicle.Propulsion.CURRENT, val=1.0, rows=ar, cols=ar)
    def compute(self, inputs, outputs):

        a = self.options['n']
        b = self.options['o']
        c = self.options['p']
        thr = inputs[Dynamic.Vehicle.Propulsion.THROTTLE]
        transition = 0.05
        m = a*b*c*transition**(c - 1) / (b*transition**c + 1)**2

        outputs['current_out'] = inputs[Dynamic.Vehicle.Propulsion.CURRENT] #* thr
        outputs['efficiency'] = np.where(thr >= transition, a * (1 - 1 / (1 + b*thr**c)), m* thr + ((a * (1 - 1 / (1 + b*transition**c)))-m*transition))
        outputs['voltage_out'] = inputs['voltage_in'] * inputs[Dynamic.Vehicle.Propulsion.THROTTLE] * outputs['efficiency']
        outputs['power'] = (outputs['efficiency'] - 1) * inputs[Dynamic.Vehicle.Propulsion.CURRENT] * inputs['voltage_in']

    def compute_partials(self, inputs, partials):

        a = self.options['n']
        b = self.options['o']
        c = self.options['p']
        t = inputs[Dynamic.Vehicle.Propulsion.THROTTLE]
        transition=0.05

        m = a*b*c*transition**(c - 1) / (b*transition**c + 1)**2
        efficiency = np.where(t >= transition, a * (1 - 1 / (1 + b*t**c)), m*t + ((a * (1 - 1 / (1 + b*transition**c)))-m*transition))

        t_safe = np.where(t > 0, t, transition)
        partials['efficiency', Dynamic.Vehicle.Propulsion.THROTTLE] = np.where(t>=transition, a*b*c*t_safe**(c - 1) / (b*t_safe**c + 1)**2, m)

        # partials['']
        partials['voltage_out', 'voltage_in'] = inputs[Dynamic.Vehicle.Propulsion.THROTTLE] * efficiency
        partials['voltage_out', Dynamic.Vehicle.Propulsion.THROTTLE] = inputs['voltage_in'] * (efficiency + inputs[Dynamic.Vehicle.Propulsion.THROTTLE] * partials['efficiency', Dynamic.Vehicle.Propulsion.THROTTLE])

        partials['power', 'voltage_in'] = (efficiency - 1) * inputs[Dynamic.Vehicle.Propulsion.CURRENT]
        partials['power', Dynamic.Vehicle.Propulsion.CURRENT] = (efficiency - 1) * inputs['voltage_in']
        partials['power', Dynamic.Vehicle.Propulsion.THROTTLE] = inputs[Dynamic.Vehicle.Propulsion.CURRENT] * inputs['voltage_in'] * partials['efficiency', Dynamic.Vehicle.Propulsion.THROTTLE]


class Motor(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        # Percentage multiply on no-load RPM to estimate load (1.0 = no derate).
        self.options.declare('load_factor', default=1.0, types=float)

    def setup(self):
        nn = self.options['num_nodes']


        add_aviary_input(self, Aircraft.Engine.Motor.IDLE_CURRENT,  units='A')

        add_aviary_input(self, Aircraft.Engine.Motor.RESISTANCE, units='ohm')
        add_aviary_input(self, Aircraft.Engine.Motor.KV, units='rpm/V')
        self.add_input('voltage_in', val=np.zeros(nn), units = 'V')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.CURRENT, shape=(nn,), units='A')
        self.add_input('current', val=np.zeros(nn), units = 'A')

        ################ TODO Alex #####################
        add_aviary_output(self, Dynamic.Vehicle.Propulsion.RPM, shape=(nn,), units='rpm')

        # (no bound here: RPM is derated via load_factor instead; an output upper= is inert on an explicit comp anyway)
        ################ TODO Alex #####################
        self.add_output('power', val=np.zeros(nn), units='W')

        ar=np.arange(nn)

        self.declare_partials(
            [
                Dynamic.Vehicle.Propulsion.RPM,
                'power'
            ],
            [
                'voltage_in',
                'current',
                # Dynamic.Vehicle.Propulsion.CURRENT,
            ],
            rows=ar, cols=ar
        )

        self.declare_partials(
            [Dynamic.Vehicle.Propulsion.RPM,],
            [Aircraft.Engine.Motor.RESISTANCE, Aircraft.Engine.Motor.KV]
        )

        self.declare_partials(
            ['power'],
            [Aircraft.Engine.Motor.RESISTANCE, Aircraft.Engine.Motor.IDLE_CURRENT]
        )





    def compute(self, inputs, outputs):
        R = inputs[Aircraft.Engine.Motor.RESISTANCE]
        kv = inputs[Aircraft.Engine.Motor.KV]
        lf = self.options['load_factor']
        voltage_prop = inputs['voltage_in'] - inputs['current'] * R
        outputs[Dynamic.Vehicle.Propulsion.RPM] = lf * kv * voltage_prop
        outputs['power'] = -inputs['current']**2 * R - inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * voltage_prop


    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        R = inputs[Aircraft.Engine.Motor.RESISTANCE]
        lf = self.options['load_factor']

        voltage_prop = inputs['voltage_in'] - inputs['current'] * R
        dvoltage_prop_dvoltage_in = 1
        dvoltage_prop_dcurrent = -R
        dvoltage_prop_dresistance = -inputs['current']

        partials[Dynamic.Vehicle.Propulsion.RPM, 'voltage_in'] = lf * inputs[Aircraft.Engine.Motor.KV] * dvoltage_prop_dvoltage_in
        partials[Dynamic.Vehicle.Propulsion.RPM, 'current'] = lf * inputs[Aircraft.Engine.Motor.KV] * dvoltage_prop_dcurrent
        partials[Dynamic.Vehicle.Propulsion.RPM, Aircraft.Engine.Motor.RESISTANCE] = lf * inputs[Aircraft.Engine.Motor.KV] * dvoltage_prop_dresistance
        partials[Dynamic.Vehicle.Propulsion.RPM, Aircraft.Engine.Motor.KV] = lf * voltage_prop

        partials['power', 'voltage_in'] = -inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * dvoltage_prop_dvoltage_in
        partials['power', 'current'] = -2 * inputs['current'] * R - inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * dvoltage_prop_dcurrent
        partials['power', Aircraft.Engine.Motor.RESISTANCE] = -inputs['current']**2 - inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * dvoltage_prop_dresistance
        partials['power', Aircraft.Engine.Motor.IDLE_CURRENT] = -voltage_prop





#TODO: reading in of data should be changed later:
from aviary.subsystems.propulsion.UAV.Parsing.PropDataReader import PropDataReader
xt, ct, cp = PropDataReader()
ct = ct.flatten()
cp = cp.flatten()
order = np.lexsort((xt[:,3], xt[:,2], xt[:,1], xt[:,0]))
xt = xt[order]

class Vectorization(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
    def setup(self):
        nn=self.options['num_nodes']
        add_aviary_input(self, Aircraft.Engine.Propeller.DIAMETER, units = 'm')
        add_aviary_input(self, Aircraft.Engine.Propeller.PITCH, units = 'inch')

        self.add_output('temp_diameter', val=np.zeros(nn), units='m')
        self.add_output('temp_pitch', val=np.zeros(nn), units='inch')

        self.declare_partials('temp_diameter', Aircraft.Engine.Propeller.DIAMETER, val=1)
        self.declare_partials('temp_pitch', Aircraft.Engine.Propeller.PITCH, val=1)
    def compute(self, inputs, outputs):
        nn=self.options['num_nodes']

        outputs['temp_diameter'] = inputs[Aircraft.Engine.Propeller.DIAMETER] * np.ones(nn)
        outputs['temp_pitch'] = inputs[Aircraft.Engine.Propeller.PITCH] * np.ones(nn)




class PropCoefficients(om.MetaModelSemiStructuredComp):
    def initialize(self):
        self.options.declare('method', default='lagrange2', types=str)
        self.options.declare('extrapolate', default=True, types=bool)
        self.options.declare('training_data_gradients', default=True, types=bool)
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        nn = self.options['vec_size']
        self.add_input('temp_diameter', val=np.zeros(nn), training_data = xt[:, 0], units='m',desc="propeller diameter")
        self.add_input('temp_pitch', val=np.zeros(nn), training_data = xt[:, 1], units="inch", desc="propeller pitch")
        self.add_input(Dynamic.Vehicle.Propulsion.RPM, val=np.zeros(nn), training_data = xt[:, 2], units='rev/s')
        self.add_input(Dynamic.Mission.VELOCITY, val=np.zeros(nn), training_data = xt[:, 3], units='m/s')

        self.add_output('ct', training_data = ct[order], units='unitless')
        self.add_output('cp', training_data = cp[order], units='unitless')


class Propeller(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare(Aircraft.Engine.NUM_ENGINES, default=1)
    #TODO: ask about adding more propellers
    def setup(self):
        nn = self.options['num_nodes']
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]
        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, shape=(nn,), units='kg/m**3')
        add_aviary_input(self, Aircraft.Engine.Propeller.DIAMETER, units = 'm')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.RPM, shape=(nn,), units='rev/s')
        self.add_input("ct", val=np.zeros(nn), units='unitless')
        self.add_input("cp", val=np.zeros(nn), units='unitless')



        add_aviary_output(self, Dynamic.Vehicle.Propulsion.THRUST, shape=(nn,), units='N')
        add_aviary_output(self, Dynamic.Vehicle.Propulsion.PROP_POWER, shape=(nn,), units='W')
        self.add_output('rpm_constraint', val=np.zeros(nn), units='rev/s')
        ar =np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.THRUST,
            [
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Vehicle.Propulsion.RPM,
                'ct',
            ],
            rows=ar, cols=ar,
        )

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.PROP_POWER,
            [
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Vehicle.Propulsion.RPM,
                'cp',
            ],
            rows=ar, cols=ar
        )

        self.declare_partials(
            'rpm_constraint',
            Dynamic.Vehicle.Propulsion.RPM,
            rows=ar, cols=ar
        )
        self.declare_partials(
            'rpm_constraint',
            Aircraft.Engine.Propeller.DIAMETER,
        )

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.THRUST,
            Aircraft.Engine.Propeller.DIAMETER,
        )

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.PROP_POWER,
            Aircraft.Engine.Propeller.DIAMETER,
        )

    def compute(self, inputs, outputs):
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        D = inputs[Aircraft.Engine.Propeller.DIAMETER]
        n = inputs [Dynamic.Vehicle.Propulsion.RPM]
        outputs[Dynamic.Vehicle.Propulsion.THRUST] = rho * n**2 * D**4 * inputs["ct"] * num_engines
        outputs[Dynamic.Vehicle.Propulsion.PROP_POWER] = (rho * n**3 * D**5 * inputs["cp"])
        n_limit = 63.5 / D
        outputs['rpm_constraint'] = n - n_limit


    def compute_partials(self, inputs, partials):
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        D = inputs[Aircraft.Engine.Propeller.DIAMETER]
        n = inputs [Dynamic.Vehicle.Propulsion.RPM]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        partials[Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Atmosphere.DENSITY] = n**2 * D**4 * inputs["ct"] * num_engines
        partials[Dynamic.Vehicle.Propulsion.THRUST, Aircraft.Engine.Propeller.DIAMETER] = rho * n**2 * 4 * D**3 * inputs["ct"] * num_engines
        partials[Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Vehicle.Propulsion.RPM] = rho * 2 * n * D**4 * inputs["ct"] * num_engines
        partials[Dynamic.Vehicle.Propulsion.THRUST, 'ct'] = rho * n**2 * D**4 * num_engines

        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Atmosphere.DENSITY] = n**3 * D**5 * inputs["cp"]
        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, Aircraft.Engine.Propeller.DIAMETER] = rho * n**3 * 5 * D**4 * inputs["cp"]
        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.RPM] = rho * 3 * n**2 * D**5 * inputs["cp"]
        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, 'cp'] = rho * n**3 * D**5

        partials['rpm_constraint', Dynamic.Vehicle.Propulsion.RPM] = np.ones(len(n))
        partials['rpm_constraint', Aircraft.Engine.Propeller.DIAMETER] = 63.5 / D**2
