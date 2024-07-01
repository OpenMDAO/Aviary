import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class SizeBattery(om.ExplicitComponent):
    '''
    Calculates battery mass from specific energy and additional mass
    '''

    def initialize(self):
        self.options.declare(
            'aviary_inputs', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Battery.PACK_MASS, val=0.0, units='kg',
                         desc='mass of energy-storing components of battery')
        add_aviary_input(self, Aircraft.Battery.ADDITIONAL_MASS, val=0.0, units='kg',
                         desc='mass of non energy-storing components of battery')
        add_aviary_input(self, Aircraft.Battery.PACK_ENERGY_DENSITY, val=0.0, units='kJ/kg',
                         desc='energy density of battery pack')

        add_aviary_output(self, Aircraft.Battery.MASS, val=0.0,
                          units='kg', desc='total battery mass')
        add_aviary_output(self, Aircraft.Battery.ENERGY_CAPACITY, val=0.0,
                          units='kJ', desc='total battery energy storage')

    def compute(self, inputs, outputs):
        energy_density_kj_kg = inputs[Aircraft.Battery.PACK_ENERGY_DENSITY]
        addtl_mass = inputs[Aircraft.Battery.ADDITIONAL_MASS]
        pack_mass = inputs[Aircraft.Battery.PACK_MASS]

        total_mass = pack_mass + addtl_mass
        total_energy = pack_mass * energy_density_kj_kg

        outputs[Aircraft.Battery.MASS] = total_mass
        outputs[Aircraft.Battery.ENERGY_CAPACITY] = total_energy

    def setup_partials(self):
        self.declare_partials(Aircraft.Battery.ENERGY_CAPACITY,
                              Aircraft.Battery.PACK_ENERGY_DENSITY)
        self.declare_partials(Aircraft.Battery.ENERGY_CAPACITY,
                              Aircraft.Battery.PACK_MASS)

        self.declare_partials(Aircraft.Battery.MASS,
                              Aircraft.Battery.ADDITIONAL_MASS, val=1.0)
        self.declare_partials(Aircraft.Battery.MASS,
                              Aircraft.Battery.PACK_MASS, val=1.0)

    def compute_partials(self, inputs, J):
        energy_density_kj_kg = inputs[Aircraft.Battery.PACK_ENERGY_DENSITY]
        pack_mass = inputs[Aircraft.Battery.PACK_MASS]

        J[Aircraft.Battery.ENERGY_CAPACITY,
            Aircraft.Battery.PACK_ENERGY_DENSITY] = pack_mass
        J[Aircraft.Battery.ENERGY_CAPACITY,
            Aircraft.Battery.PACK_MASS] = energy_density_kj_kg
