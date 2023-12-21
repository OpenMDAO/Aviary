import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class EngineMiscMass(om.ExplicitComponent):
    '''
    Calculates miscellaneous mass for a set of a engine model.

    Assumptions
    -----------
    Calculates total sum of all misc engine mass on the aircraft

    Currently using engine-level additional mass (scaled by num_engines)
    and propulsion-level starter and controls mass, not multi-engine safe!
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(
            self, Aircraft.Engine.ADDITIONAL_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.MISC_MASS_SCALER, val=0.0)
        add_aviary_input(
            self, Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_STARTER_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_MISC_MASS, val=0.0)

        self.declare_partials(
            of=Aircraft.Propulsion.TOTAL_MISC_MASS,
            wrt=[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                 Aircraft.Propulsion.TOTAL_STARTER_MASS,
                 Aircraft.Engine.ADDITIONAL_MASS,
                 Aircraft.Propulsion.MISC_MASS_SCALER],
            val=1
        )

    def compute(self, inputs, outputs):
        # TODO temporarily using engine-level additional mass and multiplying
        #      by num_engines to get propulsion-level additional mass
        options: AviaryValues = self.options['aviary_options']
        num_engines = options.get_val(Aircraft.Engine.NUM_ENGINES)

        addtl_mass = sum(inputs[Aircraft.Engine.ADDITIONAL_MASS] * num_engines)
        ctrl_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS]
        starter_mass = inputs[Aircraft.Propulsion.TOTAL_STARTER_MASS]
        scaler = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]

        misc_mass = (starter_mass + addtl_mass + ctrl_mass) * scaler

        outputs[Aircraft.Propulsion.TOTAL_MISC_MASS] = misc_mass

    def compute_partials(self, inputs, J):
        # TODO temporarily using engine-level additional mass and multiplying
        #      by num_engines to get propulsion-level additional mass
        options: AviaryValues = self.options['aviary_options']
        num_engines = options.get_val(Aircraft.Engine.NUM_ENGINES)

        addtl_mass = inputs[Aircraft.Engine.ADDITIONAL_MASS] * num_engines
        ctrl_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS]
        starter_mass = inputs[Aircraft.Propulsion.TOTAL_STARTER_MASS]
        scaler = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]

        J[Aircraft.Propulsion.TOTAL_MISC_MASS,
          Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS
          ] = scaler

        J[Aircraft.Propulsion.TOTAL_MISC_MASS,
          Aircraft.Propulsion.TOTAL_STARTER_MASS
          ] = scaler

        J[Aircraft.Propulsion.TOTAL_MISC_MASS,
          Aircraft.Engine.ADDITIONAL_MASS
          ] = scaler * num_engines

        J[Aircraft.Propulsion.TOTAL_MISC_MASS,
          Aircraft.Propulsion.MISC_MASS_SCALER
          ] = starter_mass + addtl_mass + ctrl_mass
