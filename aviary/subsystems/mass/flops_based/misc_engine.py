import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class EngineMiscMass(om.ExplicitComponent):
    """
    Calculates miscellaneous mass for a set of a engine model.

    Assumptions
    -----------
    Calculates total sum of all misc engine mass on the aircraft

    Currently using engine-level additional mass (scaled by num_engines)
    and propulsion-level starter and controls mass, not heterogeneous engine safe!
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Engine.ADDITIONAL_MASS, shape=num_engine_type, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.MISC_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_STARTER_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_MISC_MASS, units='lbm')

        self.declare_partials(
            of=Aircraft.Propulsion.TOTAL_MISC_MASS,
            wrt=[
                Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS,
                Aircraft.Propulsion.TOTAL_STARTER_MASS,
                Aircraft.Engine.ADDITIONAL_MASS,
                Aircraft.Propulsion.MISC_MASS_SCALER,
            ],
            val=1,
        )

    def compute(self, inputs, outputs):
        # TODO temporarily using engine-level additional mass and multiplying
        #      by num_engines to get propulsion-level additional mass
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        addtl_mass = sum(inputs[Aircraft.Engine.ADDITIONAL_MASS] * num_engines)
        ctrl_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS]
        starter_mass = inputs[Aircraft.Propulsion.TOTAL_STARTER_MASS]
        scaler = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]

        misc_mass = (starter_mass + addtl_mass + ctrl_mass) * scaler

        outputs[Aircraft.Propulsion.TOTAL_MISC_MASS] = misc_mass

    def compute_partials(self, inputs, J):
        # TODO temporarily using engine-level additional mass and multiplying
        #      by num_engines to get propulsion-level additional mass
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]

        addtl_mass = inputs[Aircraft.Engine.ADDITIONAL_MASS] * num_engines
        ctrl_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS]
        starter_mass = inputs[Aircraft.Propulsion.TOTAL_STARTER_MASS]
        scaler = inputs[Aircraft.Propulsion.MISC_MASS_SCALER]

        J[Aircraft.Propulsion.TOTAL_MISC_MASS, Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = (
            scaler
        )

        J[Aircraft.Propulsion.TOTAL_MISC_MASS, Aircraft.Propulsion.TOTAL_STARTER_MASS] = scaler

        J[Aircraft.Propulsion.TOTAL_MISC_MASS, Aircraft.Engine.ADDITIONAL_MASS] = (
            scaler * num_engines
        )

        J[Aircraft.Propulsion.TOTAL_MISC_MASS, Aircraft.Propulsion.MISC_MASS_SCALER] = (
            starter_mass + sum(addtl_mass) + ctrl_mass
        )
