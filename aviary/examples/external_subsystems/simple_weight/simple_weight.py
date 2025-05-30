"""
This is a simplified example of a component that computes a weight for the
wing and horizontal tail. The calculation is nonsensical, but it shows how
a basic external subsystem an use hierarchy inputs and set new values for
parts of the weight calculation.

The wing and tail weights will replace Aviary's internally-computed values.

This examples shows that you can use the Aviary hierarchy in your component
(as we do for the wing and engine weight), but you can also use your own
local names (as we do for 'Tail'), and promote them in your builder.
"""

import openmdao.api as om

from aviary.variable_info.variables import Aircraft


class SimpleWeight(om.ExplicitComponent):
    """
    A simple component that computes a wing mass as a function of the engine mass.
    These values are not representative of any existing aircraft, and the component
    is meant to demonstrate the concept of an externally calculated subsystem mass.
    """

    def setup(self):
        self.add_input(Aircraft.Engine.MASS, 1.0, units='lbm')

        self.add_output(Aircraft.Wing.MASS, 1.0, units='lbm')
        self.add_output('Tail', 1.0, units='lbm')

        self.declare_partials(Aircraft.Wing.MASS, Aircraft.Engine.MASS, val=1.5)
        self.declare_partials('Tail', Aircraft.Engine.MASS, val=0.7)

    def compute(self, inputs, outputs):
        outputs[Aircraft.Wing.MASS] = 1.5 * inputs[Aircraft.Engine.MASS]
        outputs['Tail'] = 0.7 * inputs[Aircraft.Engine.MASS]
