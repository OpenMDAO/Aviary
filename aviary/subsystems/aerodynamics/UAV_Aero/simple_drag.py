import numpy as np
import openmdao.api as om

from aviary.subsystems.aerodynamics.aero_common import DynamicPressure
from aviary.subsystems.aerodynamics.flops_based.drag import SimpleDrag
from aviary.subsystems.aerodynamics.flops_based.lift import LiftEqualsWeight
from aviary.variable_info.variables import Aircraft, Dynamic


class SimplestDragCoeff(om.ExplicitComponent):
    """
    Simple representation of aircraft drag as CD = CD_zero + k * CL**2.

    Values are fictional. Typically, some higher fidelity method will go here instead.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', default=1, types=int, desc='Number of nodes along mission segment'
        )

        self.options.declare('CD_zero', default=0.04)
        self.options.declare('k', default=0.04)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('cl', val=np.zeros(nn), units='unitless')

        self.add_output('CD', val=np.zeros(nn), units='unitless')

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials('CD', 'cl', rows=arange, cols=arange)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        CD_zero = self.options['CD_zero']
        k = self.options['k']

        cl = inputs['cl']

        outputs['CD'] = CD_zero + k * cl**2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k = self.options['k']

        cl = inputs['cl']

        partials['CD', 'cl'] = 2.0 * k * cl


class WingArea(om.ExplicitComponent):
    """
    Reference wing area from span and root chord.

    Rectangular planform (taper = 1, as in the CSV): S = span * root_chord. For a tapered
    wing use S = span * root_chord * (1 + taper) / 2 and add Aircraft.Wing.TAPER_RATIO.
    Feeding this into lift/drag gives span and root_chord a real aerodynamic gradient.
    """

    def setup(self):
        self.add_input(Aircraft.Wing.SPAN, val=1.0, units='m')
        self.add_input(Aircraft.Wing.ROOT_CHORD, val=1.0, units='m')

        self.add_output('wing_area', val=1.0, units='m**2')

    def setup_partials(self):
        self.declare_partials('wing_area', [Aircraft.Wing.SPAN, Aircraft.Wing.ROOT_CHORD])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['wing_area'] = inputs[Aircraft.Wing.SPAN] * inputs[Aircraft.Wing.ROOT_CHORD]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials['wing_area', Aircraft.Wing.SPAN] = inputs[Aircraft.Wing.ROOT_CHORD]
        partials['wing_area', Aircraft.Wing.ROOT_CHORD] = inputs[Aircraft.Wing.SPAN]


class SimpleAeroGroup(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', default=1, types=int, desc='Number of nodes along mission segment'
        )

    def setup(self):
        nn = self.options['num_nodes']

        # Compute the reference area from span & root_chord (both design variables that also
        # feed the wing mass) so lift/drag get a real gradient w.r.t. span and chord.
        self.add_subsystem(
            'WingArea',
            WingArea(),
            promotes_inputs=[Aircraft.Wing.SPAN, Aircraft.Wing.ROOT_CHORD],
            promotes_outputs=['wing_area'],
        )

        self.add_subsystem(
            'DynamicPressure',
            DynamicPressure(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Atmosphere.MACH,
                Dynamic.Atmosphere.STATIC_PRESSURE,
            ],
            promotes_outputs=[Dynamic.Atmosphere.DYNAMIC_PRESSURE],
        )

        self.add_subsystem(
            'Lift',
            LiftEqualsWeight(num_nodes=nn),
            promotes_inputs=[
                (Aircraft.Wing.AREA, 'wing_area'),
                Dynamic.Vehicle.MASS,
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            ],
            promotes_outputs=[Dynamic.Vehicle.LIFT_COEFFICIENT, Dynamic.Vehicle.LIFT],
        )

        self.add_subsystem(
            'SimpleDragCoeff',
            SimplestDragCoeff(num_nodes=nn),
            promotes_inputs=[('cl', Dynamic.Vehicle.LIFT_COEFFICIENT)],
            promotes_outputs=[('CD', Dynamic.Vehicle.DRAG_COEFFICIENT)],
        )

        self.add_subsystem(
            'SimpleDrag',
            SimpleDrag(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.DRAG_COEFFICIENT,
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                (Aircraft.Wing.AREA, 'wing_area'),
            ],
            promotes_outputs=[Dynamic.Vehicle.DRAG],
        )
