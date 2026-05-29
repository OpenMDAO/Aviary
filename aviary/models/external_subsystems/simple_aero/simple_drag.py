import numpy as np
import openmdao.api as om

from aviary.subsystems.aerodynamics.aero_common import DynamicPressure
from aviary.subsystems.aerodynamics.flops_based.drag import SimpleDrag
from aviary.subsystems.aerodynamics.flops_based.lift import LiftEqualsWeight
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic


class SimpleDragCoeff(om.ExplicitComponent):
    """Simple representation of aircraft drag as parabolic equation CD = CD_zero + k * CL**2."""

    def initialize(self):
        self.options.declare(
            'num_nodes', default=1, types=int, desc='Number of nodes along mission segment'
        )

        self.options.declare('CD_zero', default=0.01, desc='Zero-lift drag coefficient')
        self.options.declare('k', default=0.04, desc='Induced drag factor')

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Dynamic.Vehicle.LIFT_COEFFICIENT, units='unitless', shape=nn)
        add_aviary_output(self, Dynamic.Vehicle.DRAG_COEFFICIENT, units='unitless', shape=nn)

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            Dynamic.Vehicle.LIFT_COEFFICIENT,
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        CD_zero = self.options['CD_zero']
        k = self.options['k']

        cl = inputs[Dynamic.Vehicle.LIFT_COEFFICIENT]

        outputs[Dynamic.Vehicle.DRAG_COEFFICIENT] = CD_zero + k * cl**2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        k = self.options['k']

        cl = inputs[Dynamic.Vehicle.LIFT_COEFFICIENT]

        partials[Dynamic.Vehicle.DRAG_COEFFICIENT, Dynamic.Vehicle.LIFT_COEFFICIENT] = 2.0 * k * cl


class SimpleAeroGroup(om.Group):
    def initialize(self):
        self.options.declare(
            'num_nodes', default=1, types=int, desc='Number of nodes along mission segment'
        )

    def setup(self):
        nn = self.options['num_nodes']

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
                Aircraft.Wing.AREA,
                Dynamic.Vehicle.MASS,
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            ],
            promotes_outputs=[Dynamic.Vehicle.LIFT_COEFFICIENT, Dynamic.Vehicle.LIFT],
        )

        self.add_subsystem(
            'SimpleDragCoeff',
            SimpleDragCoeff(num_nodes=nn),
            promotes_inputs=[(Dynamic.Vehicle.LIFT_COEFFICIENT, Dynamic.Vehicle.LIFT_COEFFICIENT)],
            promotes_outputs=[Dynamic.Vehicle.DRAG_COEFFICIENT],
        )

        self.add_subsystem(
            'SimpleDrag',
            SimpleDrag(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.DRAG_COEFFICIENT,
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                Aircraft.Wing.AREA,
            ],
            promotes_outputs=[Dynamic.Vehicle.DRAG],
        )
