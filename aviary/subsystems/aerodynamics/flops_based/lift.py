import numpy as np
import openmdao.api as om

import aviary.constants as constants
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic

grav_metric = constants.GRAV_METRIC_FLOPS


class SimpleLift(om.ExplicitComponent):
    """Calculate lift as a function of wing area, dynamic pressure, and lift coefficient."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')

        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

        self.add_input(
            name='cl', val=np.ones(nn), desc='current coefficient of lift', units='unitless'
        )

        add_aviary_output(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(Dynamic.Vehicle.LIFT, Aircraft.Wing.AREA)

        self.declare_partials(
            Dynamic.Vehicle.LIFT,
            [Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'cl'],
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CL = inputs['cl']

        outputs[Dynamic.Vehicle.LIFT] = q * S * CL

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CL = inputs['cl']

        partials[Dynamic.Vehicle.LIFT, Aircraft.Wing.AREA] = q * CL
        partials[Dynamic.Vehicle.LIFT, Dynamic.Atmosphere.DYNAMIC_PRESSURE] = S * CL
        partials[Dynamic.Vehicle.LIFT, 'cl'] = q * S


class LiftEqualsWeight(om.ExplicitComponent):
    """
    Compute lift given aircraft mass.

    This is valid for non-accelerating phases only.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')

        add_aviary_input(self, Dynamic.Vehicle.MASS, shape=nn, units='kg')

        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

        self.add_output(
            name='cl', val=np.ones(nn), desc='current coefficient of lift', units='unitless'
        )

        add_aviary_output(self, Dynamic.Vehicle.LIFT, shape=nn, units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        row_col = np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.LIFT, Dynamic.Vehicle.MASS, rows=row_col, cols=row_col, val=grav_metric
        )

        self.declare_partials(
            Dynamic.Vehicle.LIFT,
            [Aircraft.Wing.AREA, Dynamic.Atmosphere.DYNAMIC_PRESSURE],
            dependent=False,
        )

        self.declare_partials('cl', Aircraft.Wing.AREA)

        self.declare_partials(
            'cl',
            [Dynamic.Vehicle.MASS, Dynamic.Atmosphere.DYNAMIC_PRESSURE],
            rows=row_col,
            cols=row_col,
        )

    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        weight = grav_metric * inputs[Dynamic.Vehicle.MASS]

        outputs['cl'] = weight / (q * S)

        outputs[Dynamic.Vehicle.LIFT] = weight

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        weight = grav_metric * inputs[Dynamic.Vehicle.MASS]

        f = weight / q
        # df = 0.
        g = S
        # dg = 1.
        partials['cl', Aircraft.Wing.AREA] = -f / g**2

        partials['cl', Dynamic.Vehicle.MASS] = grav_metric / (q * S)

        f = weight / S
        # df = 0.
        g = q
        # dg = 1.
        partials['cl', Dynamic.Atmosphere.DYNAMIC_PRESSURE] = -f / g**2
