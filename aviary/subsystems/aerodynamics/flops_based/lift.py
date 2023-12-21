import numpy as np
import openmdao.api as om

import aviary.constants as constants
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic

grav_metric = constants.GRAV_METRIC_FLOPS


class SimpleLift(om.ExplicitComponent):
    '''
    Calculate lift as a function of wing area, dynamic pressure, and lift coefficient.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, val=1., units='m**2')

        self.add_input(
            Dynamic.Mission.DYNAMIC_PRESSURE, val=np.ones(nn), units='N/m**2',
            desc='pressure caused by fluid motion')

        self.add_input(
            name='cl', val=np.ones(nn), desc='current coefficient of lift',
            units='unitless')

        self.add_output(name=Dynamic.Mission.LIFT,
                        val=np.ones(nn), desc='Lift', units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(Dynamic.Mission.LIFT, Aircraft.Wing.AREA)

        self.declare_partials(
            Dynamic.Mission.LIFT, [Dynamic.Mission.DYNAMIC_PRESSURE, 'cl'], rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Mission.DYNAMIC_PRESSURE]
        CL = inputs['cl']

        outputs[Dynamic.Mission.LIFT] = q * S * CL

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Mission.DYNAMIC_PRESSURE]
        CL = inputs['cl']

        partials[Dynamic.Mission.LIFT, Aircraft.Wing.AREA] = q * CL
        partials[Dynamic.Mission.LIFT, Dynamic.Mission.DYNAMIC_PRESSURE] = S * CL
        partials[Dynamic.Mission.LIFT, 'cl'] = q * S


class LiftEqualsWeight(om.ExplicitComponent):
    """
    Compute lift given aircraft mass.

    This is valid for non-accelerating phases only.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, varname=Aircraft.Wing.AREA, val=1.0, units='m**2')

        self.add_input(
            name=Dynamic.Mission.MASS, val=np.ones(nn), desc='current aircraft mass',
            units='kg')

        self.add_input(
            Dynamic.Mission.DYNAMIC_PRESSURE, val=np.ones(nn), units='N/m**2',
            desc='pressure caused by fluid motion')

        self.add_output(
            name='cl', val=np.ones(nn), desc='current coefficient of lift',
            units='unitless')

        self.add_output(name=Dynamic.Mission.LIFT,
                        val=np.ones(nn), desc='Lift', units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        row_col = np.arange(nn)

        self.declare_partials(
            Dynamic.Mission.LIFT, Dynamic.Mission.MASS, rows=row_col, cols=row_col, val=grav_metric)

        self.declare_partials(
            Dynamic.Mission.LIFT, [Aircraft.Wing.AREA, Dynamic.Mission.DYNAMIC_PRESSURE], dependent=False)

        self.declare_partials('cl', Aircraft.Wing.AREA)

        self.declare_partials(
            'cl', [Dynamic.Mission.MASS, Dynamic.Mission.DYNAMIC_PRESSURE], rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Mission.DYNAMIC_PRESSURE]
        weight = grav_metric * inputs[Dynamic.Mission.MASS]

        outputs['cl'] = weight / (q * S)

        outputs[Dynamic.Mission.LIFT] = weight

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Mission.DYNAMIC_PRESSURE]
        weight = grav_metric * inputs[Dynamic.Mission.MASS]

        f = weight / q
        # df = 0.
        g = S
        # dg = 1.
        partials['cl', Aircraft.Wing.AREA] = -f / g**2

        partials['cl', Dynamic.Mission.MASS] = grav_metric / (q * S)

        f = weight / S
        # df = 0.
        g = q
        # dg = 1.
        partials['cl', Dynamic.Mission.DYNAMIC_PRESSURE] = -f / g**2
