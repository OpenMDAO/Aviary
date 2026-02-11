import numpy as np
import openmdao.api as om

from aviary.variable_info.variables import Dynamic


class DistanceComp(om.ExplicitComponent):
    """Computes distance for a simple cruise phase."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            'time',
            val=np.ones(nn),
            units='s',
            desc='Vector of time points to compute distance.',
        )
        self.add_input(
            'cruise_distance_initial',
            val=0.0,
            units='NM',
            desc='Total distance at the start of the cruise phase.',
        )

        self.add_input(
            'TAS_cruise',
            val=0.0001 * np.ones(nn),
            units='NM/s',
            desc='Constant true airspeed at each point in cruise.'
        )

        self.add_output(
            Dynamic.Mission.DISTANCE,
            shape=(nn,),
            units='NM',
            desc='Computed distance at each point in the cruise phase.',
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        row_col = np.arange(nn)

        self.declare_partials(
            Dynamic.Mission.DISTANCE,
            'TAS_cruise',
            rows=row_col,
            cols=row_col,
        )
        self.declare_partials(
            Dynamic.Mission.DISTANCE,
            'cruise_distance_initial',
            rows=[0],
            cols=[0],
            val=1.0,
        )

        # Sparsity pattern includes a vertical row at i=0
        xtra_row = np.arange(nn - 1) + 1
        xtra_col = np.zeros(nn - 1)

        all_row = np.hstack((row_col, xtra_row))
        all_col = np.hstack((row_col, xtra_col))

        self.declare_partials(Dynamic.Mission.DISTANCE, 'time', rows=all_row, cols=all_col)

    def compute(self, inputs, outputs):
        v_x = inputs['TAS_cruise']
        r0 = inputs['cruise_distance_initial']
        t = inputs['time']
        t0 = t[0]

        outputs[Dynamic.Mission.DISTANCE] = r0 + v_x * (t - t0)

    def compute_partials(self, inputs, J):
        v_x = inputs['TAS_cruise']
        t = inputs['time']
        t0 = t[0]
        nn = self.options['num_nodes']

        J[Dynamic.Mission.DISTANCE, 'TAS_cruise'] = t - t0

        J[Dynamic.Mission.DISTANCE, 'time'][0] = 0.0
        J[Dynamic.Mission.DISTANCE, 'time'][1:nn] = v_x[1:]
        J[Dynamic.Mission.DISTANCE, 'time'][nn:] = -v_x[1:]
