import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variable_meta_data import _MetaData as _meta_data
from aviary.variable_info.variables import Aircraft, Dynamic


class SimpleCD(om.ExplicitComponent):
    """
    Apply the final drag coefficient factors to the unscaled drag.

    These optional factors (default: 1.0) increase or decrease the drag
    coefficient before calculating drag.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, units='unitless')

        add_aviary_input(self, Dynamic.Atmosphere.MACH, shape=nn, units='unitless')

        self.add_input(
            'CD_prescaled', val=np.ones(nn), units='unitless', desc='total drag coefficient'
        )

        self.add_output('CD', val=np.ones(nn), units='unitless', desc='total drag')

    def setup_partials(self):
        self.declare_partials(
            'CD',
            [
                'CD_prescaled',
                Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
            ],
        )

        self.declare_partials('CD', Dynamic.Atmosphere.MACH, dependent=False)

        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials('CD', 'CD_prescaled', rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs):
        FCDSUB = inputs[Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR]
        FCDSUP = inputs[Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR]
        M = inputs[Dynamic.Atmosphere.MACH]

        CD_prescaled = inputs['CD_prescaled']

        idx_sup = np.where(M >= 1.0)
        CD_scaled = CD_prescaled * FCDSUB
        CD_scaled[idx_sup] = CD_prescaled[idx_sup] * FCDSUP
        outputs['CD'] = CD_scaled

    def compute_partials(self, inputs, partials):
        FCDSUB = inputs[Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR]
        FCDSUP = inputs[Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR]
        M = inputs[Dynamic.Atmosphere.MACH]
        CD_prescaled = inputs['CD_prescaled']

        idx_sup = np.where(M >= 1.0)
        CD_scaled = CD_prescaled * FCDSUB
        CD_scaled[idx_sup] = CD_prescaled[idx_sup] * FCDSUP

        idx_sub = np.where(M < 1.0)
        dCD = np.ones_like(CD_prescaled)
        dCD[idx_sub] = FCDSUB
        dCD[idx_sup] = FCDSUP
        partials['CD', 'CD_prescaled'] = dCD

        dF = np.zeros_like(CD_prescaled)
        dF[idx_sub] = CD_prescaled[idx_sub]
        partials['CD', Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR] = dF

        dF = np.zeros_like(CD_prescaled)
        dF[idx_sup] = CD_prescaled[idx_sup]
        partials['CD', Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR] = dF


class SimpleDrag(om.ExplicitComponent):
    """Calculate drag as a function of wing area, dynamic pressure, and drag coefficient."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')

        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

        self.add_input('CD', val=np.ones(nn), units='unitless', desc='total drag coefficient')

        add_aviary_output(self, Dynamic.Vehicle.DRAG, shape=nn, units='N')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(Dynamic.Vehicle.DRAG, Aircraft.Wing.AREA)

        self.declare_partials(
            Dynamic.Vehicle.DRAG,
            [Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'CD'],
            rows=rows_cols,
            cols=rows_cols,
        )

    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CD = inputs['CD']

        outputs[Dynamic.Vehicle.DRAG] = q * S * CD

    def compute_partials(self, inputs, partials):
        S = inputs[Aircraft.Wing.AREA]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        CD = inputs['CD']

        partials[Dynamic.Vehicle.DRAG, Aircraft.Wing.AREA] = q * CD
        partials[Dynamic.Vehicle.DRAG, Dynamic.Atmosphere.DYNAMIC_PRESSURE] = S * CD
        partials[Dynamic.Vehicle.DRAG, 'CD'] = q * S


class TotalDrag(om.Group):
    """
    Calculate drag as a function of wing area, dynamic pressure, and lift-dependent and
    lift-independent drag coefficients.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    dependent drag coefficient before calculating the total drag coefficient.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    independent drag coefficient before calculating the total drag coefficient.

    Note, the lift-dependent drag coefficient includes contributions from the pressure
    drag coefficient.

    Apply optional factors (default: 1.0) for increasing or decreasing the total drag
    coefficient before calculating drag. The effect is cumulative with the above
    optional factors.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        FCDI_desc = _meta_data[Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR]['desc']
        FCD0_desc = _meta_data[Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR]['desc']

        kwargs = {
            'CDI': dict(
                val=np.ones(nn),
                units='unitless',
                desc='lift-dependent drag coefficient,'
                ' including contributions from pressure drag coefficient',
            ),
            'CD0': dict(
                val=np.ones(nn), units='unitless', desc='lift-independent drag coefficient'
            ),
            'FCDI': dict(val=1.0, units='unitless', desc=FCDI_desc),
            'FCD0': dict(val=1.0, units='unitless', desc=FCD0_desc),
            'CD_prescaled': dict(val=np.ones(nn), units='unitless', desc='total drag coefficient'),
        }

        total_drag_comp = self.add_subsystem(
            'total_drag_coeff',
            om.ExecComp('CD_prescaled = CDI * FCDI + CD0 * FCD0', **kwargs),
            promotes_inputs=[
                'CDI',
                'CD0',
                ('FCDI', Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR),
                ('FCD0', Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR),
            ],
            promotes_outputs=['*'],
        )
        total_drag_comp.declare_coloring(show_summary=False)

        self.add_subsystem('simple_CD', SimpleCD(num_nodes=nn), promotes=['*'])
        self.add_subsystem('simple_drag', SimpleDrag(num_nodes=nn), promotes=['*'])
