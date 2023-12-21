import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variable_meta_data import _MetaData as _meta_data
from aviary.variable_info.variables import Aircraft, Dynamic


class SimpleDrag(om.ExplicitComponent):
    '''
    Calculate drag as a function of wing area, dynamic pressure, and drag coefficient.

    Apply optional factors (default: 1.0) for increasing or decreasing the drag
    coefficient before calculating drag.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Wing.AREA, val=1., units='m**2')
        add_aviary_input(self, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, val=1.)
        add_aviary_input(self, Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, val=1.)

        self.add_input(
            Dynamic.Mission.MACH, val=np.ones(nn), units='unitless',
            desc='ratio of local fluid speed to local speed of sound')

        self.add_input(
            Dynamic.Mission.DYNAMIC_PRESSURE, val=np.ones(nn), units='N/m**2',
            desc='pressure caused by fluid motion')

        self.add_input(
            'drag_coefficient', val=np.ones(nn), units='unitless',
            desc='total drag coefficient')

        self.add_output(Dynamic.Mission.DRAG, val=np.ones(nn),
                        units='N', desc='total drag')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(
            Dynamic.Mission.DRAG,
            [
                Aircraft.Wing.AREA,
                Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR])

        self.declare_partials(Dynamic.Mission.DRAG,
                              Dynamic.Mission.MACH, dependent=False)

        self.declare_partials(
            Dynamic.Mission.DRAG,
            [Dynamic.Mission.DYNAMIC_PRESSURE, 'drag_coefficient'],
            rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs):
        S = inputs[Aircraft.Wing.AREA]
        FCDSUB = inputs[Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR]
        FCDSUP = inputs[Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR]
        M = inputs[Dynamic.Mission.MACH]
        q = inputs[Dynamic.Mission.DYNAMIC_PRESSURE]
        CD = inputs['drag_coefficient']

        idx_sup = np.where(M >= 1.0)
        CD_scaled = CD * FCDSUB
        CD_scaled[idx_sup] = CD[idx_sup] * FCDSUP

        outputs[Dynamic.Mission.DRAG] = q * S * CD_scaled

    def compute_partials(self, inputs, partials):
        S = inputs[Aircraft.Wing.AREA]
        FCDSUB = inputs[Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR]
        FCDSUP = inputs[Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR]
        M = inputs[Dynamic.Mission.MACH]
        q = inputs[Dynamic.Mission.DYNAMIC_PRESSURE]
        CD = inputs['drag_coefficient']

        idx_sup = np.where(M >= 1.0)
        CD_scaled = CD * FCDSUB
        CD_scaled[idx_sup] = CD[idx_sup] * FCDSUP

        partials[Dynamic.Mission.DRAG, Aircraft.Wing.AREA] = q * CD_scaled
        partials[Dynamic.Mission.DRAG, Dynamic.Mission.DYNAMIC_PRESSURE] = S * CD_scaled

        idx_sub = np.where(M < 1.0)
        dCD = q * S
        dCD[idx_sub] *= FCDSUB
        dCD[idx_sup] *= FCDSUP
        partials[Dynamic.Mission.DRAG, 'drag_coefficient'] = dCD

        drag_unscaled = q * S * CD
        dF = np.zeros(CD.shape)
        dF[idx_sub] = drag_unscaled[idx_sub]
        partials[Dynamic.Mission.DRAG, Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR] = dF

        dF = np.zeros(CD.shape)
        dF[idx_sup] = drag_unscaled[idx_sup]
        partials[Dynamic.Mission.DRAG, Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR] = dF


class TotalDragCoeff(om.ExplicitComponent):
    '''
    Calculate the total drag coefficient from lift-dependent and lift-independent drag
    coefficients.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    dependent drag coefficient before calculating the total drag coefficient.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    independent drag coefficient before calculating the total drag coefficient.

    Note, the lift-dependent drag coefficient includes contirbutions from the pressure
    drag coefficient.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.)
        add_aviary_input(self, Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.)

        self.add_input(
            'CDI', val=np.ones(nn), units='unitless',
            desc='lift-dependent drag coefficient,'
            ' including contributions from pressure drag coefficient')

        self.add_input(
            'CD0', val=np.ones(nn), units='unitless',
            desc='lift-independent drag coefficient')

        self.add_output(
            'drag_coefficient', val=np.ones(nn), units='unitless',
            desc='total drag coefficient')

    def setup_partials(self):
        nn = self.options['num_nodes']
        rows_cols = np.arange(nn)

        self.declare_partials(
            'drag_coefficient',
            [
                Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
                Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR])

        self.declare_partials(
            'drag_coefficient', ['CDI', 'CD0'], rows=rows_cols, cols=rows_cols)

    def compute(self, inputs, outputs):
        FCD0 = inputs[Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR]
        FCDI = inputs[Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR]
        CDI = inputs['CDI']
        CD0 = inputs['CD0']

        outputs['drag_coefficient'] = CDI * FCDI + CD0 * FCD0

    def compute_partials(self, inputs, partials):
        FCD0 = inputs[Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR]
        FCDI = inputs[Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR]
        CDI = inputs['CDI']
        CD0 = inputs['CD0']

        partials['drag_coefficient', Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR] = CD0

        partials[
            'drag_coefficient', Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR] = CDI

        partials['drag_coefficient', 'CDI'] = FCDI
        partials['drag_coefficient', 'CD0'] = FCD0


class TotalDrag(om.Group):
    '''
    Calculate drag as a function of wing area, dynamic pressure, and lift-dependent and
    lift-independent drag coefficients.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    dependent drag coefficient before calculating the total drag coefficient.

    Apply an optional factor (default: 1.0) for increasing or decreasing the lift-
    independent drag coefficient before calculating the total drag coefficient.

    Note, the lift-dependent drag coefficient includes contirbutions from the pressure
    drag coefficient.

    Apply optional factors (default: 1.0) for increasing or decreasing the total drag
    coefficient before calculating drag. The effect is cummulative with the above
    optional factors.
    '''

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        FCDI_desc = _meta_data[Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR]['desc']
        FCD0_desc = _meta_data[Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR]['desc']

        kwargs = {
            'CDI': dict(
                val=np.ones(nn), units='unitless',
                desc='lift-dependent drag coefficient,'
                ' including contributions from pressure drag coefficient'),
            'CD0': dict(
                val=np.ones(nn), units='unitless',
                desc='lift-independent drag coefficient'),
            'FCDI': dict(val=1., units='unitless', desc=FCDI_desc),
            'FCD0': dict(val=1., units='unitless', desc=FCD0_desc),
            'CD': dict(val=np.ones(nn), units='unitless', desc='total drag coefficient')
        }

        total_drag_comp = self.add_subsystem(
            'total_drag_coeff',
            om.ExecComp('CD = CDI * FCDI + CD0 * FCD0', **kwargs),
            promotes_inputs=[
                'CDI', 'CD0',
                ('FCDI', Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR),
                ('FCD0', Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR)],
            promotes_outputs=[('CD', 'drag_coefficient')])
        total_drag_comp.declare_coloring(show_summary=False)

        self.add_subsystem('simple_drag', SimpleDrag(num_nodes=nn), promotes=['*'])
