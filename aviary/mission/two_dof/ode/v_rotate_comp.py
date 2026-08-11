import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft


class VRotateComp(om.ExplicitComponent):
    """
    Component that computes V_rotate based on vehicle properties and speed buffers.
    NOTE: This component is not used.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA)

        self.add_input('CL_max', shape=(1,), units='unitless', desc='Maximum lift coefficient')
        self.add_input('mass', shape=(1,), units='lbm', desc='Vehicle mass at rotation point.')
        self.add_input('density', shape=(1,), units='slug/ft**3', desc='Density at rotation point.')
        self.add_input(
            'dV1',
            shape=(1,),
            units='ft/s',
            desc='Increment of engine failure decision speed above stall speed.',
        )
        self.add_input(
            'dVR',
            shape=(1,),
            units='ft/s',
            desc='Increment of takeoff rotation speed above engine failure decision speed.',
        )

        self.add_output(
            'Vrot',
            shape=(1,),
            units='ft/s',
            desc='Speed at which takeoff rotation should be initiated.',
        )

        # Constant partials
        self.declare_partials(of='Vrot', wrt=['dV1', 'dVR'], val=1.0)
        # Partials of nonlinear terms
        self.declare_partials(
            of='Vrot',
            wrt=[
                'mass',
                'density',
                Aircraft.Wing.AREA,
                'CL_max',
            ],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        rho = inputs['density']
        wing_area = inputs[Aircraft.Wing.AREA]
        mass = inputs['mass']
        CL_max = inputs['CL_max']
        dV1 = inputs['dV1']
        dVR = inputs['dVR']

        outputs['Vrot'] = (
            ((2 * mass * GRAV_ENGLISH_LBM) / (rho * wing_area * CL_max)) ** 0.5 + dV1 + dVR
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        rho = inputs['density']
        wing_area = inputs[Aircraft.Wing.AREA]
        mass = inputs['mass']
        CL_max = inputs['CL_max']
        dV1 = inputs['dV1']
        dVR = inputs['dVR']

        K = 0.5 * ((2 * mass * GRAV_ENGLISH_LBM) / (rho * wing_area * CL_max)) ** 0.5

        partials['Vrot', 'mass'] = K / mass
        partials['Vrot', 'density'] = -K / rho
        partials['Vrot', Aircraft.Wing.AREA] = -K / wing_area
        partials['Vrot', 'CL_max'] = -K / CL_max
