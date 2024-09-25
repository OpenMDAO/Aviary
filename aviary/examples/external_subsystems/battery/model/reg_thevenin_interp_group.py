import numpy as np

from openmdao.api import Group, MetaModelStructuredComp


# These battery values are entirely fabricated and not representative of any real battery.
# This model is used as an example of how to integrate external subsystems in Aviary.
T_bp = np.array([0., 60.])
SOC_bp = np.linspace(0., 1., 32)

tU_oc = np.vstack((np.linspace(2.8, 4.2, 32), np.linspace(2.7, 4.1, 32)))
tC_Th = np.ones((2, 32)) * 1000.0
tR_Th = np.vstack((np.linspace(0.06, 0.03, 32), np.linspace(0.04, 0.02, 32)))
tR_0 = np.vstack((np.linspace(0.06, 0.03, 32), np.linspace(0.04, 0.02, 32)))


class RegTheveninInterpGroup(Group):
    """
    Thevenin resistance and voltage computation by interpolation using temperature,
    state of charge, and capacitance.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        thevenin_interp_comp = MetaModelStructuredComp(method='slinear', vec_size=n,
                                                       training_data_gradients=False, extrapolate=True)

        thevenin_interp_comp.add_input('T_batt', 1.0, T_bp, units='degC')
        thevenin_interp_comp.add_input('SOC', 1.0, SOC_bp, units=None)
        thevenin_interp_comp.add_output('C_Th', 1.0, tC_Th, units='F')
        # add R_th and R_0 to get total R
        thevenin_interp_comp.add_output('R_Th', 1.0, tR_Th, units='ohm')
        thevenin_interp_comp.add_output('R_0', 1.0, tR_0, units='ohm')
        thevenin_interp_comp.add_output('U_oc', 1.0, tU_oc, units='V')

        self.add_subsystem(name='interp_comp',
                           subsys=thevenin_interp_comp,
                           promotes=["*"])  # promots params and values
