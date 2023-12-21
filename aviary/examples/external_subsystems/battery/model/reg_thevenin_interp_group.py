import numpy as np

from openmdao.api import Group, MetaModelStructuredComp

from boring.src.model.maps.s18650_battery import battery


class RegTheveninInterpGroup(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self.options.declare('cell_type', types=str, default='18650')

    def setup(self):
        n = self.options['num_nodes']
        cell_type = self.options['cell_type']

        thevenin_interp_comp = MetaModelStructuredComp(method='slinear', vec_size=n,
                                                       training_data_gradients=False, extrapolate=True)

        thevenin_interp_comp.add_input('T_batt', 1.0, battery.T_bp, units='degC')
        thevenin_interp_comp.add_input('SOC', 1.0, battery.SOC_bp, units=None)
        thevenin_interp_comp.add_output('C_Th', 1.0, battery.tC_Th, units='F')
        # add R_th and R_0 to get total R
        thevenin_interp_comp.add_output('R_Th', 1.0, battery.tR_Th, units='ohm')
        thevenin_interp_comp.add_output('R_0', 1.0, battery.tR_0, units='ohm')
        thevenin_interp_comp.add_output('U_oc', 1.0, battery.tU_oc, units='V')

        self.add_subsystem(name='interp_comp',
                           subsys=thevenin_interp_comp,
                           promotes=["*"])  # promots params and values
