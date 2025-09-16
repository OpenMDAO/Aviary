import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class EmptyMassMargin(om.ExplicitComponent):
    """Calculates the empty mass margin."""

    def setup(self):
        add_aviary_input(self, Aircraft.Propulsion.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.SYSTEMS_EQUIP_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS_MARGIN_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Design.EMPTY_MASS_MARGIN, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        struct_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        sys_eq_mass = inputs[Aircraft.Design.SYSTEMS_EQUIP_MASS]
        scaler = inputs[Aircraft.Design.EMPTY_MASS_MARGIN_SCALER]

        outputs[Aircraft.Design.EMPTY_MASS_MARGIN] = (
            prop_mass + struct_mass + sys_eq_mass
        ) * scaler

    def compute_partials(self, inputs, J):
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        struct_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        sys_eq_mass = inputs[Aircraft.Design.SYSTEMS_EQUIP_MASS]
        scaler = inputs[Aircraft.Design.EMPTY_MASS_MARGIN_SCALER]

        J[Aircraft.Design.EMPTY_MASS_MARGIN, Aircraft.Propulsion.MASS] = scaler
        J[Aircraft.Design.EMPTY_MASS_MARGIN, Aircraft.Design.STRUCTURE_MASS] = scaler
        J[Aircraft.Design.EMPTY_MASS_MARGIN, Aircraft.Design.SYSTEMS_EQUIP_MASS] = scaler
        J[Aircraft.Design.EMPTY_MASS_MARGIN, Aircraft.Design.EMPTY_MASS_MARGIN_SCALER] = (
            prop_mass + struct_mass + sys_eq_mass
        )
