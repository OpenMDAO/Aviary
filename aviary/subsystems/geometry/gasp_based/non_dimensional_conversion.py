import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft


class StrutCalcs(om.ExplicitComponent):

    def initialize(self):
        add_aviary_option(self, Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, val=0)

        if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
            add_aviary_input(self, Aircraft.Strut.ATTACHMENT_LOCATION, val=0)
            add_aviary_output(
                self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0)
        else:
            add_aviary_input(
                self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0)
            add_aviary_output(self, Aircraft.Strut.ATTACHMENT_LOCATION, val=0)

    def setup_partials(self):

        if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
            self.declare_partials(
                Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, [Aircraft.Strut.ATTACHMENT_LOCATION, Aircraft.Wing.SPAN])
        else:
            self.declare_partials(
                Aircraft.Strut.ATTACHMENT_LOCATION, [Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, Aircraft.Wing.SPAN])

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        strut_loc_name = Aircraft.Strut.ATTACHMENT_LOCATION

        if self.options[Aircraft.Wing.HAS_STRUT]:
            if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
                strut_x = inputs[strut_loc_name]
                outputs[strut_loc_name+"_dimensionless"] = strut_x / wing_span
            else:
                strut_x = inputs[strut_loc_name+"_dimensionless"]
                outputs[strut_loc_name] = strut_x * wing_span

    def compute_partials(self, inputs, partials):
        wing_span = inputs[Aircraft.Wing.SPAN]
        strut_loc_name = Aircraft.Strut.ATTACHMENT_LOCATION

        if self.options[Aircraft.Wing.HAS_STRUT]:
            if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
                partials[strut_loc_name+"_dimensionless", strut_loc_name] = 1 / wing_span
                partials[strut_loc_name+"_dimensionless",
                         Aircraft.Wing.SPAN] = - inputs[strut_loc_name] / wing_span**2
            else:
                partials[strut_loc_name, strut_loc_name+"_dimensionless"] = wing_span
                partials[strut_loc_name,
                         Aircraft.Wing.SPAN] = inputs[strut_loc_name+"_dimensionless"]


class FoldCalcs(om.ExplicitComponent):

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, val=0)

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN, val=0)
            add_aviary_output(self, Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, val=0)
        else:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, val=0)
            add_aviary_output(self, Aircraft.Wing.FOLDED_SPAN, val=0)

    def setup_partials(self):

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            self.declare_partials(
                Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, [Aircraft.Wing.FOLDED_SPAN, Aircraft.Wing.SPAN])
        else:
            self.declare_partials(
                Aircraft.Wing.FOLDED_SPAN, [Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, Aircraft.Wing.SPAN])

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        folded_span_name = Aircraft.Wing.FOLDED_SPAN

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            fold_y = inputs[folded_span_name]
            outputs[folded_span_name+"_dimensionless"] = fold_y / wing_span
        else:
            fold_y = inputs[folded_span_name+"_dimensionless"]
            outputs[folded_span_name] = fold_y * wing_span

    def compute_partials(self, inputs, partials):
        wing_span = inputs[Aircraft.Wing.SPAN]
        folded_span_name = Aircraft.Wing.FOLDED_SPAN

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            partials[folded_span_name+"_dimensionless", folded_span_name] = 1 / wing_span
            partials[folded_span_name+"_dimensionless", Aircraft.Wing.SPAN] = - \
                inputs[folded_span_name] / (wing_span**2)
        else:
            partials[folded_span_name, folded_span_name+"_dimensionless"] = wing_span
            partials[folded_span_name,
                     Aircraft.Wing.SPAN] = inputs[folded_span_name+"_dimensionless"]


class DimensionalNonDimensionalInterchange(om.Group):

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):

        if self.options[Aircraft.Wing.HAS_STRUT]:
            self.add_subsystem(
                "strut_calcs",
                StrutCalcs(),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )

        if self.options[Aircraft.Wing.HAS_FOLD]:
            self.add_subsystem(
                "fold_calcs",
                FoldCalcs(),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )
