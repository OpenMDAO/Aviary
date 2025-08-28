import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class StrutCalcs(om.ExplicitComponent):
    """Given strut location as a non-dimensional function of wing half-span or distance from aircraft center, compute the other value not provided."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
            add_aviary_input(self, Aircraft.Strut.ATTACHMENT_LOCATION, units='ft')
            add_aviary_output(
                self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, units='unitless'
            )
        else:
            add_aviary_input(
                self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, units='unitless'
            )
            add_aviary_output(self, Aircraft.Strut.ATTACHMENT_LOCATION, units='ft')

    def setup_partials(self):
        if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
            self.declare_partials(
                Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS,
                [Aircraft.Strut.ATTACHMENT_LOCATION, Aircraft.Wing.SPAN],
            )
        else:
            self.declare_partials(
                Aircraft.Strut.ATTACHMENT_LOCATION,
                [Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, Aircraft.Wing.SPAN],
            )

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        strut_loc_name = Aircraft.Strut.ATTACHMENT_LOCATION

        if self.options[Aircraft.Wing.HAS_STRUT]:
            if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
                strut_x = inputs[strut_loc_name]
                outputs[strut_loc_name + '_dimensionless'] = strut_x / wing_span
            else:
                strut_x = inputs[strut_loc_name + '_dimensionless']
                outputs[strut_loc_name] = strut_x * wing_span

    def compute_partials(self, inputs, partials):
        wing_span = inputs[Aircraft.Wing.SPAN]
        strut_loc_name = Aircraft.Strut.ATTACHMENT_LOCATION

        if self.options[Aircraft.Wing.HAS_STRUT]:
            if self.options[Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED]:
                partials[strut_loc_name + '_dimensionless', strut_loc_name] = 1 / wing_span
                partials[strut_loc_name + '_dimensionless', Aircraft.Wing.SPAN] = (
                    -inputs[strut_loc_name] / wing_span**2
                )
            else:
                partials[strut_loc_name, strut_loc_name + '_dimensionless'] = wing_span
                partials[strut_loc_name, Aircraft.Wing.SPAN] = inputs[
                    strut_loc_name + '_dimensionless'
                ]


class FoldCalcs(om.ExplicitComponent):
    """Dimensional and non-dimensional conversion of fold calculation."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED)

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN, units='ft')
            add_aviary_output(self, Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, units='unitless')
        else:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, units='unitless')
            add_aviary_output(self, Aircraft.Wing.FOLDED_SPAN, units='ft')

    def setup_partials(self):
        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            self.declare_partials(
                Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS,
                [Aircraft.Wing.FOLDED_SPAN, Aircraft.Wing.SPAN],
            )
        else:
            self.declare_partials(
                Aircraft.Wing.FOLDED_SPAN,
                [Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, Aircraft.Wing.SPAN],
            )

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        folded_span_name = Aircraft.Wing.FOLDED_SPAN

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            fold_y = inputs[folded_span_name]
            outputs[folded_span_name + '_dimensionless'] = fold_y / wing_span
        else:
            fold_y = inputs[folded_span_name + '_dimensionless']
            outputs[folded_span_name] = fold_y * wing_span

    def compute_partials(self, inputs, partials):
        wing_span = inputs[Aircraft.Wing.SPAN]
        folded_span_name = Aircraft.Wing.FOLDED_SPAN

        if self.options[Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED]:
            partials[folded_span_name + '_dimensionless', folded_span_name] = 1 / wing_span
            partials[folded_span_name + '_dimensionless', Aircraft.Wing.SPAN] = -inputs[
                folded_span_name
            ] / (wing_span**2)
        else:
            partials[folded_span_name, folded_span_name + '_dimensionless'] = wing_span
            partials[folded_span_name, Aircraft.Wing.SPAN] = inputs[
                folded_span_name + '_dimensionless'
            ]


class DimensionalNonDimensionalInterchange(om.Group):
    """Dimensional and non-dimensional conversion of strut and fold calculation if any."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Wing.HAS_FOLD)
        add_aviary_option(self, Aircraft.Wing.HAS_STRUT)

    def setup(self):
        if self.options[Aircraft.Wing.HAS_STRUT]:
            self.add_subsystem(
                'strut_calcs',
                StrutCalcs(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        if self.options[Aircraft.Wing.HAS_FOLD]:
            self.add_subsystem(
                'fold_calcs',
                FoldCalcs(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
