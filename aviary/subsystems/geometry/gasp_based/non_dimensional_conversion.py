import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class StrutCalcs(om.ExplicitComponent):
    """
    Given strut location as a non-dimensional function of wing half-span or distance from aircraft center, compute the other value not provided.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, val=0)

        if self.options["aviary_options"].get_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
            add_aviary_input(self, Aircraft.Strut.ATTACHMENT_LOCATION, val=0)
            add_aviary_output(
                self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0)
        else:
            add_aviary_input(
                self, Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, val=0)
            add_aviary_output(self, Aircraft.Strut.ATTACHMENT_LOCATION, val=0)

    def setup_partials(self):

        if self.options["aviary_options"].get_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
            self.declare_partials(
                Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, [Aircraft.Strut.ATTACHMENT_LOCATION, Aircraft.Wing.SPAN])
        else:
            self.declare_partials(
                Aircraft.Strut.ATTACHMENT_LOCATION, [Aircraft.Strut.ATTACHMENT_LOCATION_DIMENSIONLESS, Aircraft.Wing.SPAN])

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        strut_loc_name = Aircraft.Strut.ATTACHMENT_LOCATION

        if self.options["aviary_options"].get_val(Aircraft.Wing.HAS_STRUT, units='unitless'):
            if self.options["aviary_options"].get_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
                strut_x = inputs[strut_loc_name]
                outputs[strut_loc_name+"_dimensionless"] = strut_x / wing_span
            else:
                strut_x = inputs[strut_loc_name+"_dimensionless"]
                outputs[strut_loc_name] = strut_x * wing_span

    def compute_partials(self, inputs, partials):
        wing_span = inputs[Aircraft.Wing.SPAN]
        strut_loc_name = Aircraft.Strut.ATTACHMENT_LOCATION

        if self.options["aviary_options"].get_val(Aircraft.Wing.HAS_STRUT, units='unitless'):
            if self.options["aviary_options"].get_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
                partials[strut_loc_name+"_dimensionless", strut_loc_name] = 1 / wing_span
                partials[strut_loc_name+"_dimensionless",
                         Aircraft.Wing.SPAN] = - inputs[strut_loc_name] / wing_span**2
            else:
                partials[strut_loc_name, strut_loc_name+"_dimensionless"] = wing_span
                partials[strut_loc_name,
                         Aircraft.Wing.SPAN] = inputs[strut_loc_name+"_dimensionless"]


class FoldCalcs(om.ExplicitComponent):
    """
    Dimensional and non-dimensional conversion of fold calculation.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SPAN, val=0)

        if self.options["aviary_options"].get_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN, val=25)
            add_aviary_output(self, Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, val=0)
        else:
            add_aviary_input(self, Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, val=0)
            add_aviary_output(self, Aircraft.Wing.FOLDED_SPAN, val=0)

    def setup_partials(self):

        if self.options["aviary_options"].get_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
            self.declare_partials(
                Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, [Aircraft.Wing.FOLDED_SPAN, Aircraft.Wing.SPAN])
        else:
            self.declare_partials(
                Aircraft.Wing.FOLDED_SPAN, [Aircraft.Wing.FOLDED_SPAN_DIMENSIONLESS, Aircraft.Wing.SPAN])

    def compute(self, inputs, outputs):
        wing_span = inputs[Aircraft.Wing.SPAN]
        folded_span_name = Aircraft.Wing.FOLDED_SPAN

        if self.options["aviary_options"].get_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
            fold_y = inputs[folded_span_name]
            outputs[folded_span_name+"_dimensionless"] = fold_y / wing_span
        else:
            fold_y = inputs[folded_span_name+"_dimensionless"]
            outputs[folded_span_name] = fold_y * wing_span

    def compute_partials(self, inputs, partials):
        wing_span = inputs[Aircraft.Wing.SPAN]
        folded_span_name = Aircraft.Wing.FOLDED_SPAN

        if self.options["aviary_options"].get_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, units='unitless'):
            partials[folded_span_name+"_dimensionless", folded_span_name] = 1 / wing_span
            partials[folded_span_name+"_dimensionless", Aircraft.Wing.SPAN] = - \
                inputs[folded_span_name] / (wing_span**2)
        else:
            partials[folded_span_name, folded_span_name+"_dimensionless"] = wing_span
            partials[folded_span_name,
                     Aircraft.Wing.SPAN] = inputs[folded_span_name+"_dimensionless"]


class DimensionalNonDimensionalInterchange(om.Group):
    """
    Dimensional and non-dimensional conversion of strut and fold calculation if any.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):

        aviary_options = self.options['aviary_options']

        if aviary_options.get_val(Aircraft.Wing.HAS_STRUT, units='unitless'):
            self.add_subsystem(
                "strut_calcs",
                StrutCalcs(aviary_options=aviary_options,),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )

        if aviary_options.get_val(Aircraft.Wing.HAS_FOLD, units='unitless'):
            self.add_subsystem(
                "fold_calcs",
                FoldCalcs(aviary_options=aviary_options,),
                promotes_inputs=["aircraft:*"],
                promotes_outputs=["aircraft:*"],
            )
