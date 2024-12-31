import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft


class TailVolCoef(om.ExplicitComponent):
    """GASP tail volume coefficient fallback calculation.

    This component can be used to compute a volume coefficient for either a horizontal
    or vertical tail. The volume coefficient is based on an empirical relationship
    using gross aircraft parameters such as fuselage length, wing area, etc. For a
    horizontal tail, the wing chord is input. For a vertical tail, the wing span is
    input.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )
        self.options.declare(
            "vertical",
            default=False,
            types=bool,
            desc=(
                "Use vertical tail volume coefficient relationship, "
                "otherwise assume a horizontal tail"
            ),
        )

    def setup(self):
        # coefficients used in the empirical equation
        if self.options["vertical"]:
            self.k = [0.07, 0.0434, 0.336]
        else:
            self.k = [0.43, 0.38, 0.85]

        add_aviary_input(self, Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0)

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, val=129.4)

        self.add_input("cab_w", 13.1, units="ft", desc="SWF: Cabin width")

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370)

        self.add_input(
            "wing_ref",
            12.615,
            units="ft",
            desc=(
                "CBARW | B: Wing reference parameter. Wing chord for a "
                "horizontal tail. Wing span for a vertical tail."
            ),
        )

        self.add_output(
            "vol_coef", units="unitless", desc="VBARH | VBARV: Tail volume coefficient")

    def setup_partials(self):
        self.declare_partials("vol_coef", "*")

    def compute(self, inputs, outputs):
        htail_loc, fus_len, cab_w, wing_area, wing_ref = inputs.values()
        k1, k2, k3 = self.k
        ch1 = k1 - k2 * htail_loc
        outputs["vol_coef"] = k3 * fus_len * cab_w**2 / (wing_area * wing_ref) + ch1

    def compute_partials(self, inputs, J):
        htail_loc, fus_len, cab_w, wing_area, wing_ref = inputs.values()
        k1, k2, k3 = self.k
        J["vol_coef", Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION] = -k2
        J["vol_coef", Aircraft.Fuselage.LENGTH] = (
            k3 * cab_w**2 / (wing_area * wing_ref)
        )
        J["vol_coef", "cab_w"] = 2 * k3 * fus_len * cab_w / (wing_area * wing_ref)
        J["vol_coef", Aircraft.Wing.AREA] = (
            -k3 * fus_len * cab_w**2 / (wing_area**2 * wing_ref)
        )
        J["vol_coef", "wing_ref"] = (
            -k3 * fus_len * cab_w**2 / (wing_area * wing_ref**2)
        )


class TailSize(om.ExplicitComponent):
    """GASP tail geometry calculations.

    This component can be used for either a horizontal tail or vertical tail. For a
    horizontal tail, the ratio of wing chord to tail moment arm and the wing chord are
    input for tail moment arm calculation. For a vertical tail, the ratio of wing span
    to tail moment arm and the wing span are input.
    """

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        # defaults here for Large Single Aisle 1 horizontal tail
        self.add_input(
            "vol_coef", 1.189, units="unitless",
            desc="VBARH | VBARV: Horizontal tail volume coefficient"
        )

        add_aviary_input(self, Aircraft.Wing.AREA, val=1370)

        self.add_input(
            "r_arm",
            0.2307,
            units="unitless",
            desc=(
                "COELTH | BOELTV: For a horizontal tail, the ratio of "
                "wing chord to tail moment arm. For a vertical tail, the "
                "ratio of wing span to vertical tail moment arm."
            ),
        )
        self.add_input(
            "wing_ref",
            12.615,
            units="ft",
            desc=(
                "CBARW | B: Reference wing parameter for tail moment arm. "
                "For a horizontal tail, the mean wing chord. For a "
                "vertical tail, the wing span."
            ),
        )
        self.add_input(
            "ar", 4.75, units="unitless", desc="ARHT | ARVT: Tail aspect ratio.")
        self.add_input(
            "tr", 0.352, units="unitless", desc="SLMH | SLMV: Tail taper ratio.")

        self.add_output("area", units="ft**2", desc="SHT | SVT: Tail area")
        self.add_output("span", units="ft", desc="BHT | BVT: Tail span")
        self.add_output("rchord", units="ft", desc="CRCLHT | CRCLVT: Tail root chord")
        self.add_output(
            "chord", units="ft", desc="CBARHT | CBARVT: Tail mean aerodynamic chord"
        )
        self.add_output("arm", units="ft", desc="ELTH | ELTV: Tail moment arm")

    def setup_partials(self):
        self.declare_partials("area", ["vol_coef", Aircraft.Wing.AREA, "r_arm"])
        self.declare_partials("span", ["vol_coef", Aircraft.Wing.AREA, "r_arm", "ar"])
        self.declare_partials(
            "rchord", ["vol_coef", Aircraft.Wing.AREA, "r_arm", "ar", "tr"]
        )
        self.declare_partials(
            "chord", ["vol_coef", Aircraft.Wing.AREA, "r_arm", "ar", "tr"]
        )
        self.declare_partials("arm", ["r_arm", "wing_ref"])

    def compute(self, inputs, outputs):
        vol_coef, wing_area, r_arm, wing_ref, ar, tr = inputs.values()

        area = vol_coef * wing_area * r_arm
        span = np.sqrt(area * ar)
        rchord = 2 * area / span / (1 + tr)
        chord = 2 / 3.0 * rchord * ((1 + tr) - (tr / (1 + tr)))
        arm = wing_ref / r_arm

        outputs["area"] = area
        outputs["span"] = span
        outputs["rchord"] = rchord
        outputs["chord"] = chord
        outputs["arm"] = arm

    def compute_partials(self, inputs, J):
        vol_coef, wing_area, r_arm, wing_ref, ar, tr = inputs.values()

        J["area", "vol_coef"] = wing_area * r_arm
        J["area", Aircraft.Wing.AREA] = vol_coef * r_arm
        J["area", "r_arm"] = vol_coef * wing_area

        cse1 = np.sqrt(vol_coef * wing_area * r_arm * ar)
        J["span", "vol_coef"] = cse1 / (2 * vol_coef)
        J["span", Aircraft.Wing.AREA] = cse1 / (2 * wing_area)
        J["span", "r_arm"] = cse1 / (2 * r_arm)
        J["span", "ar"] = cse1 / (2 * ar)

        cse2 = cse1 * (tr + 1)
        J["rchord", "vol_coef"] = wing_area * r_arm / cse2
        J["rchord", Aircraft.Wing.AREA] = vol_coef * r_arm / cse2
        J["rchord", "r_arm"] = wing_area * vol_coef / cse2
        J["rchord", "ar"] = -vol_coef * wing_area * r_arm / (ar * cse2)
        J["rchord", "tr"] = -2 * vol_coef * wing_area * r_arm / (cse2 * (tr + 1))

        cse3 = tr - (tr / (tr + 1)) + 1
        J["chord", "vol_coef"] = 2 / 3.0 * wing_area * r_arm * cse3 / cse2
        J["chord", Aircraft.Wing.AREA] = 2 / 3.0 * vol_coef * r_arm * cse3 / cse2
        J["chord", "r_arm"] = 2 / 3.0 * vol_coef * wing_area * cse3 / cse2
        J["chord", "ar"] = -2 / 3.0 * vol_coef * wing_area * r_arm * cse3 / (ar * cse2)
        J["chord", "tr"] = 4 / 3.0 * cse1 * (tr - 1) / (ar * (tr + 1) ** 3)

        J["arm", "r_arm"] = -wing_ref / r_arm**2
        J["arm", "wing_ref"] = 1.0 / r_arm


class EmpennageSize(om.Group):
    """GASP geometry calculations for both horizontal and vertical tails.

    Volume coefficients for the tails may be either specified directly (default) or
    computed via empirical relationships to general airplane parameters.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        # TODO: For cruciform/T-tail configurations, GASP checks to make sure the V tail
        # chord at the H tail location is greater than the H tail root chord. If not, it
        # overrides the H tail taper ratio so they match. If that leads to a H tail root
        # chord greater than the H tail tip chord, it sets the taper ratio to 1 and
        # overrides the H tail aspect ratio. H tail taper ratio is used in landing gear
        # mass calculation.

        aviary_options = self.options['aviary_options']

        # higher inputs that are input to groups other than this one, or calculated in groups other than this one
        higher_level_inputs_htail_vc = [
            ("cab_w", Aircraft.Fuselage.AVG_DIAMETER),
            ("wing_ref", Aircraft.Wing.AVERAGE_CHORD),
        ]
        higher_level_inputs_vtail_vc = [
            ("cab_w", Aircraft.Fuselage.AVG_DIAMETER),
            ("wing_ref", Aircraft.Wing.SPAN),
        ]
        higher_level_inputs_htail = [
            ("wing_ref", Aircraft.Wing.AVERAGE_CHORD),
            ("tr", Aircraft.HorizontalTail.TAPER_RATIO),
        ]
        higher_level_inputs_vtail = [
            ("wing_ref", Aircraft.Wing.SPAN),
            ("ar", Aircraft.VerticalTail.ASPECT_RATIO),
        ]

        # inputs to htail and vtail that are purely promoted to change the name
        rename_inputs_htail = [
            ("r_arm", Aircraft.HorizontalTail.MOMENT_RATIO),
            ("ar", Aircraft.HorizontalTail.ASPECT_RATIO),
        ]
        rename_inputs_vtail = [
            ("r_arm", Aircraft.VerticalTail.MOMENT_RATIO),
            ("tr", Aircraft.VerticalTail.TAPER_RATIO),
        ]

        # outputs that are used in groups other than this one
        higher_level_outputs_htail = [
            ("area", Aircraft.HorizontalTail.AREA),
            ("span", Aircraft.HorizontalTail.SPAN),
            ("rchord", Aircraft.HorizontalTail.ROOT_CHORD),
            ("arm", Aircraft.HorizontalTail.MOMENT_ARM),
        ]
        higher_level_outputs_vtail = [
            ("area", Aircraft.VerticalTail.AREA),
            ("span", Aircraft.VerticalTail.SPAN),
            ("rchord", Aircraft.VerticalTail.ROOT_CHORD),
            ("arm", Aircraft.VerticalTail.MOMENT_ARM),
        ]

        # outputs from htail and vtail that are purely promoted to change the name
        rename_outputs_htail = [
            ("chord", Aircraft.HorizontalTail.AVERAGE_CHORD),
        ]
        rename_outputs_vtail = [
            ("chord", Aircraft.VerticalTail.AVERAGE_CHORD),
        ]

        # outputs from components in this group that are used in this group
        connected_outputs_htail_vc = [
            ("vol_coef", Aircraft.HorizontalTail.VOLUME_COEFFICIENT),
        ]
        connected_outputs_vtail_vc = [
            ("vol_coef", Aircraft.VerticalTail.VOLUME_COEFFICIENT),
        ]

        # inputs to components in this group that are calculated in this group
        connected_inputs_htail = [
            ("vol_coef", Aircraft.HorizontalTail.VOLUME_COEFFICIENT),
        ]
        connected_inputs_vtail = [
            ("vol_coef", Aircraft.VerticalTail.VOLUME_COEFFICIENT),
        ]

        if self.options["aviary_options"].get_val(Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, units='unitless'):
            self.add_subsystem(
                "htail_vc",
                TailVolCoef(aviary_options=aviary_options),
                promotes_inputs=higher_level_inputs_htail_vc + ["aircraft:*"],
                promotes_outputs=connected_outputs_htail_vc,
            )
        if self.options["aviary_options"].get_val(Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, units='unitless'):
            self.add_subsystem(
                "vtail_vc",
                TailVolCoef(aviary_options=aviary_options, vertical=True),
                promotes_inputs=higher_level_inputs_vtail_vc + ["aircraft:*"],
                promotes_outputs=connected_outputs_vtail_vc,
            )

        self.add_subsystem(
            "htail",
            TailSize(aviary_options=aviary_options,),
            promotes_inputs=higher_level_inputs_htail
            + rename_inputs_htail
            + connected_inputs_htail
            + ["aircraft:*"],
            promotes_outputs=higher_level_outputs_htail + rename_outputs_htail,
        )

        self.add_subsystem(
            "vtail",
            TailSize(aviary_options=aviary_options,),
            promotes_inputs=higher_level_inputs_vtail
            + rename_inputs_vtail
            + connected_inputs_vtail
            + ["aircraft:*"],
            promotes_outputs=higher_level_outputs_vtail + rename_outputs_vtail,
        )

        self.set_input_defaults(Aircraft.Wing.AVERAGE_CHORD, 12.615, units="ft")
        self.set_input_defaults(Aircraft.Wing.SPAN, 117.8054, units="ft")

        # override horizontal tail defaults
        self.set_input_defaults(Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145)
        self.set_input_defaults(Aircraft.VerticalTail.MOMENT_RATIO, 2.362)
        self.set_input_defaults(Aircraft.VerticalTail.ASPECT_RATIO, 1.67)
        self.set_input_defaults(Aircraft.VerticalTail.TAPER_RATIO, 0.801)
