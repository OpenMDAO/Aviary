import openmdao.api as om

from aviary.utils.utils import mass_to_force_english, mass_to_force_english_derivative
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class MiscControlMass(om.ExplicitComponent):
    """
    Computation of total mass of cockpit controls, and stability augmentation system. The
    methodology is based on the GASP weight equations, modified to output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        add_aviary_input(self, Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(
            self, Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS, units='lbm'
        )
        add_aviary_input(self, Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(
            self,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
            units='unitless',
        )
        add_aviary_input(self, Aircraft.Controls.CONTROL_MASS_INCREMENT, units='lbm')

        add_aviary_output(self, Aircraft.Controls.COCKPIT_CONTROL_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Controls.COCKPIT_CONTROL_MASS,
            [
                Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER,
                Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
                Aircraft.Design.GROSS_MASS,
            ],
        )
        self.declare_partials(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
            [
                Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
                Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]

        c_mass_trend_cockpit_control = inputs[Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT]

        stab_aug_ref_mass = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS]
        CK15 = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER]
        CK19 = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER]

        intermediate_cockpit_control_mass = (
            c_mass_trend_cockpit_control * (gross_mass_initial / 1000.0) ** 0.41
        )
        cockpit_control_mass = CK15 * intermediate_cockpit_control_mass
        stab_control_mass = CK19 * stab_aug_ref_mass

        outputs[Aircraft.Controls.COCKPIT_CONTROL_MASS] = cockpit_control_mass
        outputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS] = stab_control_mass

    def compute_partials(self, inputs, J):
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        c_mass_trend_cockpit_control = inputs[Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT]

        stab_aug_ref_mass = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS]
        CK15 = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER]
        CK19 = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER]

        J[
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER,
        ] = stab_aug_ref_mass
        J[
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS,
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_REFERENCE_MASS,
        ] = CK19

        J[
            Aircraft.Controls.COCKPIT_CONTROL_MASS,
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER,
        ] = c_mass_trend_cockpit_control * (gross_mass_initial / 1000.0) ** 0.41

        J[
            Aircraft.Controls.COCKPIT_CONTROL_MASS,
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT,
        ] = CK15 * (gross_mass_initial / 1000.0) ** 0.41

        J[
            Aircraft.Controls.COCKPIT_CONTROL_MASS,
            Aircraft.Design.GROSS_MASS,
        ] = (
            CK15
            * c_mass_trend_cockpit_control
            * 0.41
            * (gross_mass_initial / 1000.0) ** -0.59
            / 1000.0
        )


class SurfaceControlMass(om.ExplicitComponent):
    """
    Computation mass of surface controls. The methodology is based on the GASP weight equations,
    modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Mission.GRAVITY, units='ft/s**2')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.ULTIMATE_LOAD_FACTOR, units='unitless')
        self.add_input('min_dive_vel', units='kn', desc='VDMIN: dive velocity')
        add_aviary_input(self, Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Controls.COCKPIT_CONTROL_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Wing.SURFACE_CONTROL_MASS,
            [
                Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
                Aircraft.Wing.AREA,
                Aircraft.Design.GROSS_MASS,
                Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
                'min_dive_vel',
                Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER,
                Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER,
                Aircraft.Controls.COCKPIT_CONTROL_MASS,
            ],
        )

    def compute(self, inputs, outputs):
        gravity = self.options[Mission.GRAVITY]

        c_mass_trend_wing_control = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT]
        wing_area = inputs[Aircraft.Wing.AREA]
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        CK15 = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER]
        CK18 = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        min_dive_vel = inputs['min_dive_vel']

        gross_wt_initial = mass_to_force_english(
            (gross_mass_initial, 'lbm'), gravity
        )  # control surfaces sized based on forces

        dive_param = (1.15 * min_dive_vel) ** 2 / 391.0
        intermediate_control_mass = (
            c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )

        intermediate_cockpit_control_mass = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS] / CK15
        wing_control_mass = CK18 * (intermediate_control_mass - intermediate_cockpit_control_mass)

        outputs[Aircraft.Wing.SURFACE_CONTROL_MASS] = (
            wing_control_mass  # implied 1 lbf -> 1 lbm conversion, since GASP equations were designed at standard gravity
        )

    def compute_partials(self, inputs, J):
        gravity = self.options[Mission.GRAVITY]

        c_mass_trend_wing_control = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT]
        wing_area = inputs[Aircraft.Wing.AREA]
        gross_mass_initial = inputs[Aircraft.Design.GROSS_MASS]
        ULF = inputs[Aircraft.Wing.ULTIMATE_LOAD_FACTOR]
        CK15 = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER]
        CK18 = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER]
        min_dive_vel = inputs['min_dive_vel']

        gross_wt_initial = mass_to_force_english((gross_mass_initial, 'lbm'), gravity)
        dforce_dmass = mass_to_force_english_derivative(gravity)

        dive_param = (1.15 * min_dive_vel) ** 2 / 391.0
        intermediate_control_mass = (
            c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )

        intermediate_cockpit_control_mass = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS] / CK15

        J[
            Aircraft.Wing.SURFACE_CONTROL_MASS,
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT,
        ] = (
            CK18
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.AREA] = 0.317 * (
            c_mass_trend_wing_control
            * wing_area ** (0.317 - 1)
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param**0.345
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Design.GROSS_MASS] = (
            CK18
            * (
                0.602
                * c_mass_trend_wing_control
                * wing_area**0.317
                * (gross_wt_initial / 1000.0) ** (0.602 - 1)
                * (1 / 1000)
                * ULF**0.525
                * dive_param**0.345
            )
            * dforce_dmass
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.ULTIMATE_LOAD_FACTOR] = (
            0.525
            * c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF ** (0.525 - 1)
            * dive_param**0.345
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER] = (
            intermediate_control_mass - intermediate_cockpit_control_mass
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, 'min_dive_vel'] = (
            0.345
            * c_mass_trend_wing_control
            * wing_area**0.317
            * (gross_wt_initial / 1000.0) ** 0.602
            * ULF**0.525
            * dive_param ** (0.345 - 1)
            * 2
            * 1.15
            * min_dive_vel
            * 1.15
            / 391
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Controls.COCKPIT_CONTROL_MASS] = -(
            CK18 / CK15
        )

        J[Aircraft.Wing.SURFACE_CONTROL_MASS, Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER] = (
            CK18 * inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS] / CK15**2
        )


class SumControlMass(om.ExplicitComponent):
    """Computation of control mass."""

    def setup(self):
        add_aviary_input(self, Aircraft.Controls.CONTROL_MASS_INCREMENT, units='lbm')
        add_aviary_input(self, Aircraft.Controls.COCKPIT_CONTROL_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS, units='lbm')

        add_aviary_output(self, Aircraft.Controls.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Controls.MASS, '*', val=1.0)

    def compute(self, inputs, outputs):
        delta_control_mass = inputs[Aircraft.Controls.CONTROL_MASS_INCREMENT]

        cockpit_control_mass = inputs[Aircraft.Controls.COCKPIT_CONTROL_MASS]
        wing_control_mass = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS]
        stab_control_mass = inputs[Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS]

        outputs[Aircraft.Controls.MASS] = (
            cockpit_control_mass + wing_control_mass + stab_control_mass + delta_control_mass
        )


class ControlMassGroup(om.Group):
    """Group of all control components for GASP-based mass."""

    def setup(self):
        self.add_subsystem(
            'misc_control',
            MiscControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'surface_control',
            SurfaceControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
        self.add_subsystem(
            'sum_control',
            SumControlMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )
