import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft
from openmdao.utils.units import convert_units


class TrappedFuelCapacity(om.ExplicitComponent):
    """Compute the maximum fuel that can be carried in the fuselage."""

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, units='lbm')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        unusable_fuel_coeff = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]

        trapped_fuel_wt = unusable_fuel_coeff * (wing_area**0.5) * fuel_vol_frac / 0.430

        if fuel_vol_frac <= 0.075:  # note: this technically creates a discontinuity # won't change
            trapped_fuel_wt = unusable_fuel_coeff * 0.18 * (wing_area**0.5)

        outputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS] = trapped_fuel_wt

    def compute_partials(self, inputs, J):
        wing_area = inputs[Aircraft.Wing.AREA]
        fuel_vol_frac = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
        unusable_fuel_coeff = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT]

        dtrapped_fuel_wt_dmass_coeff_12 = (wing_area**0.5) * fuel_vol_frac / 0.430
        dtrapped_fuel_wt_dwing_area = (
            0.5 * unusable_fuel_coeff * (wing_area**-0.5) * fuel_vol_frac / 0.430
        )
        dtrapped_fuel_wt_dfuel_vol_frac = unusable_fuel_coeff * (wing_area**0.5) / 0.430

        if fuel_vol_frac <= 0.075:  # note: this technically creates a discontinuity # won't change
            dtrapped_fuel_wt_dmass_coeff_12 = 0.18 * (wing_area**0.5)
            dtrapped_fuel_wt_dwing_area = 0.5 * unusable_fuel_coeff * 0.18 * (wing_area**-0.5)
            dtrapped_fuel_wt_dfuel_vol_frac = 0.0

        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Wing.AREA] = dtrapped_fuel_wt_dwing_area
        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.WING_FUEL_FRACTION] = (
            dtrapped_fuel_wt_dfuel_vol_frac
        )
        J[Aircraft.Fuel.UNUSABLE_FUEL_MASS, Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT] = (
            dtrapped_fuel_wt_dmass_coeff_12
        )

    """Compute the maximum fuel that can be carried in the wing's enclosed space."""

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.DENSITY, units='lbm/galUS')
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY_AREA, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY_TERM_A, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY_TERM_B, units='unitless')
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_FRACTION, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='ft')
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, units='unitless')
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, units='unitless')

        add_aviary_output(self, Aircraft.Fuel.WING_FUEL_CAPACITY, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        wing_ref_cap_terma = inputs[Aircraft.Fuel.WING_REF_CAPACITY_TERM_A]
        wing_area = inputs[Aircraft.Wing.AREA]

        if wing_ref_cap_terma.real > 0.0:
            wing_ref_cap = inputs[Aircraft.Fuel.WING_REF_CAPACITY]
            wing_ref_cap_area = inputs[Aircraft.Fuel.WING_REF_CAPACITY_AREA]
            wing_ref_cap_termb = inputs[Aircraft.Fuel.WING_REF_CAPACITY_TERM_B]

            fuel_cap_wing = (
                wing_ref_cap
                + wing_ref_cap_terma * (wing_area**1.5 - wing_ref_cap_area**1.5)
                + wing_ref_cap_termb * (wing_area - wing_ref_cap_area)
            )

        else:
            fuel_density = convert_units(inputs[Aircraft.Fuel.DENSITY], 'lbm/galUS', 'lbm/ft**3')
            volume_fraction = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
            span = inputs[Aircraft.Wing.SPAN]
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]
            # calculate volume of wing assuming truncated rectangular based pyramid:
            volume_of_wing = (
                (2 / 3)
                * wing_area**2
                * thickness_to_chord
                * (1.0 - taper_ratio / (1.0 + taper_ratio) ** 2)
                / span
            )
            fuel_cap_wing = fuel_density * volume_fraction * volume_of_wing

        outputs[Aircraft.Fuel.WING_FUEL_CAPACITY] = fuel_cap_wing

    def compute_partials(self, inputs, partials):
        wing_ref_cap_terma = inputs[Aircraft.Fuel.WING_REF_CAPACITY_TERM_A]
        wing_area = inputs[Aircraft.Wing.AREA]

        if wing_ref_cap_terma.real > 0.0:
            wing_ref_cap_area = inputs[Aircraft.Fuel.WING_REF_CAPACITY_AREA]
            wing_ref_cap_termb = inputs[Aircraft.Fuel.WING_REF_CAPACITY_TERM_B]

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.WING_REF_CAPACITY] = 1.0

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.WING_REF_CAPACITY_TERM_A] = (
                wing_area**1.5 - wing_ref_cap_area**1.5
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.WING_REF_CAPACITY_TERM_B] = (
                wing_area - wing_ref_cap_area
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.AREA] = (
                1.5 * wing_ref_cap_terma * wing_area**0.5 + wing_ref_cap_termb
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.WING_REF_CAPACITY_AREA] = (
                -1.5 * wing_ref_cap_terma * wing_ref_cap_area**0.5 - wing_ref_cap_termb
            )

        else:
            fuel_density = convert_units(inputs[Aircraft.Fuel.DENSITY], 'lbm/galUS', 'lbm/ft**3')
            volume_fraction = inputs[Aircraft.Fuel.WING_FUEL_FRACTION]
            span = inputs[Aircraft.Wing.SPAN]
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

            den = 1.0 + taper_ratio
            tr_fact = 1.0 - taper_ratio / den**2
            dfact = -1.0 / den**2 + 2.0 * taper_ratio / den**3

            conversion_factor = convert_units(1.0, 'lbm/galUS', 'lbm/ft**3')
            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.DENSITY] = (
                volume_fraction * (2 / 3) * wing_area**2 * thickness_to_chord * tr_fact / span
            ) * conversion_factor

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.WING_FUEL_FRACTION] = (
                fuel_density * (2 / 3) * wing_area**2 * thickness_to_chord * tr_fact / span
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.SPAN] = (
                -fuel_density
                * volume_fraction
                * (2 / 3)
                * wing_area**2
                * thickness_to_chord
                * tr_fact
                / span**2
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.TAPER_RATIO] = (
                fuel_density
                * volume_fraction
                * (2 / 3)
                * wing_area**2
                * thickness_to_chord
                * dfact
                / span
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.THICKNESS_TO_CHORD] = (
                fuel_density * volume_fraction * (2 / 3) * wing_area**2 * tr_fact / span
            )

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.AREA] = (
                2.0
                * fuel_density
                * volume_fraction
                * (2 / 3)
                * wing_area
                * thickness_to_chord
                * tr_fact
                / span
            )
