import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class FuelCapacityGroup(om.Group):
    """
    Compute the maximum fuel that can be carried.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options = self.options['aviary_options']

        self.add_subsystem(
            'wing_fuel_capacity',
            WingFuelCapacity(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'fuselage_fuel_capacity',
            FuselageFuelCapacity(),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'auxiliary_fuel_capacity',
            AuxFuelCapacity(),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'total_fuel_capacity',
            TotalFuelCapacity(),
            promotes_inputs=['*'], promotes_outputs=['*'])


class FuselageFuelCapacity(om.ExplicitComponent):
    """
    Compute the maximum fuel that can be carried in the fuselage.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, 0.0),
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_CAPACITY, 0.0),
        add_aviary_output(self, Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0),

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
            Aircraft.Fuel.TOTAL_CAPACITY, val=1.0)
        self.declare_partials(
            Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
            Aircraft.Fuel.WING_FUEL_CAPACITY, val=-1.0)

    def compute(self, inputs, outputs):
        outputs[Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY] = (
            inputs[Aircraft.Fuel.TOTAL_CAPACITY]
            - inputs[Aircraft.Fuel.WING_FUEL_CAPACITY])


class AuxFuelCapacity(om.ExplicitComponent):
    """
    Compute the maximum fuel that can be carried in the auxiliary tanks.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.TOTAL_CAPACITY, 0.0),
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_CAPACITY, 0.0),
        add_aviary_input(self, Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0),
        add_aviary_output(self, Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0),

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
            Aircraft.Fuel.TOTAL_CAPACITY, val=1.0)
        self.declare_partials(
            Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
            Aircraft.Fuel.WING_FUEL_CAPACITY, val=-1.0)
        self.declare_partials(
            Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
            Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, val=-1.0)

    def compute(self, inputs, outputs):
        outputs[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY] = (
            inputs[Aircraft.Fuel.TOTAL_CAPACITY]
            - inputs[Aircraft.Fuel.WING_FUEL_CAPACITY]
            - inputs[Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY])


class TotalFuelCapacity(om.ExplicitComponent):
    """
    Compute the total fuel that can be carried in all tanks.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.WING_FUEL_CAPACITY, 0.0),
        add_aviary_input(self, Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, 0.0),
        add_aviary_input(self, Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, 0.0),
        add_aviary_output(self, Aircraft.Fuel.TOTAL_CAPACITY, 0.0),

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuel.TOTAL_CAPACITY,
            Aircraft.Fuel.WING_FUEL_CAPACITY, val=1.0)
        self.declare_partials(
            Aircraft.Fuel.TOTAL_CAPACITY,
            Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY, val=1.0)
        self.declare_partials(
            Aircraft.Fuel.TOTAL_CAPACITY,
            Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY, val=1.0)

    def compute(self, inputs, outputs):
        outputs[Aircraft.Fuel.TOTAL_CAPACITY] = (
            inputs[Aircraft.Fuel.WING_FUEL_CAPACITY]
            + inputs[Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY]
            + inputs[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY])


class WingFuelCapacity(om.ExplicitComponent):
    """
    Compute the maximum fuel that can be carried in the wing's enclosed space.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.DENSITY_RATIO, 1.0)
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY, 0.0)
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY_AREA, 0.0)
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY_TERM_A, 0.0)
        add_aviary_input(self, Aircraft.Fuel.WING_REF_CAPACITY_TERM_B, 0.0)

        add_aviary_input(self, Aircraft.Fuel.CAPACITY_FACTOR, 23.0)

        add_aviary_input(self, Aircraft.Wing.AREA, 0.0)
        add_aviary_input(self, Aircraft.Wing.SPAN, 0.0)
        add_aviary_input(self, Aircraft.Wing.TAPER_RATIO, 0.0)
        add_aviary_input(self, Aircraft.Wing.THICKNESS_TO_CHORD, 0.0)

        add_aviary_output(self, Aircraft.Fuel.WING_FUEL_CAPACITY, 0.0)

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
                + wing_ref_cap_termb * (wing_area - wing_ref_cap_area))

        else:
            density_ratio = inputs[Aircraft.Fuel.DENSITY_RATIO]
            cap_factor = inputs[Aircraft.Fuel.CAPACITY_FACTOR]
            span = inputs[Aircraft.Wing.SPAN]
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

            fuel_cap_wing = (
                density_ratio * cap_factor * wing_area**2 * thickness_to_chord
                * (1.0 - taper_ratio / (1.0 + taper_ratio)**2) / span)

        outputs[Aircraft.Fuel.WING_FUEL_CAPACITY] = fuel_cap_wing

    def compute_partials(self, inputs, partials):
        wing_ref_cap_terma = inputs[Aircraft.Fuel.WING_REF_CAPACITY_TERM_A]
        wing_area = inputs[Aircraft.Wing.AREA]

        if wing_ref_cap_terma.real > 0.0:
            wing_ref_cap_area = inputs[Aircraft.Fuel.WING_REF_CAPACITY_AREA]
            wing_ref_cap_termb = inputs[Aircraft.Fuel.WING_REF_CAPACITY_TERM_B]

            partials[
                Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.WING_REF_CAPACITY] = 1.0

            partials[
                Aircraft.Fuel.WING_FUEL_CAPACITY,
                Aircraft.Fuel.WING_REF_CAPACITY_TERM_A] = \
                wing_area**1.5 - wing_ref_cap_area**1.5

            partials[
                Aircraft.Fuel.WING_FUEL_CAPACITY,
                Aircraft.Fuel.WING_REF_CAPACITY_TERM_B] = wing_area - wing_ref_cap_area

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.AREA] = \
                1.5 * wing_ref_cap_terma * wing_area**0.5 + wing_ref_cap_termb

            partials[
                Aircraft.Fuel.WING_FUEL_CAPACITY,
                Aircraft.Fuel.WING_REF_CAPACITY_AREA] = \
                -1.5 * wing_ref_cap_terma * wing_ref_cap_area**0.5 - wing_ref_cap_termb

        else:
            density_ratio = inputs[Aircraft.Fuel.DENSITY_RATIO]
            cap_factor = inputs[Aircraft.Fuel.CAPACITY_FACTOR]
            span = inputs[Aircraft.Wing.SPAN]
            taper_ratio = inputs[Aircraft.Wing.TAPER_RATIO]
            thickness_to_chord = inputs[Aircraft.Wing.THICKNESS_TO_CHORD]

            den = (1.0 + taper_ratio)
            tr_fact = (1.0 - taper_ratio / den**2)
            dfact = - 1.0/den**2 + 2.0 * taper_ratio / den**3

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.DENSITY_RATIO] = \
                cap_factor * wing_area**2 * thickness_to_chord * tr_fact / span

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Fuel.CAPACITY_FACTOR] = \
                density_ratio * wing_area**2 * thickness_to_chord * tr_fact / span

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.SPAN] = (
                -density_ratio * cap_factor * wing_area**2 * thickness_to_chord * tr_fact
                / span**2)

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.TAPER_RATIO] = (
                density_ratio * cap_factor * wing_area**2 * thickness_to_chord * dfact
                / span)

            partials[
                Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.THICKNESS_TO_CHORD] = \
                density_ratio * cap_factor * wing_area**2 * tr_fact / span

            partials[Aircraft.Fuel.WING_FUEL_CAPACITY, Aircraft.Wing.AREA] = (
                2.0 * density_ratio * cap_factor * wing_area * thickness_to_chord
                * tr_fact / span)
