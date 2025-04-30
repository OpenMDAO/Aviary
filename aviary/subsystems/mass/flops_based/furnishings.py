import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportFurnishingsGroupMass(om.ExplicitComponent):
    """
    Calculates the mass of the furnishings group using the transport/general
    aviation method.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS)
        add_aviary_option(self, Aircraft.Fuselage.NUM_FUSELAGES)

    def setup(self):
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')

        add_aviary_output(self, Aircraft.Furnishings.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of=Aircraft.Furnishings.MASS, wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        flight_crew_count = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        first_class_count = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        business_class_count = self.options[Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS]
        tourist_class_count = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]

        fuse_count = self.options[Aircraft.Fuselage.NUM_FUSELAGES]

        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]

        pax_compart_length = inputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH]

        fuse_max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        fuse_max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]

        outputs[Aircraft.Furnishings.MASS] = (
            127.0 * flight_crew_count
            + 112.0 * first_class_count
            + 78.0 * business_class_count
            + 44.0 * tourist_class_count
            + 2.6 * pax_compart_length * (fuse_max_width + fuse_max_height) * fuse_count
        ) * scaler

    def compute_partials(self, inputs, J):
        flight_crew_count = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        first_class_count = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        business_class_count = self.options[Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS]
        tourist_class_count = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]

        fuse_count = self.options[Aircraft.Fuselage.NUM_FUSELAGES]

        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        fuse_max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        fuse_max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]

        pax_compart_length = inputs[Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH]

        J[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] = (
            127.0 * flight_crew_count
            + 112.0 * first_class_count
            + 78.0 * business_class_count
            + 44.0 * tourist_class_count
            + 2.6 * pax_compart_length * (fuse_max_width + fuse_max_height) * fuse_count
        )

        J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.PASSENGER_COMPARTMENT_LENGTH] = (
            2.6 * scaler * (fuse_max_width + fuse_max_height) * fuse_count
        )

        J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.MAX_WIDTH] = J[
            Aircraft.Furnishings.MASS, Aircraft.Fuselage.MAX_HEIGHT
        ] = 2.6 * scaler * pax_compart_length * fuse_count


class BWBFurnishingsGroupMass(om.ExplicitComponent):
    """Calculates the mass of the furnishings group for HWB aircraft."""

    def initialize(self):
        add_aviary_option(self, Aircraft.BWB.NUM_BAYS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_FIRST_CLASS)
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS)
        add_aviary_option(self, Aircraft.Fuselage.MILITARY_CARGO_FLOOR)

    def setup(self):
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')

        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')
        add_aviary_input(self, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP, units='deg')

        add_aviary_output(self, Aircraft.Furnishings.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of=Aircraft.Furnishings.MASS, wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        flight_crew_count = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        first_class_count = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        business_class_count = self.options[Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS]
        tourist_class_count = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]

        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]
        fuse_max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        fuse_max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]

        weight = (
            127.0 * flight_crew_count
            + 112.0 * first_class_count
            + 78.0 * business_class_count
            + 44.0 * tourist_class_count
        )
        # outputs[Aircraft.Furnishings.MASS] = weight / GRAV_ENGLISH_LBM

        if not self.options[Aircraft.Fuselage.MILITARY_CARGO_FLOOR]:
            acabin = inputs[Aircraft.Fuselage.CABIN_AREA]
            nbay = self.options[Aircraft.BWB.NUM_BAYS]

            cos = np.cos(np.pi / 180 * (inputs[Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP]))

            weight += 2.6 * (
                (acabin / fuse_max_width) * (fuse_max_width + fuse_max_height * nbay)
                + (fuse_max_width * (1.0 + 1.0 / cos) * fuse_max_height)
            )

        outputs[Aircraft.Furnishings.MASS] = weight * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        flight_crew_count = self.options[Aircraft.CrewPayload.NUM_FLIGHT_CREW]
        first_class_count = self.options[Aircraft.CrewPayload.Design.NUM_FIRST_CLASS]
        business_class_count = self.options[Aircraft.CrewPayload.Design.NUM_BUSINESS_CLASS]
        tourist_class_count = self.options[Aircraft.CrewPayload.Design.NUM_TOURIST_CLASS]

        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]

        J[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] = (
            127.0 * flight_crew_count
            + 112.0 * first_class_count
            + 78.0 * business_class_count
            + 44.0 * tourist_class_count
        ) / GRAV_ENGLISH_LBM

        if self.options[Aircraft.Fuselage.MILITARY_CARGO_FLOOR]:
            J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.CABIN_AREA] = 0.0

            J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.MAX_WIDTH] = 0.0

            J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.MAX_HEIGHT] = 0.0

            J[Aircraft.Furnishings.MASS, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP] = 0.0

        else:
            d2r = np.radians(inputs[Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP])
            cos = np.cos(d2r)
            tan = np.tan(d2r)

            acabin = inputs[Aircraft.Fuselage.CABIN_AREA]
            nbay = self.options[Aircraft.BWB.NUM_BAYS]
            fuse_max_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
            fuse_max_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
            cabin_area = inputs[Aircraft.Fuselage.CABIN_AREA]

            J[Aircraft.Furnishings.MASS, Aircraft.Furnishings.MASS_SCALER] += (
                2.6
                * (
                    (acabin / fuse_max_width) * (fuse_max_width + fuse_max_height * nbay)
                    + (fuse_max_width * (1.0 + 1.0 / cos) * fuse_max_height)
                )
                / GRAV_ENGLISH_LBM
            )

            J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.CABIN_AREA] = (
                2.6 * scaler * (fuse_max_width + fuse_max_height * nbay) / fuse_max_width
            ) / GRAV_ENGLISH_LBM

            J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.MAX_WIDTH] = (
                2.6
                * scaler
                * (
                    -cabin_area * fuse_max_height * nbay / (fuse_max_width * fuse_max_width)
                    + (1.0 + 1.0 / cos) * fuse_max_height
                )
            ) / GRAV_ENGLISH_LBM

            J[Aircraft.Furnishings.MASS, Aircraft.Fuselage.MAX_HEIGHT] = (
                2.6
                * scaler
                * ((cabin_area / fuse_max_width) * nbay + fuse_max_width * (1.0 + 1.0 / cos))
            ) / GRAV_ENGLISH_LBM

            J[Aircraft.Furnishings.MASS, Aircraft.BWB.PASSENGER_LEADING_EDGE_SWEEP] = (
                2.6 * scaler * fuse_max_width * fuse_max_height * tan / cos / (180 / np.pi)
            ) / GRAV_ENGLISH_LBM


class AltFurnishingsGroupMassBase(om.ExplicitComponent):
    """
    Calculates the base mass of the furnishings group using the alternate
    method. The methodology is based on the FLOPS weight equations,
    modified to output mass instead of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.CrewPayload.Design.NUM_PASSENGERS)

    def setup(self):
        add_aviary_input(self, Aircraft.Furnishings.MASS_SCALER, units='unitless')

        add_aviary_output(self, Aircraft.Furnishings.MASS_BASE, units='lbm')

    def setup_partials(self):
        self.declare_partials(of=Aircraft.Furnishings.MASS_BASE, wrt='*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pax_count = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]
        scaler = inputs[Aircraft.Furnishings.MASS_SCALER]

        outputs[Aircraft.Furnishings.MASS_BASE] = (82.15 * pax_count + 3600.0) * scaler

    def compute_partials(self, inputs, J, discrete_inputs=None):
        pax_count = self.options[Aircraft.CrewPayload.Design.NUM_PASSENGERS]

        J[Aircraft.Furnishings.MASS_BASE, Aircraft.Furnishings.MASS_SCALER] = (
            82.15 * pax_count + 3600.0
        )


class AltFurnishingsGroupMass(om.ExplicitComponent):
    """
    Completes the mass calculation for the furnishings group using the
    alternate method. The methodology is based on the FLOPS weight
    equations, modified to output mass instead of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Furnishings.MASS_BASE, units='lbm')
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE, units='lbm')

        add_aviary_output(self, Aircraft.Furnishings.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            of=Aircraft.Furnishings.MASS,
            wrt=[
                Aircraft.Design.STRUCTURE_MASS,
                Aircraft.Propulsion.MASS,
                Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE,
            ],
            val=0.01,
        )

        self.declare_partials(
            of=Aircraft.Furnishings.MASS, wrt=Aircraft.Furnishings.MASS_BASE, val=1.0
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        furn_mass_base = inputs[Aircraft.Furnishings.MASS_BASE]
        struct_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        syseq_mass_base = inputs[Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE]

        outputs[Aircraft.Furnishings.MASS] = furn_mass_base + 0.01 * (
            struct_mass + prop_mass + syseq_mass_base
        )
