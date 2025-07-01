import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variables import Aircraft
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variable_meta_data import (
    ExtendedMetaData,
)


class DBFWingMass(om.ExplicitComponent):
    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.Wing.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.Wing.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.Wing.SPAR_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.Wing.SPAN, units='in')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.Wing.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.Wing.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.Wing.RIB_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.Wing.AIRFOIL_X_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.Wing.AIRFOIL_Y_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='in')

        # Skin
        add_aviary_input(self, Aircraft.Wing.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.Wing.SKIN_DENSITY, units='lbm/in**2')

        add_aviary_output(self, Aircraft.Wing.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def shoelace_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.Wing.NUM_SPARS]
        st = inputs[Aircraft.Wing.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.Wing.SPAR_DENSITY]
        span = inputs[Aircraft.Wing.SPAN]
        nr = inputs[Aircraft.Wing.NUM_RIBS]
        rt = inputs[Aircraft.Wing.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.Wing.RIB_DENSITY]
        rho_skin = inputs[Aircraft.Wing.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.Wing.WETTED_AREA]
        x_coords = inputs[Aircraft.Wing.AIRFOIL_X_COORDS]
        y_coords = inputs[Aircraft.Wing.AIRFOIL_Y_COORDS]
        chord = inputs[Aircraft.Wing.ROOT_CHORD]

        x_coords, y_coords = x_coords * chord, y_coords * chord
        cs_area = self.shoelace_area(x_coords, y_coords)

        rib_volume = nr * cs_area * rt
        spar_volume = ns * span * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.Wing.MASS] = rib_mass + spar_mass + skin_mass


class DBFHorizontalTailMass(om.ExplicitComponent):
    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.HorizontalTail.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.HorizontalTail.SPAR_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='in')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.HorizontalTail.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.HorizontalTail.RIB_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.HorizontalTail.AIRFOIL_X_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.AIRFOIL_Y_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.HorizontalTail.ROOT_CHORD, units='in')

        # Skin
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.HorizontalTail.SKIN_DENSITY, units='lbm/in**2')

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def shoelace_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.HorizontalTail.NUM_SPARS]
        st = inputs[Aircraft.HorizontalTail.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.HorizontalTail.SPAR_DENSITY]
        span = inputs[Aircraft.HorizontalTail.SPAN]
        nr = inputs[Aircraft.HorizontalTail.NUM_RIBS]
        rt = inputs[Aircraft.HorizontalTail.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.HorizontalTail.RIB_DENSITY]
        rho_skin = inputs[Aircraft.HorizontalTail.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.HorizontalTail.WETTED_AREA]
        x_coords = inputs[Aircraft.HorizontalTail.AIRFOIL_X_COORDS]
        y_coords = inputs[Aircraft.HorizontalTail.AIRFOIL_Y_COORDS]
        chord = inputs[Aircraft.HorizontalTail.ROOT_CHORD]

        x_coords, y_coords = x_coords * chord, y_coords * chord
        cs_area = self.shoelace_area(x_coords, y_coords)

        rib_volume = nr * cs_area * rt
        spar_volume = ns * span * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.HorizontalTail.MASS] = rib_mass + spar_mass + skin_mass


class DBFVerticalTailMass(om.ExplicitComponent):
    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.VerticalTail.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.VerticalTail.SPAR_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='in')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.VerticalTail.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.VerticalTail.RIB_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.VerticalTail.AIRFOIL_X_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.AIRFOIL_Y_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.VerticalTail.ROOT_CHORD, units='in')

        # Skin
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.VerticalTail.SKIN_DENSITY, units='lbm/in**2')

        add_aviary_output(self, Aircraft.VerticalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def shoelace_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.VerticalTail.NUM_SPARS]
        st = inputs[Aircraft.VerticalTail.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.VerticalTail.SPAR_DENSITY]
        span = inputs[Aircraft.VerticalTail.SPAN]
        nr = inputs[Aircraft.VerticalTail.NUM_RIBS]
        rt = inputs[Aircraft.VerticalTail.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.VerticalTail.RIB_DENSITY]
        rho_skin = inputs[Aircraft.VerticalTail.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.VerticalTail.WETTED_AREA]
        x_coords = inputs[Aircraft.VerticalTail.AIRFOIL_X_COORDS]
        y_coords = inputs[Aircraft.VerticalTail.AIRFOIL_Y_COORDS]
        chord = inputs[Aircraft.VerticalTail.ROOT_CHORD]

        x_coords, y_coords = x_coords * chord, y_coords * chord
        cs_area = self.shoelace_area(x_coords, y_coords)

        rib_volume = nr * cs_area * rt
        spar_volume = ns * span * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.VerticalTail.MASS] = rib_mass + spar_mass + skin_mass


class DBFFuselageMass(om.ExplicitComponent):
    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.Fuselage.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.Fuselage.SPAR_DENSITY, units='lbm/in**3')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.Fuselage.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.Fuselage.RIB_DENSITY, units='lbm/in**3')

        # Skin
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.Fuselage.SKIN_DENSITY, units='lbm/in**2')

        # Main Body
        add_aviary_input(self, Aircraft.Fuselage.FUSELAGE_HEIGHT, units='in')
        add_aviary_input(self, Aircraft.Fuselage.FUSELAGE_WIDTH, units='in')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='in')

        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.Fuselage.NUM_SPARS]
        st = inputs[Aircraft.Fuselage.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.Fuselage.SPAR_DENSITY]
        nr = inputs[Aircraft.Fuselage.NUM_RIBS]
        rt = inputs[Aircraft.Fuselage.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.Fuselage.RIB_DENSITY]
        rho_skin = inputs[Aircraft.Fuselage.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]
        height = inputs[Aircraft.Fuselage.FUSELAGE_HEIGHT]
        width = inputs[Aircraft.Fuselage.FUSELAGE_WIDTH]
        length = inputs[Aircraft.Fuselage.LENGTH]

        rib_volume = nr * height * width * rt
        spar_volume = ns * length * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.Fuselage.MASS] = rib_mass + spar_mass + skin_mass
