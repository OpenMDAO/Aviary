import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission
from aviary.subsystems.dbf_based_mass.dbf_mass_variables import Aircraft
from aviary.subsystems.dbf_based_mass.dbf_mass_variable_meta_data import ExtendedMetaData


class DBFWingMass(om.ExplicitComponent):
    def initialize(self):
        add_aviary_option(self, 'dbf_mode', default=True)

    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.DBFWing.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.DBFWing.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.DBFWing.SPAR_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.DBFWing.SPAN, units='in')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.DBFWing.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.DBFWing.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.DBFWing.RIB_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.DBFWing.AIRFOIL_X_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.DBFWing.AIRFOIL_Y_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.DBFWing.ROOT_CHORD, units='in')

        # Skin
        add_aviary_input(self, Aircraft.DBFWing.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.DBFWing.SKIN_DENSITY, units='lbm/in**2')

        add_aviary_output(self, Aircraft.DBFWing.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def shoelace_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.DBFWing.NUM_SPARS]
        st = inputs[Aircraft.DBFWing.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.DBFWing.SPAR_DENSITY]
        span = inputs[Aircraft.DBFWing.SPAN]
        nr = inputs[Aircraft.DBFWing.NUM_RIBS]
        rt = inputs[Aircraft.DBFWing.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.DBFWing.RIB_DENSITY]
        rho_skin = inputs[Aircraft.DBFWing.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.Wing.WETTED_AREA]
        x_coords = inputs[Aircraft.DBFWing.AIRFOIL_X_COORDS]
        y_coords = inputs[Aircraft.DBFWing.AIRFOIL_Y_COORDS]
        chord = inputs[Aircraft.DBFWing.ROOT_CHORD]

        x_coords, y_coords = x_coords * chord, y_coords * chord
        cs_area = self.shoelace_area(x_coords, y_coords)

        rib_volume = nr * cs_area * rt
        spar_volume = ns * span * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.Wing.MASS] = rib_mass + spar_mass + skin_mass


class DBFHorizontalTailMass(om.ExplicitComponent):
    def initialize(self):
        add_aviary_option(self, 'dbf_mode', default=True)

    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.DBFHorizontalTail.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.SPAR_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.SPAN, units='in')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.DBFHorizontalTail.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.RIB_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.AIRFOIL_X_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.AIRFOIL_Y_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.ROOT_CHORD, units='in')

        # Skin
        add_aviary_input(self, Aircraft.DBFHorizontalTail.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.DBFHorizontalTail.SKIN_DENSITY, units='lbm/in**2')

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def shoelace_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.DBFHorizontalTail.NUM_SPARS]
        st = inputs[Aircraft.DBFHorizontalTail.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.DBFHorizontalTail.SPAR_DENSITY]
        span = inputs[Aircraft.DBFHorizontalTail.SPAN]
        nr = inputs[Aircraft.DBFHorizontalTail.NUM_RIBS]
        rt = inputs[Aircraft.DBFHorizontalTail.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.DBFHorizontalTail.RIB_DENSITY]
        cs_area = inputs[Aircraft.DBFHorizontalTail.CROSS_SECTIONAL_AREA]
        rho_skin = inputs[Aircraft.DBFHorizontalTail.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.DBFHorizontalTail.WETTED_AREA]
        x_coords = inputs[Aircraft.DBFHorizontalTail.AIRFOIL_X_COORDS]
        y_coords = inputs[Aircraft.DBFHorizontalTail.AIRFOIL_Y_COORDS]
        chord = inputs[Aircraft.DBFHorizontalTail.ROOT_CHORD]

        x_coords, y_coords = x_coords * chord, y_coords * chord
        cs_area = self.shoelace_area(x_coords, y_coords)

        rib_volume = nr * cs_area * rt
        spar_volume = ns * span * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.HorizontalTail.MASS] = rib_mass + spar_mass + skin_mass


class DBFVerticalTailMass(om.ExplicitComponent):
    def initialize(self):
        add_aviary_option(self, 'dbf_mode', default=True)

    def setup(self):
        # Spar-related inputs
        add_aviary_input(self, Aircraft.DBFVerticalTail.NUM_SPARS, units='unitless')
        add_aviary_input(self, Aircraft.DBFVerticalTail.SPAR_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.DBFVerticalTail.SPAR_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.DBFVerticalTail.SPAN, units='in')

        # Rib-related inputs
        add_aviary_input(self, Aircraft.DBFVerticalTail.NUM_RIBS, units='unitless')
        add_aviary_input(self, Aircraft.DBFVerticalTail.RIB_THICKNESS, units='in')
        add_aviary_input(self, Aircraft.DBFVerticalTail.RIB_DENSITY, units='lbm/in**3')
        add_aviary_input(self, Aircraft.DBFVerticalTail.AIRFOIL_X_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.DBFVerticalTail.AIRFOIL_Y_COORDS, units='unitless')
        add_aviary_input(self, Aircraft.DBFVerticalTail.ROOT_CHORD, units='in')

        # Skin
        add_aviary_input(self, Aircraft.DBFVerticalTail.WETTED_AREA, units='in**2')
        add_aviary_input(self, Aircraft.DBFVerticalTail.SKIN_DENSITY, units='lbm/in**2')

        add_aviary_output(self, Aircraft.VerticalTail.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*', method='cs')

    def shoelace_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        ns = inputs[Aircraft.DBFVerticalTail.NUM_SPARS]
        st = inputs[Aircraft.DBFVerticalTail.SPAR_THICKNESS]
        rho_spar = inputs[Aircraft.DBFVerticalTail.SPAR_DENSITY]
        span = inputs[Aircraft.VerticalTail.SPAN]
        nr = inputs[Aircraft.DBFVerticalTail.NUM_RIBS]
        rt = inputs[Aircraft.DBFVerticalTail.RIB_THICKNESS]
        rho_rib = inputs[Aircraft.DBFVerticalTail.RIB_DENSITY]
        cs_area = inputs[Aircraft.DBFVerticalTail.CROSS_SECTIONAL_AREA]
        rho_skin = inputs[Aircraft.DBFVerticalTail.SKIN_DENSITY]
        wetted_area = inputs[Aircraft.DBFVerticalTail.WETTED_AREA]
        x_coords = inputs[Aircraft.DBFVerticalTail.AIRFOIL_X_COORDS]
        y_coords = inputs[Aircraft.DBFVerticalTail.AIRFOIL_Y_COORDS]
        chord = inputs[Aircraft.DBFVerticalTail.ROOT_CHORD]

        x_coords, y_coords = x_coords * chord, y_coords * chord
        cs_area = self.shoelace_area(x_coords, y_coords)

        rib_volume = nr * cs_area * rt
        spar_volume = ns * span * np.pi * (st / 2) ** 2

        rib_mass = rib_volume * rho_rib
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        outputs[Aircraft.VerticalTail.MASS] = rib_mass + spar_mass + skin_mass
