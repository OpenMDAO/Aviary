import numpy as np
import os
import jax.numpy as jnp
import openmdao.api as om

from aviary.subsystems.mass.UAV_mass.utils.materials_database import materials
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.subsystems.mass.UAV_mass.utils.load_airfoil import load_airfoil_if_needed

from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variable_metadata import (
    ExtendedMetaData,
)

class HorizontalTailMass(om.JaxExplicitComponent):
    def initialize(self):
        add_aviary_option(self, Aircraft.HorizontalTail.AIRFOIL_PATH, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.RIB_MATERIALS, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.NUM_SPARS, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SPAR_OUTER_DIAMETER, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SPAR_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SPAR_WALL_THICKNESS, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.RIB_THICKNESS, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.RIB_LIGHTENING_FACTOR, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.AREAL_SKIN_DENSITY, units='kg/m**2', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.GLUE_FACTOR, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.STRINGER_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.STRINGER_THICKNESS, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SHEETING_THICKNESS, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SHEETING_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SHEETING_COVERAGE, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.SHEETING_LIGHTENING_FACTOR, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.NUM_STRINGERS, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.HorizontalTail.MISC_MASS, units='kg', meta_data=ExtendedMetaData)
      
        self._airfoil_loaded = False

    def setup(self):
        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='m', meta_data=ExtendedMetaData, primal_name='span')
        add_aviary_input(self, Aircraft.HorizontalTail.ROOT_CHORD, units='m', meta_data=ExtendedMetaData, primal_name='root_chord')
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, units='m**2', meta_data=ExtendedMetaData, primal_name='wetted_area')

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='kg', meta_data=ExtendedMetaData, primal_name='mass')
        
    def compute_primal(self, span, root_chord, wetted_area):
        num_spars = self.options[Aircraft.HorizontalTail.NUM_SPARS]
        rib_lightening_factor = self.options[Aircraft.HorizontalTail.RIB_LIGHTENING_FACTOR]
        rib_thickness, units = self.options[Aircraft.HorizontalTail.RIB_THICKNESS]
        rib_thickness = jnp.asarray(rib_thickness)
        rho_skin, units = self.options[Aircraft.HorizontalTail.AREAL_SKIN_DENSITY]
        spar_outer_diameter, units = self.options[Aircraft.HorizontalTail.SPAR_OUTER_DIAMETER]
        rho_spar, units = self.options[Aircraft.HorizontalTail.SPAR_DENSITY]
        spar_wall_thickness, units = self.options[Aircraft.HorizontalTail.SPAR_WALL_THICKNESS]
        glue_factor = self.options[Aircraft.HorizontalTail.GLUE_FACTOR]
        stringer_thickness, units = self.options[Aircraft.HorizontalTail.STRINGER_THICKNESS]
        rho_stringer, units = self.options[Aircraft.HorizontalTail.STRINGER_DENSITY]
        sheeting_thickness, units = self.options[Aircraft.HorizontalTail.SHEETING_THICKNESS]
        sheeting_coverage = self.options[Aircraft.HorizontalTail.SHEETING_COVERAGE]
        rho_sheeting, units = self.options[Aircraft.HorizontalTail.SHEETING_DENSITY]
        sheeting_lightening_factor = self.options[Aircraft.HorizontalTail.SHEETING_LIGHTENING_FACTOR]
        num_stringer = self.options[Aircraft.HorizontalTail.NUM_STRINGERS]
        rib_materials = self.options[Aircraft.HorizontalTail.RIB_MATERIALS]
        misc_mass, units = self.options[Aircraft.HorizontalTail.MISC_MASS]

        load_airfoil_if_needed(self, Aircraft.HorizontalTail)
        chord = root_chord

        cs_area = self.n_area * (chord**2) * rib_lightening_factor
        rho_rib = jnp.array([(materials.get_item(m)[0]) for m in rib_materials])

        rib_volumes = cs_area * rib_thickness
        spar_volume = (
            num_spars
            * span
            * jnp.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        )
        sheeting_volume = (
            wetted_area * sheeting_coverage * sheeting_lightening_factor * sheeting_thickness
        )
        stringer_volume = stringer_thickness**2 * num_stringer * span

        rib_mass = jnp.sum(rib_volumes * rho_rib)
        sheeting_mass = sheeting_volume * rho_sheeting
        stringer_mass = stringer_volume * rho_stringer
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        structural_mass = stringer_mass + sheeting_mass + rib_mass + spar_mass + skin_mass
        total_mass = (1 + glue_factor) * structural_mass + misc_mass

        return total_mass