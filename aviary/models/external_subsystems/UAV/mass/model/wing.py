import numpy as np
import openmdao.api as om
import jax.numpy as jnp

from aviary.models.external_subsystems.UAV.mass.utils.UAV_enums import WingType
from aviary.models.external_subsystems.UAV.mass.utils.materials_database import materials
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.models.external_subsystems.UAV.mass.utils.airfoil_input import load_airfoil_csv
from aviary.models.external_subsystems.UAV.mass.utils.hashable_statics import hashable

from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import (
    ExtendedMetaData,
)
from aviary.utils.functions import get_path


class WingMass(om.ExplicitComponent):
    def initialize(self):
        # simple wing options
        add_aviary_option(self, Aircraft.Wing.TYPE, units='unitless', meta_data=ExtendedMetaData)
        add_aviary_option(
            self, Aircraft.Wing.FOAM_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.ROD_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData
        )
        add_aviary_option(self, Aircraft.Wing.ROD_RADIUS, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(self, Aircraft.Wing.ROD_THICKNESS, units='m', meta_data=ExtendedMetaData)

        # medium wing options
        add_aviary_option(
            self, Aircraft.Wing.AIRFOIL_PATH, units='unitless', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.RIB_MATERIALS, units='unitless', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.NUM_SPARS, units='unitless', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.SPAR_OUTER_DIAMETER, units='m', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.SPAR_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.SPAR_WALL_THICKNESS, units='m', meta_data=ExtendedMetaData
        )
        add_aviary_option(self, Aircraft.Wing.RIB_THICKNESS, units='m', meta_data=ExtendedMetaData)
        add_aviary_option(
            self, Aircraft.Wing.RIB_LIGHTENING_FACTOR, units='unitless', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.AREAL_SKIN_DENSITY, units='kg/m**2', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.GLUE_FACTOR, units='unitless', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.STRINGER_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.STRINGER_THICKNESS, units='m', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.SHEETING_THICKNESS, units='m', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.SHEETING_DENSITY, units='kg/m**3', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self, Aircraft.Wing.SHEETING_COVERAGE, units='unitless', meta_data=ExtendedMetaData
        )
        add_aviary_option(
            self,
            Aircraft.Wing.SHEETING_LIGHTENING_FACTOR,
            units='unitless',
            meta_data=ExtendedMetaData,
        )
        add_aviary_option(
            self, Aircraft.Wing.NUM_STRINGERS, units='unitless', meta_data=ExtendedMetaData
        )

    def setup(self):
        add_aviary_input(
            self, Aircraft.Wing.SPAN, units='m', meta_data=ExtendedMetaData, primal_name='span'
        )
        add_aviary_input(
            self,
            Aircraft.Wing.ROOT_CHORD,
            units='m',
            meta_data=ExtendedMetaData,
            primal_name='root_chord',
        )
        add_aviary_input(
            self,
            Aircraft.Wing.MISC_MASS,
            units='kg',
            meta_data=ExtendedMetaData
        )

        add_aviary_output(
            self, Aircraft.Wing.MASS, units='kg', meta_data=ExtendedMetaData, primal_name='mass'
        )

        # Pull n_area from the airfoil csv
        path = get_path(self.options[Aircraft.Wing.AIRFOIL_PATH])
        x, y = load_airfoil_csv(path, header=True)
        self.n_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if self.options[Aircraft.Wing.TYPE] == WingType.MEDIUM:
            rib_materials = self.options[Aircraft.Wing.RIB_MATERIALS]
            self.rho_rib = np.array([materials.get_item(m)[0] for m in rib_materials])
        else:
            self.rho_rib = np.array([0])

    def setup_partials(self):
        # primal_name mismatch breaks jax dependency inference; declare explicitly
        self.declare_partials(Aircraft.Wing.MASS, '*', method='cs')

    def compute(self, inputs, outputs):
        span = inputs[Aircraft.Wing.SPAN]
        chord = inputs[Aircraft.Wing.ROOT_CHORD]

        wing_type = self.options[Aircraft.Wing.TYPE]

        if wing_type == WingType.SIMPLE:
            # Simple wing design mass calculation
            rod_thickness, units = self.options[Aircraft.Wing.ROD_THICKNESS]
            foam_density, units = self.options[Aircraft.Wing.FOAM_DENSITY]
            radius, units = self.options[Aircraft.Wing.ROD_RADIUS]
            rod_density, units = self.options[Aircraft.Wing.ROD_DENSITY]

            airfoil_area = self.n_area * chord**2
            foam_volume = airfoil_area * span
            foam_mass_prelim = foam_volume * foam_density
            cross_section_rod_area = jnp.pi * radius**2 - jnp.pi * (radius - rod_thickness) ** 2

            # Assumes fixed number of 2 rods in the simple wing design:
            rod_volume = 2.0 * span * cross_section_rod_area

            foam_mass_final = foam_mass_prelim - 2 * foam_density * rod_volume
            rod_mass = 2 * rod_volume * rod_density

            total_mass = foam_mass_final + rod_mass

        elif wing_type == WingType.MEDIUM:
            # medium wing design mass calculation
            num_spars = self.options[Aircraft.Wing.NUM_SPARS]
            rib_lightening_factor = self.options[Aircraft.Wing.RIB_LIGHTENING_FACTOR]
            rib_thickness, units = self.options[Aircraft.Wing.RIB_THICKNESS]
            rib_thickness = jnp.asarray(rib_thickness)
            rho_skin, units = self.options[Aircraft.Wing.AREAL_SKIN_DENSITY]
            spar_outer_diameter, units = self.options[Aircraft.Wing.SPAR_OUTER_DIAMETER]
            rho_spar, units = self.options[Aircraft.Wing.SPAR_DENSITY]
            spar_wall_thickness, units = self.options[Aircraft.Wing.SPAR_WALL_THICKNESS]
            glue_factor = self.options[Aircraft.Wing.GLUE_FACTOR]
            stringer_thickness, units = self.options[Aircraft.Wing.STRINGER_THICKNESS]
            rho_stringer, units = self.options[Aircraft.Wing.STRINGER_DENSITY]
            sheeting_thickness, units = self.options[Aircraft.Wing.SHEETING_THICKNESS]
            sheeting_coverage = self.options[Aircraft.Wing.SHEETING_COVERAGE]
            rho_sheeting, units = self.options[Aircraft.Wing.SHEETING_DENSITY]
            sheeting_lightening_factor = self.options[Aircraft.Wing.SHEETING_LIGHTENING_FACTOR]
            num_stringer = self.options[Aircraft.Wing.NUM_STRINGERS]
            rib_materials = self.options[Aircraft.Wing.RIB_MATERIALS]

            misc_mass = inputs[Aircraft.Wing.MISC_MASS]

            # Wetted area is now derived from the same span x chord reference area the aero uses,
            # so span/chord drive the skin & sheeting mass terms too (was a separate input/DV).
            wetted_area = span * chord

            if len(rib_materials) != len(rib_thickness):
                raise ValueError(
                    'Mismatch in number of rib materials vs. number of rib thicknesses'
                )

            cs_area = self.n_area * (chord**2) * rib_lightening_factor
            rho_rib = self.rho_rib.reshape(-1)

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

        outputs[Aircraft.Wing.MASS] = total_mass
