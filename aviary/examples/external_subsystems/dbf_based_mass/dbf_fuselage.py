import numpy as np

import openmdao.api as om

from aviary.examples.external_subsystems.dbf_based_mass.materials_database import materials
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


def make_units_option(name, default_val, target_units, desc=None):
    return {
        'name': name,
        'default': (default_val, target_units),
        'set_function': lambda meta, val: (wrapped_convert_units(val, target_units), target_units),
        'desc': desc,
    }


class DBFFuselageMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('bulkhead_materials', types=(list,))

        # Options with unit conversion
        self.options.declare(**make_units_option('num_spars', 1.0, 'unitless'))
        self.options.declare(**make_units_option('spar_outer_diameter', 0.05, 'm'))
        self.options.declare(**make_units_option('spar_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('spar_wall_thickness', 0.005, 'm'))
        self.options.declare(**make_units_option('bulkhead_thicknesses', np.zeros(1), 'm'))
        self.options.declare(**make_units_option('bulkhead_lightening_factor', 2 / 3, 'unitless'))
        self.options.declare(**make_units_option('skin_density', 20.0, 'kg/m**2'))
        self.options.declare(**make_units_option('floor_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('floor_thickness', 0.01, 'm'))
        self.options.declare(**make_units_option('floor_length', 1.0, 'm'))
        self.options.declare(**make_units_option('glue_factor', 0.15, 'unitless'))
        self.options.declare(**make_units_option('stringer_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('stringer_thickness', 0.01, 'm'))
        self.options.declare(**make_units_option('sheeting_thickness', 0.01, 'm'))
        self.options.declare(**make_units_option('sheeting_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('sheeting_coverage', 0.4, 'unitless'))
        self.options.declare(**make_units_option('sheeting_lightening_factor', 1.0, 'unitless'))
        self.options.declare(**make_units_option('misc_mass', 0.0, 'kg'))

    def setup(self):
        # Required geometry inputs
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='m')
        add_aviary_input(self, Aircraft.Fuselage.AVG_HEIGHT, units='m')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='m**2')
        add_aviary_input(self, Aircraft.Fuselage.AVG_WIDTH, units='m')

        add_aviary_output(
            self,
            Aircraft.Fuselage.MASS,
            units='kg',
        )

    def setup_partials(self):
        self.declare_partials(
            of=Aircraft.Fuselage.MASS,
            wrt=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.AVG_HEIGHT,
                Aircraft.Fuselage.WETTED_AREA,
                Aircraft.Fuselage.AVG_WIDTH,
            ],
        )

    def compute(self, inputs, outputs):
        # Inputs
        length = inputs[Aircraft.Fuselage.LENGTH]
        height = inputs[Aircraft.Fuselage.AVG_HEIGHT]
        width = inputs[Aircraft.Fuselage.AVG_WIDTH]
        wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]

        # From options
        num_spars = self.options['num_spars'][0]
        spar_outer_diameter = self.options['spar_outer_diameter'][0]
        rho_spar = self.options['spar_density'][0]
        spar_wall_thickness = self.options['spar_wall_thickness'][0]
        bulkhead_thickness = self.options['bulkhead_thicknesses'][0]
        bulkhead_lightening_factor = self.options['bulkhead_lightening_factor'][0]
        rho_skin = self.options['skin_density'][0]
        rho_floor = self.options['floor_density'][0]
        floor_thickness = self.options['floor_thickness'][0]
        floor_length = self.options['floor_length'][0]
        glue_factor = self.options['glue_factor'][0]
        stringer_thickness = self.options['stringer_thickness'][0]
        rho_stringer = self.options['stringer_density'][0]
        sheeting_thick = self.options['sheeting_thickness'][0]
        sheeting_coverage = self.options['sheeting_coverage'][0]
        rho_sheeting = self.options['sheeting_density'][0]
        sheeting_lightening_factor = self.options['sheeting_lightening_factor'][0]
        bulkhead_materials = self.options['bulkhead_materials']
        misc_mass = self.options['misc_mass'][0]

        rho_rib = np.array([(materials.get_item(m)[0]) for m in bulkhead_materials])
        cs_area = width * height * bulkhead_lightening_factor
        rib_volumes = cs_area * bulkhead_thickness
        rib_mass = np.sum(rib_volumes * rho_rib)

        # Spar volume
        spar_volume = (
            num_spars
            * length
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        )

        # Other volumes
        sheeting_volume = (
            wetted_area * sheeting_coverage * sheeting_lightening_factor * sheeting_thick
        )
        stringer_volume = 4 * stringer_thickness**2 * (length + width + height)

        # Mass calculations
        sheeting_mass = sheeting_volume * rho_sheeting
        stringer_mass = stringer_volume * rho_stringer
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area
        floor_mass = rho_floor * floor_length * width * floor_thickness

        # Total structural mass
        structural_mass = (
            rib_mass + spar_mass + floor_mass + stringer_mass + sheeting_mass + skin_mass
        )

        total_mass = structural_mass * (1 + glue_factor) + misc_mass

        outputs[Aircraft.Fuselage.MASS] = total_mass

    def compute_partials(self, inputs, J):
        # Inputs
        height = inputs[Aircraft.Fuselage.AVG_HEIGHT]
        width = inputs[Aircraft.Fuselage.AVG_WIDTH]

        # From options
        num_spars = self.options['num_spars'][0]
        spar_outer_diameter = self.options['spar_outer_diameter'][0]
        rho_spar = self.options['spar_density'][0]
        spar_wall_thickness = self.options['spar_wall_thickness'][0]
        bulkhead_thickness = self.options['bulkhead_thicknesses'][0]
        bulkhead_lightening_factor = self.options['bulkhead_lightening_factor'][0]
        rho_skin = self.options['skin_density'][0]
        rho_floor = self.options['floor_density'][0]
        floor_thickness = self.options['floor_thickness'][0]
        floor_length = self.options['floor_length'][0]
        glue_factor = self.options['glue_factor'][0]
        stringer_thickness = self.options['stringer_thickness'][0]
        rho_stringer = self.options['stringer_density'][0]
        sheeting_thick = self.options['sheeting_thickness'][0]
        sheeting_coverage = self.options['sheeting_coverage'][0]
        rho_sheeting = self.options['sheeting_density'][0]
        sheeting_lightening_factor = self.options['sheeting_lightening_factor'][0]
        bulkhead_materials = self.options['bulkhead_materials']

        rho_rib = np.array([(materials.get_item(m)[0]) for m in bulkhead_materials])

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_HEIGHT] = (
            4 * stringer_thickness**2 * rho_stringer
            + width * bulkhead_lightening_factor * np.sum(rho_rib * bulkhead_thickness)
        ) * (1 + glue_factor)

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_WIDTH] = (
            4 * stringer_thickness**2 * rho_stringer
            + floor_length * floor_thickness * rho_floor
            + height * bulkhead_lightening_factor * np.sum(rho_rib * bulkhead_thickness)
        ) * (1 + glue_factor)

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.LENGTH] = (
            num_spars
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
            * rho_spar
            + 4 * stringer_thickness**2 * rho_stringer
        ) * (1 + glue_factor)

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = (
            rho_skin
            + sheeting_coverage * sheeting_lightening_factor * sheeting_thick * rho_sheeting
        ) * (1 + glue_factor)


if __name__ == '__main__':
    prob = om.Problem()

    # === Prepare values before setup ===
    ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])
    bulkhead_materials = np.where(ribs != 0, 'Ply', 'Balsa').tolist()
    rib_thicks = np.where(ribs == 2, 0.25, 0.125)

    fuse = DBFFuselageMass()
    fuse.options['bulkhead_materials'] = bulkhead_materials
    fuse.options['bulkhead_thicknesses'] = (rib_thicks, 'inch')
    fuse.options['num_spars'] = (0.5, 'unitless')
    fuse.options['bulkhead_lightening_factor'] = (0.18, 'unitless')
    fuse.options['sheeting_coverage'] = (1, 'unitless')
    fuse.options['sheeting_density'] = (160, 'kg/m**3')
    fuse.options['sheeting_lightening_factor'] = (0.3, 'unitless')
    fuse.options['sheeting_thickness'] = (0.03125, 'inch')
    fuse.options['glue_factor'] = (0.08, 'unitless')
    fuse.options['stringer_density'] = (160, 'kg/m**3')
    fuse.options['stringer_thickness'] = (0.375, 'inch')
    fuse.options['floor_length'] = (2, 'ft')
    fuse.options['floor_density'] = (340, 'kg/m**3')
    fuse.options['floor_thickness'] = (0.125, 'inch')
    fuse.options['skin_density'] = (20, 'g/m**2')
    fuse.options['spar_density'] = (2, 'g/cm**3')
    fuse.options['spar_outer_diameter'] = (1, 'inch')
    fuse.options['spar_wall_thickness'] = (0.0625, 'inch')
    fuse.options['misc_mass'] = (0.0, 'kg')

    prob.model.add_subsystem('dbf_fuselage', fuse, promotes_inputs=['*'], promotes_outputs=['*'])
    prob.setup()

    prob.set_val(Aircraft.Fuselage.LENGTH, val=4, units='ft')
    prob.set_val(Aircraft.Fuselage.AVG_HEIGHT, val=5, units='inch')
    prob.set_val(Aircraft.Fuselage.AVG_WIDTH, val=4, units='inch')
    prob.set_val(Aircraft.Fuselage.WETTED_AREA, val=904, units='inch**2')

    prob.run_model()

    total_mass = prob.get_val(Aircraft.Fuselage.MASS)
    print(f'Total mass of the dbf fuselage: {float(total_mass):.3f} kg')
