import numpy as np
import os

import openmdao.api as om
from openmdao.utils.cs_safe import abs as cs_abs

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


class DBFVerticalTailMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('airfoil_data_file', types=str, allow_none=False)
        self.options.declare('rib_materials', types=(list,))

        # Options with unit conversion
        self.options.declare(**make_units_option('num_spars', 1.0, 'unitless'))
        self.options.declare(**make_units_option('spar_outer_diameter', 1.0, 'm'))
        self.options.declare(**make_units_option('spar_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('spar_wall_thickness', 0.01, 'm'))
        self.options.declare(**make_units_option('rib_thicknesses', np.zeros(1), 'm'))
        self.options.declare(**make_units_option('rib_lightening_factor', 2 / 3, 'unitless'))
        self.options.declare(**make_units_option('skin_density', 20.0, 'kg/m**2'))
        self.options.declare(**make_units_option('glue_factor', 0.15, 'unitless'))
        self.options.declare(**make_units_option('stringer_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('stringer_thickness', 0.01, 'm'))
        self.options.declare(**make_units_option('sheeting_thickness', 0.01, 'm'))
        self.options.declare(**make_units_option('sheeting_density', 160.0, 'kg/m**3'))
        self.options.declare(**make_units_option('sheeting_coverage', 0.4, 'unitless'))
        self.options.declare(**make_units_option('sheeting_lightening_factor', 1.0, 'unitless'))
        self.options.declare(**make_units_option('num_stringers', 1.0, 'unitless'))
        self.options.declare(**make_units_option('misc_mass', 0.0, 'kg'))

    def setup(self):
        # Still user inputs:
        add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='m')
        add_aviary_input(self, Aircraft.VerticalTail.ROOT_CHORD, units='m')
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, units='m**2')

        add_aviary_output(self, Aircraft.VerticalTail.MASS, units='kg')

    def setup_partials(self):
        self.declare_partials(
            of=Aircraft.VerticalTail.MASS,
            wrt=[
                Aircraft.VerticalTail.SPAN,
                Aircraft.VerticalTail.ROOT_CHORD,
                Aircraft.VerticalTail.WETTED_AREA,
            ],
        )

    def load_airfoil_csv(self, file_path, delimiter=',', header=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Airfoil CSV file '{file_path}' not found.")

        skip = 1 if header else 0
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip)

        if data.shape[1] < 2:
            raise ValueError('CSV must contain at least two columns for x and y coordinates.')

        x = data[:, 0]
        y = data[:, 1]

        x_min = np.min(x)
        x_max = np.max(x)
        chord_length = x_max - x_min

        if chord_length <= 0:
            raise ValueError('Invalid airfoil: chord length must be > 0.')

        x_normalized = (x - x_min) / chord_length
        y_normalized = y / chord_length

        return x_normalized, y_normalized

    def shoelace_area(self, x, y):
        return 0.5 * cs_abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        # From inputs
        span = inputs[Aircraft.VerticalTail.SPAN]
        chord = inputs[Aircraft.VerticalTail.ROOT_CHORD]
        wetted_area = inputs[Aircraft.VerticalTail.WETTED_AREA]

        if span <= 0:
            raise ValueError(f'VerticalTail span must be > 0, got {span}')
        if chord <= 0:
            raise ValueError(f'Root chord must be > 0, got {chord}')
        if wetted_area <= 0:
            raise ValueError(f'Wetted area must be > 0, got {wetted_area}')

        # From options
        num_spars = self.options['num_spars'][0]
        rib_lightening_factor = self.options['rib_lightening_factor'][0]
        rib_thickness = self.options['rib_thicknesses'][0]
        rho_skin = self.options['skin_density'][0]
        spar_outer_diameter = self.options['spar_outer_diameter'][0]
        rho_spar = self.options['spar_density'][0]
        spar_wall_thickness = self.options['spar_wall_thickness'][0]
        glue_factor = self.options['glue_factor'][0]
        stringer_thickness = self.options['stringer_thickness'][0]
        rho_stringer = self.options['stringer_density'][0]
        sheeting_thickness = self.options['sheeting_thickness'][0]
        sheeting_coverage = self.options['sheeting_coverage'][0]
        rho_sheeting = self.options['sheeting_density'][0]
        sheeting_lightening_factor = self.options['sheeting_lightening_factor'][0]
        num_stringer = self.options['num_stringers'][0]
        rib_materials = self.options['rib_materials']
        airfoil_data_file = self.options['airfoil_data_file']
        misc_mass = self.options['misc_mass'][0]

        if len(rib_materials) != len(rib_thickness):
            raise ValueError(
                f'Length mismatch: {len(rib_materials)} rib_materials vs '
                f'{len(rib_thickness)} rib_thicknesses. These must match.'
            )

        rho_rib = np.array([(materials.get_item(m)[0]) for m in rib_materials])

        x_coords, y_coords = self.load_airfoil_csv(airfoil_data_file, header=True)
        n_area = self.shoelace_area(x_coords, y_coords)

        if n_area < 0.01:
            raise ValueError(
                f'Computed normalized airfoil area is suspiciously small: {n_area:.5f}'
            )

        cs_area = n_area * (chord**2) * rib_lightening_factor

        rib_volumes = cs_area * rib_thickness
        spar_volume = (
            num_spars
            * span
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        )
        sheeting_volume = (
            wetted_area * sheeting_coverage * sheeting_lightening_factor * sheeting_thickness
        )
        stringer_volume = stringer_thickness**2 * num_stringer * span

        rib_mass = np.sum(rib_volumes * rho_rib)
        sheeting_mass = sheeting_volume * rho_sheeting
        stringer_mass = stringer_volume * rho_stringer
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        structural_mass = stringer_mass + sheeting_mass + rib_mass + spar_mass + skin_mass
        total_mass = (1 + glue_factor) * structural_mass + misc_mass

        outputs[Aircraft.VerticalTail.MASS] = total_mass

    def compute_partials(self, inputs, J):
        # From inputs
        chord = inputs[Aircraft.VerticalTail.ROOT_CHORD]

        # From options
        num_spars = self.options['num_spars'][0]
        rib_lightening_factor = self.options['rib_lightening_factor'][0]
        rib_thickness = self.options['rib_thicknesses'][0]
        rho_skin = self.options['skin_density'][0]
        spar_outer_diameter = self.options['spar_outer_diameter'][0]
        rho_spar = self.options['spar_density'][0]
        spar_wall_thickness = self.options['spar_wall_thickness'][0]
        glue_factor = self.options['glue_factor'][0]
        stringer_thickness = self.options['stringer_thickness'][0]
        rho_stringer = self.options['stringer_density'][0]
        sheeting_thickness = self.options['sheeting_thickness'][0]
        sheeting_coverage = self.options['sheeting_coverage'][0]
        rho_sheeting = self.options['sheeting_density'][0]
        sheeting_lightening_factor = self.options['sheeting_lightening_factor'][0]
        num_stringer = self.options['num_stringers'][0]
        rib_materials = self.options['rib_materials']
        airfoil_data_file = self.options['airfoil_data_file']

        rho_rib = np.array([(materials.get_item(m)[0]) for m in rib_materials])

        x_coords, y_coords = self.load_airfoil_csv(airfoil_data_file, header=True)
        n_area = self.shoelace_area(x_coords, y_coords)

        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.SPAN] = (
            num_stringer * rho_stringer * stringer_thickness**2
            + num_spars
            * rho_spar
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        ) * (1 + glue_factor)

        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.WETTED_AREA] = (
            rho_skin
            + (sheeting_coverage * sheeting_lightening_factor * sheeting_thickness * rho_sheeting)
        ) * (1 + glue_factor)

        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.ROOT_CHORD] = (
            2 * chord * rib_lightening_factor * n_area * np.sum(rib_thickness * rho_rib)
        ) * (1 + glue_factor)


if __name__ == '__main__':
    prob = om.Problem()

    prob.model.add_subsystem(
        'dbf_vert_tail', DBFVerticalTailMass(), promotes_inputs=['*'], promotes_outputs=['*']
    )

    ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
    rib_thicks = np.where(ribs != 0, 0.125, 0.125)

    # Set required options
    vert_tail = prob.model.dbf_vert_tail
    vert_tail.options['rib_materials'] = rib_materials
    vert_tail.options['airfoil_data_file'] = (
        r'aviary\examples\external_subsystems\dbf_based_mass\mh84-il.csv'
    )
    vert_tail.options['sheeting_coverage'] = (0.4, 'unitless')
    vert_tail.options['sheeting_density'] = (160, 'kg/m**3')
    vert_tail.options['sheeting_lightening_factor'] = (1, 'unitless')
    vert_tail.options['sheeting_thickness'] = (0.03125, 'inch')
    vert_tail.options['stringer_density'] = (160, 'kg/m**3')
    vert_tail.options['stringer_thickness'] = (0.375, 'inch')
    vert_tail.options['num_stringers'] = (2.5, 'unitless')
    vert_tail.options['glue_factor'] = (0.15, 'unitless')
    vert_tail.options['num_spars'] = (1.1, 'unitless')
    vert_tail.options['rib_lightening_factor'] = (2 / 3, 'unitless')
    vert_tail.options['rib_thicknesses'] = (rib_thicks, 'inch')
    vert_tail.options['skin_density'] = (20, 'g/m**2')
    vert_tail.options['spar_density'] = (2, 'g/cm**3')
    vert_tail.options['spar_outer_diameter'] = (1, 'inch')
    vert_tail.options['spar_wall_thickness'] = (0.0625, 'inch')
    vert_tail.options['misc_mass'] = (0.0, 'kg')

    # Setup problem with constant above options
    prob.setup()

    # Set values for aero-driving variables
    prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, val=20, units='inch')
    prob.set_val(Aircraft.VerticalTail.SPAN, val=4.667, units='ft')
    prob.set_val(Aircraft.VerticalTail.WETTED_AREA, val=0.85, units='m**2')

    prob.run_model()

    total_mass = prob.get_val(Aircraft.VerticalTail.MASS)
    print(f'Total mass of the dbf vertical tail: {float(total_mass):.3f} kg')
