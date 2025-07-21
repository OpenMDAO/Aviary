import numpy as np
import os

import openmdao.api as om
from openmdao.utils.cs_safe import abs as cs_abs

from aviary.examples.external_subsystems.dbf_based_mass.materials_database import materials
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.variable_meta_data import _MetaData


def make_units_option(var_key, units=None, default_val=None, desc=None, meta_data=_MetaData):
    meta = meta_data[var_key]

    default_units = meta['units']

    if units is None:
        units = default_units
    if desc is None:
        desc = meta['desc']
    if default_val is None:
        default_val = meta['default_value']

    return {
        'name': var_key,
        'default': (default_val, units),
        'set_function': lambda meta, val: (wrapped_convert_units(val, units), units),
        'desc': desc,
    }

class DBFHorizontalTailMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('airfoil_data_file', types=str, allow_none=False)
        self.options.declare('rib_materials', types=(list,))

        # Declare options using Aircraft.HorizontalTail.Dbf metadata keys
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.NUM_SPARS, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SPAR_OUTER_DIAMETER, 'm'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SPAR_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SPAR_WALL_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.RIB_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.RIB_LIGHTENING_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SKIN_DENSITY, 'kg/m**2'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.GLUE_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.STRINGER_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.STRINGER_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SHEETING_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SHEETING_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SHEETING_COVERAGE, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.SHEETING_LIGHTENING_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.NUM_STRINGERS, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.HorizontalTail.Dbf.MISC_MASS, 'kg'))

    def setup(self):
        # Still user inputs:
        add_aviary_input(self, Aircraft.HorizontalTail.SPAN, units='m')
        add_aviary_input(self, Aircraft.HorizontalTail.ROOT_CHORD, units='m')
        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, units='m**2')

        add_aviary_output(self, Aircraft.HorizontalTail.MASS, units='kg')

    def setup_partials(self):
        self.declare_partials(
            of=Aircraft.HorizontalTail.MASS,
            wrt=[
                Aircraft.HorizontalTail.SPAN,
                Aircraft.HorizontalTail.ROOT_CHORD,
                Aircraft.HorizontalTail.WETTED_AREA,
            ],
        )

    def set_option(self, option_key, val=None, units=None):
        """
        Helper to set an OpenMDAO component option with units.

        Parameters
        ----------
        comp : om.Component
            The component instance owning the option.
        option_key : str or key
            The option key (usually Aircraft.HorizontalTail.Dbf.*).
        val : any
            The value to set.
        units : str or None
            Optional units string. If None, attempts to get default units from metadata.
        """
        from aviary.variable_info.variable_meta_data import _MetaData

        # Get default units from metadata if none given
        if units is None and option_key in _MetaData:
            units = _MetaData[option_key]['units']
        if val is None and option_key in _MetaData:
            val = _MetaData[option_key]['default_value']

        self.options[option_key] = (val, units)


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
        span = inputs[Aircraft.HorizontalTail.SPAN]
        chord = inputs[Aircraft.HorizontalTail.ROOT_CHORD]
        wetted_area = inputs[Aircraft.HorizontalTail.WETTED_AREA]

        if span <= 0:
            raise ValueError(f'HorizontalTail span must be > 0, got {span}')
        if chord <= 0:
            raise ValueError(f'Root chord must be > 0, got {chord}')
        if wetted_area <= 0:
            raise ValueError(f'Wetted area must be > 0, got {wetted_area}')

        # From options
        num_spars = self.options[Aircraft.HorizontalTail.Dbf.NUM_SPARS][0]
        rib_lightening_factor = self.options[Aircraft.HorizontalTail.Dbf.RIB_LIGHTENING_FACTOR][0]
        rib_thickness = self.options[Aircraft.HorizontalTail.Dbf.RIB_THICKNESS][0]
        rho_skin = self.options[Aircraft.HorizontalTail.Dbf.SKIN_DENSITY][0]
        spar_outer_diameter = self.options[Aircraft.HorizontalTail.Dbf.SPAR_OUTER_DIAMETER][0]
        rho_spar = self.options[Aircraft.HorizontalTail.Dbf.SPAR_DENSITY][0]
        spar_wall_thickness = self.options[Aircraft.HorizontalTail.Dbf.SPAR_WALL_THICKNESS][0]
        glue_factor = self.options[Aircraft.HorizontalTail.Dbf.GLUE_FACTOR][0]
        stringer_thickness = self.options[Aircraft.HorizontalTail.Dbf.STRINGER_THICKNESS][0]
        rho_stringer = self.options[Aircraft.HorizontalTail.Dbf.STRINGER_DENSITY][0]
        sheeting_thickness = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_THICKNESS][0]
        sheeting_coverage = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_COVERAGE][0]
        rho_sheeting = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_DENSITY][0]
        sheeting_lightening_factor = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_LIGHTENING_FACTOR][0]
        num_stringer = self.options[Aircraft.HorizontalTail.Dbf.NUM_STRINGERS][0]
        rib_materials = self.options['rib_materials']  # stays string key
        airfoil_data_file = self.options['airfoil_data_file']  # stays string key
        misc_mass = self.options[Aircraft.HorizontalTail.Dbf.MISC_MASS][0]

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

        outputs[Aircraft.HorizontalTail.MASS] = total_mass

    def compute_partials(self, inputs, J):
        # From inputs
        chord = inputs[Aircraft.HorizontalTail.ROOT_CHORD]

        # From options
        num_spars = self.options[Aircraft.HorizontalTail.Dbf.NUM_SPARS][0]
        rib_lightening_factor = self.options[Aircraft.HorizontalTail.Dbf.RIB_LIGHTENING_FACTOR][0]
        rib_thickness = self.options[Aircraft.HorizontalTail.Dbf.RIB_THICKNESS][0]
        rho_skin = self.options[Aircraft.HorizontalTail.Dbf.SKIN_DENSITY][0]
        spar_outer_diameter = self.options[Aircraft.HorizontalTail.Dbf.SPAR_OUTER_DIAMETER][0]
        rho_spar = self.options[Aircraft.HorizontalTail.Dbf.SPAR_DENSITY][0]
        spar_wall_thickness = self.options[Aircraft.HorizontalTail.Dbf.SPAR_WALL_THICKNESS][0]
        glue_factor = self.options[Aircraft.HorizontalTail.Dbf.GLUE_FACTOR][0]
        stringer_thickness = self.options[Aircraft.HorizontalTail.Dbf.STRINGER_THICKNESS][0]
        rho_stringer = self.options[Aircraft.HorizontalTail.Dbf.STRINGER_DENSITY][0]
        sheeting_thickness = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_THICKNESS][0]
        sheeting_coverage = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_COVERAGE][0]
        rho_sheeting = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_DENSITY][0]
        sheeting_lightening_factor = self.options[Aircraft.HorizontalTail.Dbf.SHEETING_LIGHTENING_FACTOR][0]
        num_stringer = self.options[Aircraft.HorizontalTail.Dbf.NUM_STRINGERS][0]
        rib_materials = self.options['rib_materials']
        airfoil_data_file = self.options['airfoil_data_file']

        rho_rib = np.array([(materials.get_item(m)[0]) for m in rib_materials])

        x_coords, y_coords = self.load_airfoil_csv(airfoil_data_file, header=True)
        n_area = self.shoelace_area(x_coords, y_coords)

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.SPAN] = (
            num_stringer * rho_stringer * stringer_thickness**2
            + num_spars
            * rho_spar
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        ) * (1 + glue_factor)

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.WETTED_AREA] = (
            rho_skin
            + (sheeting_coverage * sheeting_lightening_factor * sheeting_thickness * rho_sheeting)
        ) * (1 + glue_factor)

        J[Aircraft.HorizontalTail.MASS, Aircraft.HorizontalTail.ROOT_CHORD] = (
            2 * chord * rib_lightening_factor * n_area * np.sum(rib_thickness * rho_rib)
        ) * (1 + glue_factor)


if __name__ == '__main__':
    prob = om.Problem()

    prob.model.add_subsystem(
        'dbf_htail', DBFHorizontalTailMass(), promotes_inputs=['*'], promotes_outputs=['*']
    )

    ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
    rib_thicks = np.where(ribs != 0, 0.125, 0.125)

    # Set required options
    htail = prob.model.dbf_htail
    htail.options['rib_materials'] = rib_materials
    htail.options['airfoil_data_file'] = (
        r'aviary\examples\external_subsystems\dbf_based_mass\mh84-il.csv'
    )
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_COVERAGE, val=0.4, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_LIGHTENING_FACTOR, val=1, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    htail.set_option(Aircraft.HorizontalTail.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.NUM_STRINGERS, val=2.5, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.GLUE_FACTOR, val=0.15, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.NUM_SPARS, val=1.1, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.RIB_LIGHTENING_FACTOR, val=2/3, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.RIB_THICKNESS, val=rib_thicks, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SPAR_DENSITY, val=2, units='g/cm**3')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SPAR_OUTER_DIAMETER, val=1, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SPAR_WALL_THICKNESS, val=0.0625, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.MISC_MASS, val=0.0, units='kg')

    # Setup problem with constant above options
    prob.setup()

    # Set values for aero-driving variables
    prob.set_val(Aircraft.HorizontalTail.ROOT_CHORD, val=20, units='inch')
    prob.set_val(Aircraft.HorizontalTail.SPAN, val=4.667, units='ft')
    prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, val=0.85, units='m**2')

    prob.run_model()

    total_mass = prob.get_val(Aircraft.HorizontalTail.MASS)
    print(f'Total mass of the dbf htail: {float(total_mass):.3f} kg')
